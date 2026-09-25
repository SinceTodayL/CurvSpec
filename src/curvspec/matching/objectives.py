import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from curvspec.geometry import lorentz_ops as L
from curvspec.representation.blocks import clip_nce


class query_diverse_loss(nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mrg = config['neg_factor'][0]
        self.alpha = config['neg_factor'][1]
        self.lamda = config['neg_factor'][2]
        
    def forward(self, x, label_dict):

        bs = x.shape[0]
        x = F.normalize(x, dim=-1)
        cos = torch.matmul(x, x.t())

        N_one_hot = torch.zeros((bs, bs))
        for i, label in label_dict.items():
            N_one_hot[label[0]:(label[-1]+1), label[0]:(label[-1]+1)] = torch.ones((len(label), len(label)))
        N_one_hot = N_one_hot - torch.eye(bs)
        N_one_hot = N_one_hot.to(x.device)
    
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp))
        focal = torch.where(N_one_hot == 1, cos, torch.zeros_like(cos))
    
        neg_term = (((1 + focal) ** self.lamda) * torch.log(1 + N_sim_sum)).sum(dim=0).sum() / bs
        
        return neg_term


class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        focal_gamma = cfg.get('nce_focal_gamma', 0.0)
        self.clip_nce_criterion = clip_nce(reduction='mean', focal_gamma=focal_gamma)
        self.video_nce_criterion = clip_nce(reduction='mean', focal_gamma=focal_gamma)

        self.qdl = query_diverse_loss(cfg)
        self.tau = cfg['tau'] if 'tau' in cfg else 5.0
        self.cov_eta = cfg['eta'] if 'eta' in cfg else 0.6
        self.cov_delta = cfg['delta'] if 'delta' in cfg else 0.2
        self.lambda_cov = cfg['lambda_cov'] if 'lambda_cov' in cfg else 0.1
        self.topk_centroids = cfg['topk_centroids'] if 'topk_centroids' in cfg else 3
        self.topk_segments = cfg['topk_segments'] if 'topk_segments' in cfg else 5
        self.lambda_exclusive = cfg.get('lambda_exclusive', 0.05)
        self.learnable_loss_weights = cfg.get('learnable_loss_weights', False)
        if self.learnable_loss_weights:
            self.loss_factor_clip_nce_raw = nn.Parameter(self._inv_softplus(cfg['loss_factor'][0]))
            self.loss_factor_frame_nce_raw = nn.Parameter(self._inv_softplus(cfg['loss_factor'][1]))
            self.loss_factor_div_raw = nn.Parameter(self._inv_softplus(cfg['loss_factor'][2]))
            self.loss_factor_hyp_raw = nn.Parameter(self._inv_softplus(cfg['loss_factor_hyp']))
            self.curvature_diversity_weight_raw = nn.Parameter(self._inv_softplus(cfg['curvature_diversity_weight']))
            self.lambda_cov_raw = nn.Parameter(self._inv_softplus(cfg['lambda_cov']))
            self.lambda_exclusive_raw = nn.Parameter(self._inv_softplus(cfg.get('lambda_exclusive', 0.05)))

    @staticmethod
    def _inv_softplus(val):
        val = max(float(val), 1e-8)
        return torch.log(torch.exp(torch.tensor(val, dtype=torch.float)) - 1.0)

    @staticmethod
    def _softplus(val):
        return F.softplus(val)

    def get_loss_weights(self):
        if self.learnable_loss_weights:
            return {
                'loss_factor': [
                    self._softplus(self.loss_factor_clip_nce_raw).item(),
                    self._softplus(self.loss_factor_frame_nce_raw).item(),
                    self._softplus(self.loss_factor_div_raw).item(),
                ],
                'loss_factor_hyp': self._softplus(self.loss_factor_hyp_raw).item(),
                'curvature_diversity_weight': self._softplus(self.curvature_diversity_weight_raw).item(),
                'lambda_cov': self._softplus(self.lambda_cov_raw).item(),
                'lambda_exclusive': self._softplus(self.lambda_exclusive_raw).item(),
            }
        return {
            'loss_factor': self.cfg.get('loss_factor'),
            'loss_factor_hyp': self.cfg.get('loss_factor_hyp'),
            'curvature_diversity_weight': self.cfg.get('curvature_diversity_weight'),
            'lambda_cov': self.cfg.get('lambda_cov'),
            'lambda_exclusive': self.cfg.get('lambda_exclusive'),
        }

    def forward(self, input_list, batch):

        query_labels = batch['text_labels']
        
        clip_scale_scores = input_list[0]
        clip_scale_scores_ = input_list[1]
        label_dict = input_list[2]
        frame_scale_scores = input_list[3]
        frame_scale_scores_ = input_list[4]
        all_curvatures        = input_list[5]
        text_query_hyp        = input_list[6]
        scale_video_feat      = input_list[7]
        _curv                 = input_list[8]
        query = input_list[9]
        video_regions = input_list[10]
        frame_attn_weights = input_list[11]
        clip_attn_weights = input_list[12]
        videos_mask = input_list[13]
        ot_loss = input_list[14] if len(input_list) > 14 else torch.tensor(0.0, device=clip_scale_scores.device)
       
        if self.learnable_loss_weights:
            loss_factor_clip_nce = self._softplus(self.loss_factor_clip_nce_raw)
            loss_factor_frame_nce = self._softplus(self.loss_factor_frame_nce_raw)
            loss_factor_div = self._softplus(self.loss_factor_div_raw)
            loss_factor_hyp = self._softplus(self.loss_factor_hyp_raw)
            curvature_diversity_weight = self._softplus(self.curvature_diversity_weight_raw)
            lambda_cov = self._softplus(self.lambda_cov_raw)
            lambda_exclusive = self._softplus(self.lambda_exclusive_raw)
        else:
            loss_factor_clip_nce = self.cfg['loss_factor'][0]
            loss_factor_frame_nce = self.cfg['loss_factor'][1]
            loss_factor_div = self.cfg['loss_factor'][2]
            loss_factor_hyp = self.cfg['loss_factor_hyp']
            curvature_diversity_weight = self.cfg['curvature_diversity_weight']
            lambda_cov = self.cfg['lambda_cov']
            lambda_exclusive = self.cfg.get('lambda_exclusive', 0.05)

        if self.cfg.get('use_adaptive_fusion', False):
            loss_factor_nce = 0.5 * (loss_factor_clip_nce + loss_factor_frame_nce)
            clip_nce_loss = loss_factor_nce * self.clip_nce_criterion(query_labels, label_dict, frame_scale_scores_)
            clip_trip_loss = self.get_clip_triplet_loss(frame_scale_scores, query_labels)
            frame_nce_loss = frame_scale_scores_.sum() * 0.0
            frame_trip_loss = frame_scale_scores.sum() * 0.0
        else:
            clip_nce_loss = loss_factor_clip_nce * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
            clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)

            frame_nce_loss = loss_factor_frame_nce * self.video_nce_criterion(query_labels, label_dict, frame_scale_scores_)
            frame_trip_loss = self.get_clip_triplet_loss(frame_scale_scores, query_labels)

        loss_sim = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss
        loss_div = 0.0
        loss_hyp = 0.0
        loss_curv_reg = 0.0
        cov_loss = 0.0
        exclusive_loss = 0.0

        if self.cfg.get('enable_loss_hyp', True):
            if _curv is not None and text_query_hyp is not None:
                match_video_feat = scale_video_feat[query_labels]
                hyp_dist = L.pairwise_dist(text_query_hyp, match_video_feat, _curv)
                hyp_logits = -hyp_dist * self.cfg['logit_scale_hyp']
                
                hyp_labels = torch.arange(len(hyp_logits)).to(hyp_logits.device)
                loss_hyp = F.cross_entropy(hyp_logits, hyp_labels) * loss_factor_hyp

        if self.cfg.get('enable_loss_curv_reg', True):
            if self.cfg['use_learnable_curvature'] and all_curvatures.numel() > 1:
                clip_curvs = all_curvatures[1:5]
                frame_curvs = all_curvatures[5:9]
                
                min_gap = self.cfg.get('curvature_min_gap', 0.15)
                max_gap = self.cfg.get('curvature_max_gap', 0.6)
                
                loss_curv_sep = 0
                for curvs in [clip_curvs, frame_curvs]:
                    for i in range(len(curvs)):
                        for j in range(i+1, len(curvs)):
                            gap = torch.abs(curvs[i] - curvs[j])
                            if gap < min_gap:
                                loss_curv_sep += (min_gap - gap) ** 2
                            elif gap > max_gap:
                                loss_curv_sep += (gap - max_gap) ** 2
                
                loss_curv_reg = loss_curv_sep * curvature_diversity_weight

        if self.cfg.get('enable_loss_div', True):
            loss_div = loss_factor_div * self.qdl(query, label_dict)

        if self.cfg.get('enable_loss_cov', True):
            if _curv is not None and video_regions is not None and video_regions.shape[0] > 1:
                B_q = text_query_hyp.shape[0]
                B_v = video_regions.shape[0]
                K = video_regions.shape[1]
                
                cover = torch.zeros(B_q, B_v, device=video_regions.device)
                
                chunk_size_q = 32
                for q_start in range(0, B_q, chunk_size_q):
                    q_end = min(q_start + chunk_size_q, B_q)
                    text_chunk = text_query_hyp[q_start:q_end]
                    
                    chunk_size_v = 16
                    for v_start in range(0, B_v, chunk_size_v):
                        v_end = min(v_start + chunk_size_v, B_v)
                        regions_chunk = video_regions[v_start:v_end]
                        
                        B_q_chunk = q_end - q_start
                        B_v_chunk = v_end - v_start
                        
                        text_expanded = text_chunk.unsqueeze(1).unsqueeze(2)
                        regions_expanded = regions_chunk.unsqueeze(0)
                        
                        dists = L.dist(
                            text_expanded.expand(B_q_chunk, B_v_chunk, K, -1).reshape(-1, text_chunk.shape[-1]),
                            regions_expanded.expand(B_q_chunk, B_v_chunk, K, -1).reshape(-1, regions_chunk.shape[-1]),
                            _curv
                        )
                        dists = dists.view(B_q_chunk, B_v_chunk, K)
                        
                        region_dist = dists.min(dim=-1)[0]
                        
                        cover[q_start:q_end, v_start:v_end] = region_dist
                
                diag_indices = torch.arange(B_q, device=cover.device)
                dist_pos = cover[diag_indices, query_labels]
                
                cover_neg_mask = torch.ones_like(cover)
                cover_neg_mask[diag_indices, query_labels] = 0
                
                neg_dists = cover[cover_neg_mask.bool()].view(B_q, B_v - 1)
                
                logits = -torch.cat([dist_pos.unsqueeze(1), neg_dists], dim=1)
                
                labels = torch.zeros(B_q, dtype=torch.long, device=cover.device)
                cov_loss = lambda_cov * F.cross_entropy(logits / self.tau, labels)
        
        if self.cfg.get('enable_loss_exclusive', True):
            exclusive_loss = lambda_exclusive * ot_loss

        loss_agg = loss_sim + loss_div + loss_hyp + loss_curv_reg + cov_loss + exclusive_loss

        if self.cfg.get('log_loss_components', False):
            loss_items = {
                'loss_sim': loss_sim,
                'loss_div': loss_div,
                'loss_hyp': loss_hyp,
                'loss_curv_reg': loss_curv_reg,
                'loss_cov': cov_loss,
                'loss_exclusive': exclusive_loss,
            }
            return loss_agg, loss_items

        return loss_agg


    def get_clip_triplet_loss(self, query_context_scores, labels):
        if query_context_scores.shape[1] < 2:
            return query_context_scores.sum() * 0.0

        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.cfg['use_hard_negative']:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.cfg['hard_pool_size'],
                                 t2v_scores.shape[1]) if self.cfg['use_hard_negative'] else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def get_frame_trip_loss(self, query_context_scores):

        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q

    def get_neg_scores(self, scores, scores_masked):

        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1

        sample_max_idx = min(sample_min_idx + self.cfg['hard_pool_size'], bsz) if self.cfg['use_hard_negative'] else bsz

        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        return torch.clamp(self.cfg['margin'] + neg_score - pos_score, min=0).sum() / len(pos_score)
