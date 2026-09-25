import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from curvspec.config import AttributeDict
from curvspec.representation.blocks import CurvSpecBlock, EuclideanAttentionBlock, LinearLayer, \
                                            TrainablePositionalEncoding, ConstraintHead, VideoRegionDecomposition

from curvspec.geometry import lorentz_ops as L
import math
from curvspec.geometry.manifold import Lorentz


class CurvSpec_Net(nn.Module):
    def __init__(self, config):
        super(CurvSpec_Net, self).__init__()
        self.config = config
        
        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,
                                                           hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                         hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,
                                                          hidden_size=config.hidden_size, dropout=config.input_drop)
        
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        self.query_encoder = EuclideanAttentionBlock(AttributeDict(hidden_size=config.hidden_size, 
                                                            intermediate_size=config.hidden_size,
                                                            hidden_dropout_prob=config.drop, 
                                                            num_attention_heads=config.n_heads,
                                                            attention_probs_dropout_prob=config.drop))
        
        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                            dropout=config.input_drop, relu=True)
        
        num_lorentz_blocks = int(config.attention_num // 2)
        self.clip_manifolds = nn.ModuleList()
        for i in range(num_lorentz_blocks):
            manifold = Lorentz(k=config.curv_init, learnable=config.learn_curv)
            self.clip_manifolds.append(manifold)
        
        self.clip_encoder = CurvSpecBlock(AttributeDict(hidden_size=config.hidden_size, 
                                                 intermediate_size=config.hidden_size,
                                                 hidden_dropout_prob=config.drop, 
                                                 num_attention_heads=config.n_heads,
                                                 attention_probs_dropout_prob=config.drop, 
                                                 frame_len=32, 
                                                 sft_factor=config.sft_factor,
                                                 drop=config.drop,
                                                 lorentz_dim=config.lorentz_dim,
                                                 attention_num=config.attention_num),
                                          manifolds_list=self.clip_manifolds)
        
        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,
                                             dropout=config.input_drop, relu=True)
        
        self.frame_manifolds = nn.ModuleList()
        for i in range(num_lorentz_blocks):
            manifold = Lorentz(k=config.curv_init, learnable=config.learn_curv)
            self.frame_manifolds.append(manifold)
        
        self.frame_encoder_1 = CurvSpecBlock(AttributeDict(hidden_size=config.hidden_size, 
                                                    intermediate_size=config.hidden_size,
                                                    hidden_dropout_prob=config.drop, 
                                                    num_attention_heads=config.n_heads,
                                                    attention_probs_dropout_prob=config.drop, 
                                                    frame_len=128, 
                                                    sft_factor=config.sft_factor,
                                                    drop=config.drop,
                                                    lorentz_dim=config.lorentz_dim,
                                                    attention_num=config.attention_num),
                                             manifolds_list=self.frame_manifolds)
        
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.weight_token = None
        
        self.curv = nn.Parameter(
            torch.tensor(config.curv_init).log(), requires_grad=False
        )
        self._curv_minmax = {
            "max": math.log(config.curv_init * 10),
            "min": math.log(config.curv_init / 10),
        }
        
        self.textual_alpha = nn.Parameter(torch.tensor(config.hidden_size**-0.5).log())
        self.video_alpha = nn.Parameter(torch.tensor(config.hidden_size**-0.5).log())
        self.text_hyp_proj = LinearLayer(config.hidden_size, config.lorentz_dim)
        self.vid_hyp_proj = LinearLayer(config.hidden_size, config.lorentz_dim)
        self.modular_vector_mapping_2 = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.modular_vector_mapping_3 = nn.Linear(config.hidden_size, out_features=1, bias=False)
        
        self.video_region_decomp = VideoRegionDecomposition(
            hidden_size=config.hidden_size,
            num_regions=config.num_constraints,
            lorentz_dim=config.lorentz_dim,
            num_heads=4,
            kmeans_use_gpu=getattr(config, 'kmeans_use_gpu', False),
        )

        self.use_adaptive_fusion = getattr(config, 'use_adaptive_fusion', False)
        if self.use_adaptive_fusion:
            gate_hidden = max(16, config.hidden_size // 4)
            self.fusion_gate_q = nn.Linear(config.hidden_size, gate_hidden)
            self.fusion_gate_v = nn.Linear(config.hidden_size, gate_hidden)
            self.fusion_gate_out = nn.Linear(gate_hidden, 1)
        self.last_fusion_gate = None

        self.reset_parameters()

    def reset_parameters(self):
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def forward(self, batch):
        clip_video_feat = batch['clip_video_features']
        query_feat = batch['text_feat']
        query_mask = batch['text_mask']
        query_labels = batch['text_labels']
        
        frame_video_feat = batch['frame_video_features']
        frame_video_mask = batch['videos_mask']
        
        encoded_frame_feat, vid_proposal_feat, frame_video_mask_padded = self.encode_context(
            clip_video_feat, frame_video_feat, frame_video_mask)
        
        clip_scale_scores, clip_scale_scores_, frame_scale_scores, frame_scale_scores_ \
            = self.get_pred_from_raw_query(
            query_feat, query_mask, query_labels, vid_proposal_feat, encoded_frame_feat, return_query_feats=True)
        
        label_dict = {}
        for index, label in enumerate(query_labels):
            if label in label_dict:
                label_dict[label].append(index)
            else:
                label_dict[label] = []
                label_dict[label].append(index)
        
        video_query = self.encode_query(query_feat, query_mask)
        if video_query.dim() == 1:
            video_query = video_query.unsqueeze(0)
        
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        
        all_curvatures = [_curv]
        for manifold in self.clip_manifolds:
            manifold.k.data = torch.clamp(manifold.k.data, min=self.config.curv_init / 10, 
                                          max=self.config.curv_init * 10)
            all_curvatures.append(manifold.k)
        
        for manifold in self.frame_manifolds:
            manifold.k.data = torch.clamp(manifold.k.data, min=self.config.curv_init / 10,
                                          max=self.config.curv_init * 10)
            all_curvatures.append(manifold.k)
        
        all_curvatures = torch.stack(all_curvatures)
        
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.)
        self.video_alpha.data = torch.clamp(self.video_alpha.data, max=0.)
        
        text_query_hyp = self.text_hyp_proj(video_query)
        text_query_hyp = text_query_hyp * self.textual_alpha.exp()
        query_norm = torch.norm(text_query_hyp, p=2, dim=-1, keepdim=True)
        text_query_hyp_clipped = text_query_hyp * torch.clamp(query_norm, max=5.0) / (query_norm + 1e-8)
        text_query_hyp = L.exp_map0(text_query_hyp_clipped, _curv)
        
        video_feat_pooled = self.get_modularized_clips(vid_proposal_feat)
        scale_video_feat = self.vid_hyp_proj(video_feat_pooled)
        scale_video_feat = scale_video_feat * self.video_alpha.exp()
        video_feat_norm = torch.norm(scale_video_feat, p=2, dim=-1, keepdim=True)
        video_feat_clipped = scale_video_feat * torch.clamp(video_feat_norm, max=5.0) / (video_feat_norm + 1e-8)
        scale_video_feat = L.exp_map0(video_feat_clipped, _curv)
        
        video_regions, frame_attn_weights, clip_attn_weights = self.video_region_decomp(
            frame_features=encoded_frame_feat,
            clip_features=vid_proposal_feat,
            frame_mask=frame_video_mask_padded,
            curv=_curv,
            scale=self.video_alpha.exp()
        )
        
        ot_loss = torch.tensor(0.0, device=video_query.device)
        if self.config.get('enable_loss_exclusive', True):
            try:
                from scipy.optimize import linear_sum_assignment
            except Exception as exc:
                raise ImportError(
                    "scipy is required for Hungarian matching; please install scipy"
                ) from exc
            total_sim = []
            for video_id, query_indices in label_dict.items():
                if len(query_indices) <= 1:
                    continue
                temp_clip_emb = vid_proposal_feat[video_id]
                temp_text_emb = video_query[query_indices]
                if temp_text_emb.dim() == 1:
                    temp_text_emb = temp_text_emb.unsqueeze(0)
                K = temp_clip_emb.shape[0]
                N = temp_text_emb.shape[0]
                if N > K:
                    pooled = temp_clip_emb.mean(dim=0, keepdim=True)
                    sim_scores = torch.matmul(
                        F.normalize(temp_text_emb, dim=-1),
                        F.normalize(pooled, dim=-1).t()
                    ).squeeze(-1)
                    topk_idx = torch.topk(sim_scores, k=K, largest=True).indices
                    temp_text_emb = temp_text_emb[topk_idx]
                    N = K
                if N <= 1:
                    continue
                sim = -1.0 * torch.matmul(
                    F.normalize(temp_clip_emb, dim=-1),
                    F.normalize(temp_text_emb, dim=-1).t()
                ).permute(1, 0)
                row_ind, col_ind = linear_sum_assignment(sim.detach().cpu().numpy())
                row_ind = torch.as_tensor(row_ind, device=sim.device, dtype=torch.long)
                col_ind = torch.as_tensor(col_ind, device=sim.device, dtype=torch.long)
                total_sim.append(sim[row_ind, col_ind])
            if len(total_sim) > 0:
                # Hungarian assignment on main-branch similarity
                ot_loss = 1.0 + torch.cat(total_sim).mean()
        
        return [clip_scale_scores, clip_scale_scores_, label_dict, frame_scale_scores, frame_scale_scores_, 
                all_curvatures, text_query_hyp, scale_video_feat, _curv, video_query, video_regions, 
                frame_attn_weights, clip_attn_weights, frame_video_mask_padded, ot_loss]

    def encode_query(self, query_feat, query_mask):
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,
                                          self.query_pos_embed)
        
        video_query = self.get_modularized_queries(encoded_query, query_mask)
        
        return video_query

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):
        encoded_clip_feat = self.encode_input(clip_video_feat, None, self.clip_input_proj, self.clip_encoder,
                                               self.clip_pos_embed, self.weight_token)
        
        if frame_video_feat.shape[1] != 128:
            fix = 128 - frame_video_feat.shape[1]
            temp_feat = 0.0 * frame_video_feat.mean(dim=1, keepdim=True).repeat(1, fix, 1)
            frame_video_feat = torch.cat([frame_video_feat, temp_feat], dim=1)
            
            temp_mask = 0.0 * video_mask.mean(dim=1, keepdim=True).repeat(1, fix).type_as(video_mask)
            video_mask = torch.cat([video_mask, temp_mask], dim=1)
        
        encoded_frame_feat = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj,
                                                self.frame_encoder_1,
                                                self.frame_pos_embed, self.weight_token)
        
        if video_mask.shape[1] != encoded_frame_feat.shape[1]:
            fix = encoded_frame_feat.shape[1] - video_mask.shape[1]
            temp_mask = 0.0 * video_mask.mean(dim=1, keepdim=True).repeat(1, fix).type_as(video_mask)
            video_mask = torch.cat([video_mask, temp_mask], dim=1)
        
        encoded_frame_feat = torch.where(video_mask.unsqueeze(-1).repeat(1, 1, encoded_frame_feat.shape[-1]) == 1.0,
                                          encoded_frame_feat, 0. * encoded_frame_feat)
        
        return encoded_frame_feat, encoded_clip_feat, video_mask

    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer, weight_token=None):
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)
        else:
            return encoder_layer(feat, mask)

    def get_modularized_queries(self, encoded_query, query_mask):
        modular_attention_scores = self.modular_vector_mapping(encoded_query)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)
        return modular_queries.squeeze(1)

    def get_modularized_frames(self, encoded_query, query_mask):
        modular_attention_scores = self.modular_vector_mapping_2(encoded_query)
        modular_attention_scores = F.softmax(modular_attention_scores, dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)
        return modular_queries.squeeze(1)
    
    def get_modularized_clips(self, encoded_query):
        modular_attention_scores = self.modular_vector_mapping_3(encoded_query)
        modular_attention_scores = F.softmax(modular_attention_scores, dim=1)
        modular_queries = torch.einsum("blm,bld->bmd", modular_attention_scores, encoded_query)
        return modular_queries.squeeze(1)

    @staticmethod
    def get_clip_scale_scores(modularied_query, context_feat):
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat = F.normalize(context_feat, dim=-1)
        
        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        
        query_context_scores, indices = torch.max(clip_level_query_context_scores, dim=1)
        
        return query_context_scores

    @staticmethod
    def get_unnormalized_clip_scale_scores(modularied_query, context_feat):
        query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0)
        
        output_query_context_scores, indices = torch.max(query_context_scores, dim=1)
        
        return output_query_context_scores

    def get_fusion_gate(self, video_query, video_proposal_feat, encoded_frame_feat):
        clip_pool = video_proposal_feat.mean(dim=1)
        frame_pool = encoded_frame_feat.mean(dim=1)
        video_pool = 0.5 * (clip_pool + frame_pool)

        if video_query.dim() == 1:
            video_query = video_query.unsqueeze(0)
        if video_pool.dim() == 1:
            video_pool = video_pool.unsqueeze(0)

        query_proj = self.fusion_gate_q(video_query)
        video_proj = self.fusion_gate_v(video_pool)

        fused = torch.tanh(query_proj[:, None, :] + video_proj[None, :, :])
        gate = torch.sigmoid(self.fusion_gate_out(fused).squeeze(-1))

        return gate

    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None,
                                return_query_feats=False):
        
        video_query = self.encode_query(query_feat, query_mask)
        
        clip_scale_scores = self.get_clip_scale_scores(video_query, video_proposal_feat)
        
        frame_scale_scores = self.get_clip_scale_scores(video_query, encoded_frame_feat)

        if self.use_adaptive_fusion:
            fusion_gate = self.get_fusion_gate(video_query, video_proposal_feat, encoded_frame_feat)
            fusion_scores = fusion_gate * clip_scale_scores + (1.0 - fusion_gate) * frame_scale_scores
            self.last_fusion_gate = fusion_gate.detach()
        else:
            fusion_scores = frame_scale_scores
            self.last_fusion_gate = None
        
        if return_query_feats:
            clip_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, video_proposal_feat)
            frame_scale_scores_ = self.get_unnormalized_clip_scale_scores(video_query, encoded_frame_feat)
            if self.use_adaptive_fusion:
                fusion_scores_ = fusion_gate * clip_scale_scores_ + (1.0 - fusion_gate) * frame_scale_scores_
            else:
                fusion_scores_ = frame_scale_scores_
            
            return clip_scale_scores, clip_scale_scores_, fusion_scores, fusion_scores_
        else:
            return clip_scale_scores, fusion_scores


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)
