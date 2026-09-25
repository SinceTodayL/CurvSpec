import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from curvspec.geometry import lorentz_ops as L
from curvspec.representation.hyper_attention import LorentzMultiHeadedAttention
from curvspec.representation.kmeans import batch_kmeans_clustering


def onehot(indexes, N=None):
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().long().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    return output


class clip_nce(nn.Module):
    def __init__(self, reduction='mean', focal_gamma=0.0):
        super(clip_nce, self).__init__()
        self.reduction = reduction
        self.focal_gamma = focal_gamma

    def forward(self, labels, label_dict, q2ctx_scores=None, contexts=None, queries=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
        t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)
        if self.reduction:
            t2v_loss = t2v_denominator - t2v_nominator
            v2t_loss = v2t_denominator - v2t_nominator
            focal_gamma = getattr(self, 'focal_gamma', 0.0)
            if focal_gamma > 0:
                t2v_prob = torch.exp(t2v_nominator - t2v_denominator).clamp(1e-8, 1.0 - 1e-8)
                v2t_prob = torch.exp(v2t_nominator - v2t_denominator).clamp(1e-8, 1.0 - 1e-8)
                t2v_loss = (1.0 - t2v_prob) ** focal_gamma * t2v_loss
                v2t_loss = (1.0 - v2t_prob) ** focal_gamma * v2t_loss
            return torch.mean(t2v_loss) + torch.mean(v2t_loss)
        else:
            return t2v_denominator - t2v_nominator


class frame_nce(nn.Module):
    def __init__(self, reduction='mean', focal_gamma=0.0):
        super(frame_nce, self).__init__()
        self.reduction = reduction
        self.focal_gamma = focal_gamma

    def forward(self, q2ctx_scores=None, contexts=None, queries=None):

        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)

        nominator = torch.logsumexp(nominator, dim=1)

        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            loss = denominator - nominator
            focal_gamma = getattr(self, 'focal_gamma', 0.0)
            if focal_gamma > 0:
                prob = torch.exp(nominator - denominator).clamp(1e-8, 1.0 - 1e-8)
                loss = (1.0 - prob) ** focal_gamma * loss
            return torch.mean(loss)
        else:
            return denominator - nominator

class TrainablePositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, hidden_size, dropout=0.1):
        super(TrainablePositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(input_feat + position_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def add_position_emb(self, input_feat):
        bsz, seq_length = input_feat.shape[:2]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_feat.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)
        position_embeddings = self.position_embeddings(position_ids)
        return input_feat + position_embeddings


class LinearLayer(nn.Module):
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x

class EuclideanAttentionBlock(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanAttentionBlock, self).__init__()
        self.self = EuclideanGaussianAttention(config, wid=wid)
        self.output = FeedForward(config.hidden_size, int(4*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self(input_tensor, input_tensor, input_tensor, attention_mask)
        self_output = self.dropout1(self_output)
        input_tensor = self.norm1(input_tensor + self_output)
        tmp = self.output(input_tensor)
        tmp = self.dropout2(tmp)
        input_tensor = self.norm2(input_tensor + tmp)
        return input_tensor


class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        self.self = EuclideanGaussianAttention(config)
        self.output = FeedForward(config.hidden_size, int(1*config.hidden_size), config.hidden_dropout_prob)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, query, input_tensor, attention_mask=None):
        self_output = self.self(query, input_tensor, input_tensor, attention_mask)

        self_output = self.dropout1(self_output)
        query = self.norm1(query + self_output)
        tmp = self.output(query)
        tmp = self.dropout2(tmp)
        query = self.norm2(query + tmp)
        return query


class EuclideanGaussianAttention(nn.Module):
    def __init__(self, config, wid=None):
        super(EuclideanGaussianAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (
                config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.wid = wid
        

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def generate_gauss_weight(self, props_len, width, device):
        center = torch.arange(props_len, device=device) / props_len
        width = width * torch.ones(props_len, device=device)
        weight = torch.linspace(0, 1, props_len, device=device)
        weight = weight.view(1, -1).expand(center.size(0), -1)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / 9

        w = 0.3989422804014327

        weight = w / width * torch.exp(-(weight - center) ** 2 / (2 * width ** 2))

        return weight / torch.clamp(weight.max(dim=-1, keepdim=True)[0], min=1e-8)

    def forward(self, query_states, key_states, value_states, attention_mask=None):

        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(value_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores_ori = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores_ori = attention_scores_ori / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores_ori
        if self.wid is not None:
            gmm_mask = self.generate_gauss_weight(attention_scores.shape[-1], self.wid, attention_scores.device)
            gmm_mask = gmm_mask.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores_ori * gmm_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.unsqueeze(1)) * -10000.
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class LorentzSelfAttention(nn.Module):
    def __init__(self, config, manifold, wid=None):
        super().__init__()
        self.config = config
        self.input_proj = LinearLayer(config.hidden_size, config.lorentz_dim, dropout=config.drop)
        self.scale_alpha = nn.Parameter(torch.tensor(config.lorentz_dim**-0.5).log())
        self.lorentz_multiattention = LorentzMultiHeadedAttention(
            dropout=config.drop,
            model_dim=config.lorentz_dim+1,
            head_count=config.num_attention_heads,
            manifold=manifold,
            wid=wid
        )
        self.manifold = manifold
        self.output_proj = LinearLayer(config.lorentz_dim, config.hidden_size, dropout=config.drop)

    def forward(self, x, attention_mask=None):
        x = self.input_proj(x)
        
        self.scale_alpha.data = torch.clamp(self.scale_alpha.data, max=0.)
        x = x * self.scale_alpha.exp()
        
        x = F.pad(x, pad=(1, 0), value=0)
        x = self.manifold.expmap0(x)
        x = self.lorentz_multiattention(x, x, x, attention_mask)
        x = self.manifold.logmap0(x)[:, :, 1:]
        x = x / torch.clamp(self.scale_alpha.exp(), min=1e-8)
        output = self.output_proj(x)
        return output


class LorentzAttentionBlock(nn.Module):
    def __init__(self, config, manifold, wid=None):
        super().__init__()
        self.self_attn = LorentzSelfAttention(config, manifold, wid)
        self.output = FeedForward(config.hidden_size, int(4*config.hidden_size), config.hidden_dropout_prob)
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        self_output = self.self_attn(input_tensor, attention_mask)
        self_output = self.dropout1(self_output)
        input_tensor = self.norm1(input_tensor + self_output)
        tmp = self.output(input_tensor)
        tmp = self.dropout2(tmp)
        input_tensor = self.norm2(input_tensor + tmp)
        return input_tensor


class CurvSpecBlock(nn.Module):
    def __init__(self, config, manifolds_list):
        super(CurvSpecBlock, self).__init__()
        
        self.num_block = int(config.attention_num // 2) - 1
        
        self.e_attns = nn.ModuleList()
        self.e_attns.append(EuclideanAttentionBlock(config))
        for i in range(1, self.num_block + 1):
            wid = 2 ** i
            self.e_attns.append(EuclideanAttentionBlock(config, wid=wid))
        
        self.h_attns = nn.ModuleList()
        if len(manifolds_list) != self.num_block + 1:
            raise ValueError(f"Expected {self.num_block + 1} manifolds, got {len(manifolds_list)}")
        
        self.h_attns.append(LorentzAttentionBlock(config, manifold=manifolds_list[0]))
        for i in range(1, self.num_block + 1):
            wid = 2 ** i
            self.h_attns.append(LorentzAttentionBlock(config, manifold=manifolds_list[i], wid=wid))
        
        self.ca = CrossAttention(config)
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer2 = nn.Linear(config.hidden_size, config.frame_len)
        
        self.sft_factor = config.sft_factor

    def forward(self, input_tensor, attention_mask=None, weight_token=None):
        outputs = []
        for i in range(len(self.e_attns)):
            o = self.e_attns[i](input_tensor, attention_mask).unsqueeze(-1)
            outputs.append(o)
        
        for i in range(len(self.h_attns)):
            o = self.h_attns[i](input_tensor, attention_mask).unsqueeze(-1)
            outputs.append(o)
        
        oo = torch.cat(outputs, dim=-1)
        
        if weight_token is None:
            mean_oo = torch.mean(oo, dim=-1)
            weight_token = torch.mean(mean_oo, dim=1, keepdim=True)
        else:
            weight_token = weight_token.to(oo.device).type_as(oo).repeat(oo.shape[0], 1, 1)
        
        weight = []
        for i in range(oo.shape[-1]):
            temp_token = self.ca(weight_token, oo[..., i], attention_mask)
            weight.append(temp_token)
        
        weight = torch.cat(weight, dim=1)
        weight = self.layer1(weight)
        weight = self.dropout(F.relu(weight))
        weight = self.layer2(weight)
        
        weight = F.softmax(weight.permute(0, 2, 1) / self.sft_factor, dim=-1)
        out = torch.sum(oo * weight.unsqueeze(2).repeat(1, 1, oo.shape[2], 1), dim=-1)
        
        return out


class ConstraintHead(nn.Module):
    def __init__(self, hidden_size: int, num_constraints: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_constraints = num_constraints
        self.centroid_proj = nn.Linear(hidden_size, num_constraints * hidden_size)

    def forward(self, query: torch.Tensor, curv: torch.Tensor, scale: torch.Tensor):
        bsz, dim = query.shape
        centroids_tan = self.centroid_proj(query)
        centroids_tan = centroids_tan.view(bsz, self.num_constraints, dim)
        
        centroids_tan_norm = F.normalize(centroids_tan, dim=-1) * scale
        
        norm = torch.norm(centroids_tan_norm, p=2, dim=-1, keepdim=True)
        centroids_tan_clipped = centroids_tan_norm * torch.clamp(norm, max=5.0) / (norm + 1e-8)
        
        centroids = L.exp_map0(centroids_tan_clipped, curv)
        return centroids


class VideoRegionDecomposition(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_regions: int,
        lorentz_dim: int,
        num_heads: int = 4,
        kmeans_use_gpu: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_regions = num_regions
        self.lorentz_dim = lorentz_dim
        self.kmeans_use_gpu = kmeans_use_gpu
        
        self.region_prototypes = nn.Parameter(torch.randn(num_regions, hidden_size))
        nn.init.xavier_uniform_(self.region_prototypes)
        
        self.frame_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.clip_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.frame_norm = nn.LayerNorm(hidden_size)
        self.clip_norm = nn.LayerNorm(hidden_size)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.region_refine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.to_hyperbolic = nn.Linear(hidden_size, lorentz_dim)
        
    def forward(self, frame_features: torch.Tensor, clip_features: torch.Tensor, 
                frame_mask: torch.Tensor, curv: torch.Tensor, scale: torch.Tensor):
        B = frame_features.shape[0]
        K = self.num_regions
        
        use_gpu = self.kmeans_use_gpu and clip_features.is_cuda
        prototypes = batch_kmeans_clustering(
            clip_features,
            n_clusters=K,
            n_init=10,
            max_iter=100,
            use_gpu=use_gpu,
        )
        
        if frame_mask is not None:
            frame_key_padding_mask = (frame_mask == 0)
        else:
            frame_key_padding_mask = None
        
        frame_regions, frame_attn_weights = self.frame_cross_attn(
            query=prototypes,
            key=frame_features,
            value=frame_features,
            key_padding_mask=frame_key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        frame_regions = self.frame_norm(frame_regions + prototypes)
        
        clip_regions, clip_attn_weights = self.clip_cross_attn(
            query=prototypes,
            key=clip_features,
            value=clip_features,
            need_weights=True,
            average_attn_weights=False
        )
        clip_regions = self.clip_norm(clip_regions + prototypes)
        
        combined = torch.cat([frame_regions, clip_regions], dim=-1)
        gate = self.fusion_gate(combined)
        fused_regions = gate * frame_regions + (1 - gate) * clip_regions
        
        combined_for_refine = torch.cat([fused_regions, prototypes], dim=-1)
        refined_regions = self.region_refine(combined_for_refine)
        refined_regions = refined_regions + fused_regions
        
        regions_hyp_tan = self.to_hyperbolic(refined_regions)
        regions_hyp_tan = regions_hyp_tan * scale
        
        norm = torch.norm(regions_hyp_tan, p=2, dim=-1, keepdim=True)
        regions_hyp_tan_clipped = regions_hyp_tan * torch.clamp(norm, max=5.0) / (norm + 1e-8)
        
        regions_hyp = L.exp_map0(regions_hyp_tan_clipped, curv)
        
        return regions_hyp, frame_attn_weights, clip_attn_weights
