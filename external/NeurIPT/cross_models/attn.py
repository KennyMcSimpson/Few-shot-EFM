import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

import sys

from cross_models.cross_embed import TemporalPE, SpacialPE

from cross_models.moe_models.primary_shared_expert import MoE

class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.silu = nn.SiLU()  # SiLU activation (Swish)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.silu(x1) * x2


class FlashAttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer using FlashAttention-2
    Paper: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(FlashAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = dropout

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape # B, L, D
        _, S, _ = keys.shape # B, S, D
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1).transpose(1, 2) # B, L, H, d_keys -> B, H, L, d_keys
        keys = self.key_projection(keys).view(B, S, H, -1).transpose(1, 2) # B, S, H, d_keys -> B, H, S, d_keys
        values = self.value_projection(values).view(B, S, H, -1).transpose(1, 2) # B, S, H, d_keys -> B, H, S, d_keys
        
        out = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, L, -1)

        return self.out_projection(out)


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, factor, d_model, n_heads, args, d_ff=None, dropout=0.1, use_router=False, out_len=5000, expert=0):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.expert = expert
        self.time_attention = FlashAttentionLayer(d_model, n_heads, dropout = dropout)
        
        self.use_router = use_router
        if not self.use_router:
            self.dim_attention = FlashAttentionLayer(d_model, n_heads, dropout = dropout)
        else:
            self.dim_sender = AttentionLayer(d_model, n_heads, dropout = dropout)
            self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout)
            self.router = nn.Parameter(torch.randn(factor, d_model))
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        if self.expert == 0:
            self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff * 2),
                                    SwiGLU(),
                                    nn.Linear(d_ff, d_model))
            
            self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff * 2),
                                    SwiGLU(),
                                    nn.Linear(d_ff, d_model))  
        else:
            self.MLP1 = MoE(args, expert)
            self.MLP2 = MoE(args, expert)
        
        # Temporal-spacial embedding
        self.temporal_PE = TemporalPE(d_model, max_len=out_len)
        self.spacial_PE = SpacialPE(d_model, args)

    def forward(self, x, i, time_mask, dim_mask):
        batch, ts_dim, seg_num, d_model = x.size()
        
        if i == 1:
            x += repeat(self.temporal_PE()[:, ::i, :].unsqueeze(0), '1 1 seg_num d_model -> b ts_d seg_num d_model', b=batch, ts_d=ts_dim)
        
        # Temporal Stage
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_inn = self.norm1(time_in)
        time_enc = self.time_attention(time_inn, time_inn, time_inn, time_mask)
        dim_in = time_in + self.dropout(time_enc)
        dim_inn = self.norm2(dim_in)
        if self.expert == 0:
            dim_in = dim_in + self.dropout(self.MLP1(dim_inn))
            aux_loss1 = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        else:
            MLP1temp, aux_loss1 = self.MLP1(dim_inn)
            dim_in = dim_in + self.dropout(MLP1temp)
        dim_in = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        
        if i == 1:
            dim_in += repeat(self.spacial_PE(), '1 ts_d d_model -> (b seg_num) ts_d d_model', b=batch, seg_num=seg_num)
        
        # Spatial Stage
        dim_inn = self.norm3(dim_in)
        dim_enc = self.dim_attention(dim_inn, dim_inn, dim_inn, dim_mask)
        dim_enc = dim_in + self.dropout(dim_enc)
        dim_encn = self.norm4(dim_enc)
        if self.expert == 0:
            dim_enc = dim_enc + self.dropout(self.MLP2(dim_encn))
            aux_loss2 = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        else:
            MLP2temp, aux_loss2 = self.MLP2(dim_encn)
            dim_enc = dim_enc + self.dropout(MLP2temp)

        return rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch), aux_loss1 + aux_loss2
