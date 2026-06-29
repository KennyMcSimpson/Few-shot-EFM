import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import TwoStageAttentionLayer
from math import ceil

import sys

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.

    This module merges adjacent segments along the temporal dimension into coarser-scale segments.
    Specifically, it concatenates `win_size` consecutive segments, applies normalization,
    and projects them back to the original model dimension.
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)
            print("WARNING: merging is padding...")

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]
        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale.
    We use Progressive Mixture-of-Experts (PMoE) here
    '''
    def __init__(self, win_size, d_model, n_heads, d_ff, depth, dropout, factor, use_router, out_len, args, enc_expert):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(factor, d_model, n_heads, args, d_ff, dropout, use_router, out_len, expert=enc_expert)
            )
    
    def forward(self, x, i, time_mask, dim_mask):
        _, ts_dim, _, _ = x.shape
        aux_loss = torch.tensor(0.0, dtype=torch.float32, device=x.device)

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            x, enc_loss = layer(x, i, time_mask, dim_mask)
            aux_loss += enc_loss
        
        return x, aux_loss

class Encoder(nn.Module):
    '''
    The Encoder of NeurIPT.
    We use Amplitude-Aware Masked Pretraining (AAMP) in stage1.
    '''
    def __init__(self, e_blocks, merge_layers, d_model, n_heads, d_ff, block_depth, dropout, factor, use_router, out_len, stage, args):
        super(Encoder, self).__init__()
        self.n_heads = args.n_heads
        self.encode_blocks = nn.ModuleList()
        for i in range(e_blocks):
            self.encode_blocks.append(
                scale_block(
                    merge_layers[i], d_model, n_heads, d_ff, block_depth, dropout, 
                    factor, use_router, out_len, args, enc_expert=args.enc_expert[i],
                )
            )
            
        self.stage1 = True if stage == "stage1" else False

    def forward(self, x, mask_info=None):
        device = x.device
        encode_x = []
        encode_x.append(x)
        aux_loss = torch.tensor(0.0, dtype=torch.float32, device=x.device)
        
        if self.stage1:
            time_mask, dim_mask = [], []
            
            for i in range(len(mask_info)):
                b, ts_dim, seg_num = mask_info[i].shape
                time_mask.append(~mask_info[i].view(-1,seg_num).to(device))
                time_mask[i] = time_mask[i].unsqueeze(2).expand(b*ts_dim, seg_num, seg_num).contiguous()
                time_idx = torch.arange(seg_num, device=device)
                time_mask[i][:, time_idx, time_idx] = True
                time_mask[i] = time_mask[i].unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                
                dim_mask.append(~mask_info[i].transpose(1, 2).contiguous().view(-1,ts_dim).to(device))
                dim_mask[i] = dim_mask[i].unsqueeze(2).expand(b*seg_num, ts_dim, ts_dim).contiguous()
                dim_idx = torch.arange(ts_dim, device=device)
                dim_mask[i][:, dim_idx, dim_idx] = True
                dim_mask[i] = dim_mask[i].unsqueeze(1).expand(-1, self.n_heads, -1, -1)

            for i, block in enumerate(self.encode_blocks):
                x, enc_loss = block(x, i+1, time_mask[i], dim_mask[i])
                aux_loss += enc_loss
                encode_x.append(x)
        else:
            time_mask, dim_mask = None, None
            for i, block in enumerate(self.encode_blocks):
                x, enc_loss = block(x, i+1, time_mask=None, dim_mask=None)
                aux_loss += enc_loss
                encode_x.append(x)

        return encode_x, aux_loss