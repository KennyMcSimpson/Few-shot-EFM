import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

from utils.tools import get_3d_pos, seperate_3d_dims

import sys

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed
    

class TemporalPE(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TemporalPE, self).__init__()

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe_t', pe)

    def forward(self):
        return self.pe_t


class SpacialPE(nn.Module):
    def __init__(self, d_model, args, max_range=50):
        super(SpacialPE, self).__init__()
        self.max_range = max_range
        
        d_model_x, d_model_y, d_model_z = seperate_3d_dims(d_model)
        
        edge_pos_list = get_3d_pos(max_range-1, args)

        pe_x = self._generate_pe(d_model_x, edge_pos_list[:, 0])
        pe_y = self._generate_pe(d_model_y, edge_pos_list[:, 1])
        pe_z = self._generate_pe(d_model_z, edge_pos_list[:, 2])
        
        pe_3d = torch.cat([pe_x, pe_y, pe_z], dim=-1)
        
        self.register_buffer('pe_3d', pe_3d)
        
    def _generate_pe(self, d_model, edge_pos):
        position = torch.arange(0, self.max_range).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(self.max_range, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe[edge_pos].unsqueeze(0)

    def forward(self):
        return self.pe_3d  # size = [1, ts_d, d_model]