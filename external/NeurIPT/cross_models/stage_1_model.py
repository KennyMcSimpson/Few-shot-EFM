import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.attn import TwoStageAttentionLayer
from cross_models.cross_embed import DSW_embedding

from cross_models.cross_embed import TemporalPE, SpacialPE

from math import ceil

import sys

class Stage_1_model(nn.Module):
    '''
    The stage 1 model of NeurIPT.
    In this stage, Amplitude-Aware Masked Pretraining (AAMP) 
    and Progressive Mixture-of-Experts (PMoE) are used to improve the pre-training performance.
    '''
    def __init__(self, data_dim, out_len, seg_len, merge_layers, args, factor=10,
                 d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                 dropout=0.0, use_norm = True, use_router=False):
        super(Stage_1_model, self).__init__()
        self.out_len = out_len
        self.use_norm = use_norm
        self.d_model = d_model

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        
        temporal_PE = TemporalPE(self.d_model, max_len=self.out_len)() # (1, seg_num, d_model)
        spacial_PE = SpacialPE(self.d_model, args)()  # (1, ts_d, d_model)
        dec_in = repeat(temporal_PE, '1 seg_num d_model -> ts_d seg_num d_model', ts_d=args.data_dim)
        dec_in = dec_in + repeat(spacial_PE, '1 ts_d d_model -> ts_d seg_num d_model', seg_num=args.out_len)
        dec_in = repeat(dec_in, 'ts_d seg_num d_model -> b ts_d seg_num d_model', b=args.batch_size_stage_1)
        self.register_buffer('dec_in', dec_in)

        # Encoder
        self.encoder = Encoder(
            e_layers, merge_layers, d_model, n_heads, d_ff, block_depth=1, dropout=dropout, 
            factor=factor, use_router=use_router, out_len=out_len, stage="stage1", args=args,
        )
        
        # Decoder
        self.decoder = Decoder(
            seg_len, e_layers+1, d_model, n_heads, d_ff, dropout, 
            factor=factor, use_router=use_router, args=args,
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x_seq, mask_info, re_mask_info):
        batch_size, seg_num, ts_dim = x_seq.size()

        enc_in = self.enc_value_embedding(x_seq)
        
        enc_out, enc_aux_loss = self.encoder(enc_in, mask_info)

        # using mask token
        MaskToken = self.mask_token.view(1, 1, 1, enc_out[0].shape[-1]).expand_as(enc_out[0])
        MaskInfo = re_mask_info[0].unsqueeze(-1).expand_as(enc_out[0])
        enc_out[0] = torch.where(MaskInfo, MaskToken, enc_out[0])
        
        for i in range(len(re_mask_info)):
            MaskToken = self.mask_token.view(1, 1, 1, enc_out[i+1].shape[-1]).expand_as(enc_out[i+1])
            MaskInfo = re_mask_info[i].unsqueeze(-1).expand_as(enc_out[i+1])
            enc_out[i+1] = torch.where(MaskInfo, MaskToken, enc_out[i+1])

        predict_y, dec_aux_loss = self.decoder(self.dec_in, enc_out)
        predict_y = predict_y[mask_info[0].transpose(1, 2)].view(batch_size, -1, ts_dim)
        
        return predict_y, enc_aux_loss + dec_aux_loss