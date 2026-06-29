import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.attn import TwoStageAttentionLayer
from cross_models.cross_embed import DSW_embedding
from cross_models.classification import ClassificationHead

from math import ceil

import sys

class Stage_2_model(nn.Module):
    '''
    The stage 2 model of NeurIPT.
    In this stage, Progressive Mixture-of-Experts (PMoE) is used here.
    The PMoE can be used in the encoder,decoder and cross-attention layers.
    '''
    def __init__(self, data_dim, out_len, seg_len, merge_layers, args, factor=10,
                 d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                 dropout=0.0, use_norm=True, c_layers=3, num_classes=8, d_middle=32, part=None, use_router=False):
        super(Stage_2_model, self).__init__()
        self.out_len = out_len
        self.use_norm = use_norm

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)

        # Encoder
        self.encoder = Encoder(
            e_layers, merge_layers, d_model, n_heads, d_ff, block_depth=1, dropout=dropout, 
            factor=factor, use_router=use_router, out_len=out_len, stage="stage2", args=args,
        )

        # Classification Head
        self.classification_head = ClassificationHead(c_layers+1, d_model, num_classes, data_dim, d_middle, part, args)
        
    def forward(self, x_seq):
        batch_size, _, ts_d = x_seq.size()

        # Embedding
        x_seq = self.enc_value_embedding(x_seq)
        
        # Encoder
        enc_out, enc_aux_loss = self.encoder(x_seq)

        # Classification Head
        logits = self.classification_head(enc_out)

        return logits, enc_aux_loss