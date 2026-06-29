import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import FlashAttentionLayer, TwoStageAttentionLayer, SwiGLU
from cross_models.moe_models.primary_shared_expert import MoE

class DecoderLayer(nn.Module):
    '''
    The decoder layer of NeurIPT, each layer will make a prediction at its scale
    Progressive Mixture-of-Experts (PMoE) can be used here (optional).
    '''
    def __init__(self, seg_len, d_model, n_heads, d_ff, dropout, factor, use_router, args, dec_expert, cross_expert):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(factor, d_model, n_heads, args, d_ff, dropout, use_router, expert=dec_expert)
        self.cross_attention = FlashAttentionLayer(d_model, n_heads, dropout = dropout)
        
        self.norm1_q = nn.LayerNorm(d_model)
        self.norm1_kv = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.cross_expert = cross_expert
        
        if self.cross_expert == 0:
            self.MLP = nn.Sequential(nn.Linear(d_model, d_ff * 2),
                                    SwiGLU(),
                                    nn.Linear(d_ff, d_model))
        else:
            self.MLP = MoE(args, cross_expert)
        
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        '''
        x: the input of decoder layer
        cross: the output of the corresponding encoder layer
        '''
        batch = x.shape[0]
        x, self_aux_loss = self.self_attention(x, i=0, time_mask=None, dim_mask=None)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        
        x = self.norm1_q(x)
        cross = self.norm1_kv(cross)
        attn_out = self.cross_attention(x, cross, cross, attn_mask=None)
        x = x + self.dropout(attn_out)
        
        dec_out = self.norm2(x)
        if self.cross_expert == 0:
            y = self.MLP(dec_out)
            cross_aux_loss = 0
        else:
            y, cross_aux_loss = self.MLP(dec_out)
        
        dec_out = dec_out + y
        dec_out = rearrange(dec_out, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        layer_predict = self.linear_pred(dec_out)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_out, layer_predict, self_aux_loss + cross_aux_loss

class Decoder(nn.Module):
    '''
    The decoder of NeurIPT, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, d_model, n_heads, d_ff, dropout, factor, use_router, args):
        super(Decoder, self).__init__()

        self.decode_layers = nn.ModuleList()
        for i in range(d_layers):
            self.decode_layers.append(
                DecoderLayer(
                    seg_len, d_model, n_heads, d_ff, dropout, factor, use_router, args, args.dec_expert[i], args.cross_expert[i]
                )
            )

    def forward(self, x, cross):
        final_predict = None
        ts_d = x.shape[1]
        aux_loss = 0
        
        for i, layer in enumerate(self.decode_layers):
            cross_enc = cross[i]
            x, layer_predict, dec_loss = layer(x, cross_enc)
            aux_loss += dec_loss
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict += layer_predict
        
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)

        return final_predict, aux_loss

