import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import sys


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.silu = nn.SiLU()  # SiLU activation (Swish)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return self.silu(x1) * x2


class ClassificationHead(nn.Module):
    '''
    The classification head of NeurIPT, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, c_layers, d_model, num_classes, ts_dim, d_middle, part=None, args=None):
        super(ClassificationHead, self).__init__()

        self.c_layers = c_layers
        
        # Intra-Inter Lobe Pooling (IILP)
        if part == "Hemispheres":
            if args.data_dim == 22:
                self.groups = [
                    [0, 1, 2, 3, 8, 9, 10, 14, 15, 16, 17],
                    [4, 5, 6, 7, 11, 12, 13, 18, 19, 20, 21],
                ]
            elif args.data_dim == 20:
                self.groups = [
                    [0, 1, 2, 3, 8, 9, 12, 13, 14, 15],
                    [4, 5, 6, 7, 10, 11, 16, 17, 18, 19],
                ]
            elif args.data_dim == 17:
                self.groups = [
                    [0, 2, 3, 6, 7, 8, 11, 12, 15],
                    [1, 4, 5, 8, 9, 10, 13, 14, 16],
                ]
            elif args.data_dim == 16:
                self.groups = [
                    [0, 1, 4, 5, 6, 7, 8, 9],
                    [2, 3, 10, 11, 12, 13, 14, 15],
                ]
            elif args.data_dim == 3:
                self.groups = [
                    [0, 1, 2],
                ]
            elif args.data_dim == 2:
                self.groups = [
                    [0, 1],
                ]
        elif part == "Sagittal":
            if args.data_dim == 22:
                self.groups = [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11, 12, 13],
                    [14, 15, 16, 17],
                    [18, 19, 20, 21],
                ]
            elif args.data_dim == 20:
                self.groups = [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                    [16, 17, 18, 19],
                ]
        elif part == "Coronal":
            if args.data_dim == 22:
                self.groups = [
                    [0, 4, 14, 18],
                    [1, 5, 15, 19],
                    [2, 6, 16, 20],
                    [3, 7, 17, 21],
                    [8, 9, 10, 11, 12, 13],
                ]
            elif args.data_dim == 20:
                self.groups = [
                    [0, 4, 12, 16],
                    [1, 5, 13, 17],
                    [2, 6, 14, 18],
                    [3, 7, 15, 19],
                    [8, 9, 10, 11],
                ]
        elif part == "Functional":
            if args.data == "BCICIV-2A-SC-TTS":
                self.groups = [
                    
                ]
            elif args.data_dim == 22:
                self.groups = [
                    [0, 4, 14, 18],
                    [9, 10, 11, 12, 15, 16, 19, 20],
                    [3, 7, 17, 21],
                    [1, 2, 8, 9],
                    [5, 6, 12, 13],
                ]
            elif args.data_dim == 20:
                self.groups = [
                    [0, 4, 12, 16],
                    [8, 9, 10, 11, 13, 14, 17, 18],
                    [3, 7, 15, 19],
                    [1, 2, 8],
                    [5, 6, 11],
                ]
            elif args.data_dim == 17:
                self.groups = [
                    [0, 1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15, 16],
                ]
            elif args.data_dim == 16:
                self.groups = [
                    [4, 7, 10, 13],
                    [0, 1, 2, 3],
                    [5, 6, 8, 9],
                    [11, 12, 14, 15],
                ]
            elif args.data_dim == 3:
                self.groups = [
                    [0],
                    [1],
                    [2],
                ]
            elif args.data_dim == 2:
                self.groups = [
                    [0],
                    [1],
                ]
        self.classification_layer = nn.Linear(
            c_layers * d_model * len(self.groups), 
            num_classes if num_classes > 2 else 1,
        )


    def forward(self, enc_out):
        global_pools = []

        for i in range(self.c_layers):
            # group
            # layer_output = enc_out[i+1]
            layer_output = enc_out[-i]
            global_pool = []
            for this_group in self.groups:
                local_pool = layer_output[:, this_group, :, :].mean(dim=(1, 2))
                local_pool = local_pool.view(local_pool.size(0), -1)
                global_pool.append(local_pool)

            global_pool = torch.cat(global_pool, dim=1)

            global_pools.append(global_pool)
        
        global_pools = torch.cat(global_pools, dim=1)
        logits = self.classification_layer(global_pools)

        return logits