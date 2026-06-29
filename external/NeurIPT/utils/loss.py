import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCE_loss(nn.Module):
    '''
        Paper: https://arxiv.org/abs/2212.02015
        Code: https://github.com/XuZhengzhuo/LiVT
    '''
    def __init__(self, args,
                 cls_num,
                 target_threshold=None, 
                 type=None,
                 reduction='mean', 
                 pos_weight=None, 
                 K=1., 
                 lam=1., 
                 use_lam=False,
                 binary=False,
                 ):
        super(BCE_loss, self).__init__()
        self.lam = lam
        self.K = K
        self.use_lam = use_lam
        self.smoothing = args.smoothing
        self.cls_num = cls_num  # Samples of each class
        self.num_classes = args.num_classes if args.num_classes > 2 else 1  # Total categories
        self.target_threshold = target_threshold
        self.weight = None
        self.pi = None
        self.reduction = reduction
        self.binary = True if binary and type else False
        self.register_buffer('pos_weight', pos_weight)

        if type == 'Bal':
            self._cal_bal_pi()
        elif type == 'CB':
            self._cal_cb_weight(args)

    def _cal_bal_pi(self):
        cls_num = torch.Tensor(self.cls_num)
        self.pi = cls_num / torch.sum(cls_num)

    def _cal_cb_weight(self, args):
        eff_beta = 0.9999
        effective_num = 1.0 - np.power(eff_beta, args.cls_num)
        per_cls_weights = (1.0 - eff_beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(args.cls_num)
        self.weight = torch.FloatTensor(per_cls_weights).to(args.device)

    def _bal_sigmod_bias(self, x):
        pi = self.pi.to(x.device)
        bias = torch.log(pi) - torch.log(1-pi)
        x = x + self.K * bias
        return x

    def _neg_reg(self, labels, logits, weight=None):
        if weight == None:
            weight = torch.ones_like(labels).to(logits.device)
        pi = self.pi.to(logits.device)
        bias = torch.log(pi) - torch.log(1-pi)
        logits = logits * (1 - labels) * self.lam + logits * labels # neg + pos
        logits = logits + self.K * bias
        weight = weight / self.lam * (1 - labels) + weight * labels # neg + pos
        return logits, weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.use_lam:
            x, weight = self._neg_reg(target, x)
        else:
            if self.pi != None:
                x = self._bal_sigmod_bias(x)
        C = x.shape[-1] # + log C
        if self.binary:
            x = x.gather(dim=1, index=target.long())
        return C * F.binary_cross_entropy_with_logits(x, target, weight, self.pos_weight, reduction=self.reduction)