import torch
import torch.nn as nn
from math import ceil



class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        from cross_models.attn import SwiGLU
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            SwiGLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MoE(nn.Module):
    '''
    Here is detailed of Progressive Mixture-of-Experts (PMoE).
    '''
    def __init__(self, args, expert=0):
        super().__init__()
        self.input_dim = args.input_dim        
        self.num_experts = expert
        self.top_k = ceil(self.num_experts * args.top_k)     
        self.hidden_dim = args.hidden_dim       
        self.output_dim = args.output_dim     
        self.use_shared_expert = getattr(args, 'use_shared_expert', True)
        self.noise_std = getattr(args, 'noise_std', 1e-2)
        self.w_importance = getattr(args, 'w_importance', 0.01)
        self.hidden_dim_shared = args.hidden_dim_shared
        self.gate = nn.Linear(self.input_dim, self.num_experts)
        '''
        - input_dim: input dimension
        - hidden_dim: hidden dimension of experts
        - output_dim: output dimension
        - num_experts: number of experts
        - top_k: number of experts to use
        - use_shared_expert: whether to use a shared expert
        - hidden_dim_shared: hidden dimension of shared expert
        - noise_std: standard deviation of noise added to gate logits
        - w_importance: weight for importance loss
        '''

        self.experts = nn.ModuleList([
            Expert(self.input_dim, self.hidden_dim, self.output_dim)
            for _ in range(self.num_experts)
        ])

        if self.use_shared_expert:
            self.shared_expert = Expert(self.input_dim, self.hidden_dim_shared, self.output_dim)
        else:
            self.shared_expert = None

    def forward(self, x):
        shape = x.size()
        x = x.view(-1, x.size(-1))
        batch_size = x.shape[0]
        expert_capacity = batch_size // self.num_experts
        device = x.device

        shared_output = 0
        if self.use_shared_expert:
            shared_output = self.shared_expert(x)

        # Get the logits from the gate
        logits = self.gate(x)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        #auxiliary loss
        if self.training:
            importance = probs.sum(0)
            importance_mean = importance.mean()
            importance_var = torch.var(importance)
            importance_loss = self.w_importance * (importance_var / (importance_mean ** 2 + 1e-8))

            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            routing_probs = probs * mask
            expert_usage = routing_probs.mean(0)
            routing_weights = routing_probs.sum(0) 
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum() / (batch_size ** 2)
            aux_loss = importance_loss + load_balance_loss
        else:
            aux_loss = 0


        flat_indices = topk_indices.view(-1)
        flat_probs = topk_probs.view(-1)
        output_dim = self.experts[0].net[-1].out_features

        sample_indices = torch.arange(batch_size, device=device)[:, None]  
        sample_indices = sample_indices.expand(-1, self.top_k).flatten()

        # Get the expert outputs
        moe_output = torch.zeros(batch_size, output_dim, device=device)
        for expert_idx in range(self.num_experts):
            expert_mask = flat_indices == expert_idx
            expert_samples = sample_indices[expert_mask]
            expert_weights = flat_probs[expert_mask]

            if len(expert_samples) > expert_capacity:
                expert_samples = expert_samples[:expert_capacity]
                expert_weights = expert_weights[:expert_capacity]

            if len(expert_samples) == 0:
                continue

            expert_input = x[expert_samples]
            expert_output = self.experts[expert_idx](expert_input)
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            moe_output.index_add_(0, expert_samples, weighted_output)


        final_output = shared_output + moe_output
        return final_output.view(*shape[:-1], -1) , aux_loss