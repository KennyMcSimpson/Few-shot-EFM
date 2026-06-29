import copy
from collections import defaultdict
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CSBrain(nn.Module):
    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=5,
        n_layer=12,
        nhead=8,
        TemEmbed_kernel_sizes=[(1,), (3,), (5,)],
        brain_regions=None,
        sorted_indices=None,
    ):
        super().__init__()
        if brain_regions is None:
            brain_regions = []
        if sorted_indices is None:
            sorted_indices = list(range(len(brain_regions)))

        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        self.TemEmbed_kernel_sizes = TemEmbed_kernel_sizes
        self.TemEmbedEEGLayer = TemEmbedEEGLayer(
            dim_in=in_dim,
            dim_out=out_dim,
            kernel_sizes=self.TemEmbed_kernel_sizes,
            stride=1,
        )
        self.brain_regions = list(brain_regions)
        self.area_config = generate_area_config(sorted(self.brain_regions))
        self.BrainEmbedEEGLayer = BrainEmbedEEGLayer(dim_in=in_dim, dim_out=out_dim)
        self.sorted_indices = list(sorted_indices)

        encoder_layer = CSBrainTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            area_config=self.area_config,
            sorted_indices=self.sorted_indices,
            batch_first=True,
            activation=F.gelu,
        )
        self.encoder = CSBrainTransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        # x: [B, C, patch_num, patch_size]
        x = x[:, self.sorted_indices, :, :]
        patch_emb = self.patch_embedding(x, mask)
        for layer_idx in range(self.encoder.num_layers):
            patch_emb = self.TemEmbedEEGLayer(patch_emb) + patch_emb
            patch_emb = self.BrainEmbedEEGLayer(patch_emb, self.area_config) + patch_emb
            patch_emb = self.encoder.layers[layer_idx](patch_emb, self.area_config)
        return self.proj_out(patch_emb)


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(nn.Linear(d_model // 2 + 1, d_model), nn.Dropout(0.1))

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        spectral_in = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(spectral_in, dim=-1, norm="forward")
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, spectral_in.shape[1] // 2 + 1)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        return patch_emb + positional_embedding


class CSBrainTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        bias: bool = True,
        area_config: dict = None,
        sorted_indices: list = None,
    ):
        super().__init__()
        if area_config is None:
            area_config = {}
        self.inter_region_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.inter_window_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)
        self.global_fc = nn.Linear(d_model, d_model, bias=bias)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = getattr(F, activation, F.relu) if isinstance(activation, str) else activation
        self.area_config = area_config
        self.region_attn_mask = None
        self.region_indices_dict = None
        if area_config:
            total_channels = sum(
                len(range(info["slice"].start or 0, info["slice"].stop, info["slice"].step or 1))
                if isinstance(info["slice"], slice) else len(info["slice"])
                for info in area_config.values()
            )
            builder = RegionAttentionMaskBuilder(total_channels, area_config)
            self.region_attn_mask = builder.get_mask()
            self.region_indices_dict = builder.get_region_indices()

    def forward(self, src, area_config=None, src_mask=None, src_key_padding_mask=None):
        x = src
        x = x + self._inter_window_attention(self.norm1(x), src_mask, src_key_padding_mask)
        if self.region_attn_mask is None and area_config is not None:
            x = x + self._inter_region_attention_dynamic(self.norm2(x), area_config, src_mask, src_key_padding_mask)
        else:
            x = x + self._inter_region_attention_static(self.norm2(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm3(x))
        return x

    def _inter_region_attention_static(self, x, attn_mask=None, key_padding_mask=None):
        if self.region_attn_mask is None or self.region_indices_dict is None:
            raise ValueError("no initialized region attention mask or region indices dictionary")
        batch, chans, time_steps, features = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(batch * time_steps, chans, features)
        global_features = torch.zeros_like(x_flat)
        for _, region_indices in self.region_indices_dict.items():
            region_x = x[:, region_indices, :, :]
            region_global = region_x.mean(dim=1, keepdim=True).permute(0, 2, 1, 3).reshape(batch * time_steps, 1, features)
            for idx in region_indices:
                global_features[:, idx:idx + 1, :] = region_global
        x_enhanced = x_flat + self.global_fc(global_features)
        attn_output = self.inter_region_attn(
            x_enhanced,
            x_enhanced,
            x_enhanced,
            attn_mask=self.region_attn_mask.to(x.device),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        attn_output = attn_output.reshape(batch, time_steps, chans, features).permute(0, 2, 1, 3)
        return self.dropout1(attn_output)

    def _inter_region_attention_dynamic(self, x, area_config, attn_mask=None, key_padding_mask=None):
        # Current Ada TUEV path uses the static region mask. This dynamic fallback
        # keeps the module safe if area_config is rebuilt externally.
        self.area_config = area_config
        builder = RegionAttentionMaskBuilder(x.shape[1], area_config)
        self.region_attn_mask = builder.get_mask()
        self.region_indices_dict = builder.get_region_indices()
        return self._inter_region_attention_static(x, attn_mask, key_padding_mask)

    def _inter_window_attention(self, x, attn_mask=None, key_padding_mask=None):
        batch, chans, time_steps, features = x.shape
        window_size = min(time_steps, 5)
        num_windows = time_steps // window_size
        original_time = time_steps
        if time_steps % window_size != 0:
            pad_length = window_size - (time_steps % window_size)
            x = F.pad(x, (0, 0, 0, pad_length))
            time_steps = time_steps + pad_length
            num_windows = time_steps // window_size
        x = x.view(batch, chans, num_windows, window_size, features)
        x = x.permute(0, 3, 1, 2, 4).reshape(batch * window_size * chans, num_windows, features)
        x = self.inter_window_attn(x, x, x, attn_mask=None, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x.reshape(batch, window_size, chans, num_windows, features).permute(0, 2, 3, 1, 4)
        x = x.reshape(batch, chans, time_steps, features)
        if time_steps != original_time:
            x = x[:, :, :original_time, :]
        return self.dropout2(x)

    def _ff_block(self, x):
        batch, chans, time_steps, features = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * time_steps, chans, features)
        x_ff = self.linear2(self.dropout(self.activation(self.linear1(x_reshaped))))
        x_ff = x_ff.reshape(batch, time_steps, chans, features).permute(0, 2, 1, 3)
        return self.dropout3(x_ff)


class CSBrainTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, area_config: dict, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: Optional[bool] = None) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, area_config, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TemEmbedEEGLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_sizes, stride=1):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, len(kernel_sizes))]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_scale, kernel_size=(kt, 1), stride=(stride, 1), padding=((kt - 1) // 2, 0))
            for (kt,), dim_scale in zip(kernel_sizes, dim_scales)
        ])

    def forward(self, x):
        batch, chans, time_steps, d_model = x.shape
        x = x.view(batch * chans, d_model, time_steps, 1)
        fmaps = [conv(x) for conv in self.convs]
        x = torch.cat(fmaps, dim=1)
        return x.view(batch, chans, time_steps, -1)


class BrainEmbedEEGLayer(nn.Module):
    def __init__(self, dim_in=200, dim_out=200, total_regions=5):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        kernel_sizes = [1, 3, 5]
        dim_scales = [dim_out // (2 ** (i + 1)) for i in range(len(kernel_sizes) - 1)]
        dim_scales.append(dim_out - sum(dim_scales))
        self.region_blocks = nn.ModuleDict({
            f"region_{i}": nn.ModuleList([
                nn.Conv2d(in_channels=dim_in, out_channels=dim_scale, kernel_size=(k, 1), padding=(0, 0), groups=1)
                for k, dim_scale in zip(kernel_sizes, dim_scales)
            ])
            for i in range(total_regions)
        })

    def forward(self, x, area_config):
        batch, chans, time_steps, features = x.shape
        output = torch.zeros((batch, chans, time_steps, self.dim_out), device=x.device, dtype=x.dtype)
        for region_key, region_info in area_config.items():
            if region_key not in self.region_blocks:
                continue
            channel_slice = region_info["slice"]
            n_electrodes = region_info["channels"]
            x_region = x[:, channel_slice, :, :]
            x_trans = x_region.permute(0, 2, 1, 3).reshape(-1, n_electrodes, features)
            x_trans = x_trans.permute(0, 2, 1).unsqueeze(-1)
            fmap_outputs = []
            for conv, k in zip(self.region_blocks[region_key], [1, 3, 5]):
                pad_size = (k - 1) // 2
                if n_electrodes == 1:
                    x_padded = F.pad(x_trans, (0, 0, pad_size, pad_size), mode="constant", value=0)
                else:
                    x_padded = F.pad(x_trans, (0, 0, pad_size, pad_size), mode="circular")
                fmap_outputs.append(conv(x_padded))
            fmap_cat = torch.cat(fmap_outputs, dim=1)
            fmap_out = fmap_cat.squeeze(-1).permute(0, 2, 1).reshape(batch, time_steps, n_electrodes, self.dim_out)
            output[:, channel_slice, :, :] = fmap_out.permute(0, 2, 1, 3)
        return output


class RegionAttentionMaskBuilder:
    def __init__(self, num_channels: int, area_config: dict, device=None):
        self.num_channels = num_channels
        self.area_config = area_config
        self.device = device
        self.region_indices_dict = self._process_region_indices()
        self.attention_mask = self._build_attention_mask()

    def _process_region_indices(self):
        region_indices_dict = {}
        for region_name, region_info in self.area_config.items():
            region_slice = region_info["slice"]
            if isinstance(region_slice, slice):
                start = region_slice.start or 0
                stop = region_slice.stop
                step = region_slice.step or 1
                region_indices = list(range(start, stop, step))
            else:
                region_indices = list(region_slice)
            region_indices_dict[region_name] = region_indices
        return region_indices_dict

    def _build_attention_mask(self):
        device = self.device if self.device is not None else torch.device("cpu")
        region_attn_mask = torch.ones(self.num_channels, self.num_channels, device=device) * float("-inf")
        num_groups = max(len(indices) for indices in self.region_indices_dict.values())
        groups = [[] for _ in range(num_groups)]
        for group_id in range(num_groups):
            for _, region_indices in self.region_indices_dict.items():
                if len(region_indices) == 0:
                    continue
                groups[group_id].append(region_indices[group_id % len(region_indices)])
        for group in groups:
            for idx1 in group:
                for idx2 in group:
                    region_attn_mask[idx1, idx2] = 0
        return region_attn_mask

    def get_mask(self):
        return self.attention_mask

    def get_region_indices(self):
        return self.region_indices_dict


def generate_area_config(brain_regions):
    region_to_channels = defaultdict(list)
    for channel_idx, region in enumerate(brain_regions):
        region_to_channels[region].append(channel_idx)
    area_config = {}
    for region, channels in region_to_channels.items():
        area_config[f"region_{region}"] = {"channels": len(channels), "slice": slice(channels[0], channels[-1] + 1)}
    return area_config


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
