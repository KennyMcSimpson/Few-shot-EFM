# --------------------------------------------------------
# LoRA utilities for AdaBrain-Bench fine-tuning experiments.
# This file is intentionally independent from the four EEGFM backbones.
# It injects trainable low-rank adapters into existing modules without
# modifying the original model source files.
# --------------------------------------------------------

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_contracts import resolve_backbone_bd_sites
from .module_b_signal_alignment import (
    LoRAConv1d1x1,
    install_input_side_lora,
    is_module_b_target,
    should_lora_bridge,
    should_lora_input_side,
)
from .module_c_lora_search import parse_module_ids
from .module_d_semantic_refinement import (
    layer_selected_for_semantic,
    max_semantic_layer_index,
    should_lora_semantic_ffn,
)
from .module_e_structural_routing import (
    is_module_e_spatial_attention_target,
    is_module_e_structural_mixing_target,
    is_module_e_temporal_attention_target,
    should_lora_structural_routing,
)

LEGACY_FFN_BASELINE_TARGETS = frozenset(
    (
        "qv_ffn",
        "qkvo_ffn",
        "attn_ffn",
        "spatial_attn_ffn",
        "temporal_attn_ffn",
        "bridge_ffn",
        "bridge_ffn_last2",
        "bridge_last2ffn_pure",
    )
)


class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.

    Forward:
        y = base_linear(x) + scaling * B(A(dropout(x)))

    The original linear layer is kept inside this wrapper and frozen. Only
    lora_A/lora_B are trainable.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_runtime_scale = 1.0
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + lora_out * self.scaling * float(getattr(self, "lora_runtime_scale", 1.0))


class LoRAMergedQKVLinear(nn.Module):
    """
    LoRA wrapper for merged QKV projection, e.g. ViT-style attn.qkv.

    The wrapped Linear has out_features = 3 * hidden_dim. This module allows
    selective LoRA on q/k/v chunks. Default usage in this project is q+v.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        enable_lora: Sequence[str] = ("q", "v"),
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRAMergedQKVLinear expects nn.Linear, got {type(base)}")
        if base.out_features % 3 != 0:
            raise ValueError(
                f"Merged QKV Linear should have out_features divisible by 3, got {base.out_features}"
            )
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_runtime_scale = 1.0
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.chunk_size = base.out_features // 3
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.enable_lora = tuple([x.lower() for x in enable_lora])
        self.chunk_names = ("q", "k", "v")

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in self.chunk_names:
            if name in self.enable_lora:
                self.lora_A[name] = nn.Parameter(torch.empty(self.r, self.in_features))
                self.lora_B[name] = nn.Parameter(torch.zeros(self.chunk_size, self.r))

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        for name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name])

    def _delta_weight(self) -> torch.Tensor:
        """
        Build the merged QKV LoRA delta weight with the same shape as base.weight.

        Some AdaBrain-Bench backbones, especially LaBraM, do not call
        self.qkv(x). They directly call F.linear(x, self.qkv.weight, bias).
        Therefore this wrapper must expose a differentiable .weight property.
        """
        pieces = []
        base_weight = self.base.weight
        for name in self.chunk_names:
            if name in self.lora_A:
                delta = self.lora_B[name] @ self.lora_A[name]
                delta = delta.to(device=base_weight.device, dtype=base_weight.dtype) * self.scaling * float(getattr(self, "lora_runtime_scale", 1.0))
            else:
                delta = torch.zeros(
                    self.chunk_size,
                    self.in_features,
                    device=base_weight.device,
                    dtype=base_weight.dtype,
                )
            pieces.append(delta)
        return torch.cat(pieces, dim=0)

    @property
    def weight(self) -> torch.Tensor:
        # This makes LaBraM-style F.linear(x, self.qkv.weight, ...) work.
        return self.base.weight + self._delta_weight()

    @property
    def bias(self):
        # Preserve nn.Linear-like API. Some backbones read self.qkv.bias.
        return self.base.bias

    def forward(self, x):
        # When the backbone calls self.qkv(x), use the same effective weight.
        # Note: for backbones that directly read .weight, lora_dropout cannot be
        # applied without editing the backbone attention forward, so this baseline
        # treats merged-QKV LoRA as deterministic weight adaptation.
        return F.linear(x, self.weight, self.bias)


class LoRAMultiheadAttention(nn.Module):
    """
    LoRA wrapper for torch.nn.MultiheadAttention.

    This is used for CBraMod, where attention is implemented by nn.MultiheadAttention
    and Q/K/V are stored in a merged in_proj_weight parameter rather than separate
    nn.Linear submodules.

    Supported first-round target: q/v; optional k and o are also supported.
    """

    def __init__(
        self,
        base: nn.MultiheadAttention,
        r: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.0,
        enable_lora: Sequence[str] = ("q", "v"),
    ):
        super().__init__()
        if not isinstance(base, nn.MultiheadAttention):
            raise TypeError(f"LoRAMultiheadAttention expects nn.MultiheadAttention, got {type(base)}")
        if not getattr(base, "_qkv_same_embed_dim", True):
            raise NotImplementedError("LoRA wrapper currently supports only same q/k/v embed dim MHA.")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_runtime_scale = 1.0
        self.embed_dim = base.embed_dim
        self.enable_lora = tuple([x.lower() for x in enable_lora])
        self.dropout_layer = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in ("q", "k", "v", "o"):
            if name in self.enable_lora:
                self.lora_A[name] = nn.Parameter(torch.empty(self.r, self.embed_dim))
                self.lora_B[name] = nn.Parameter(torch.zeros(self.embed_dim, self.r))

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        for name in self.lora_A.keys():
            nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name])

    def _delta_weight(self, name: str, like: torch.Tensor) -> torch.Tensor:
        if name not in self.lora_A:
            return torch.zeros(self.embed_dim, self.embed_dim, device=like.device, dtype=like.dtype)
        delta = self.lora_B[name] @ self.lora_A[name]
        return delta.to(device=like.device, dtype=like.dtype) * self.scaling * float(getattr(self, "lora_runtime_scale", 1.0))

    def forward(self, query, key, value, **kwargs):
        # nn.MultiheadAttention handles batch_first internally by transposing before
        # F.multi_head_attention_forward. We reproduce that behavior because we need
        # to pass an effective in_proj_weight.
        is_batched = query.dim() == 3
        batch_first = self.base.batch_first

        if batch_first and is_batched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        in_w = self.base.in_proj_weight
        q_w, k_w, v_w = in_w.chunk(3, dim=0)
        q_w = q_w + self._delta_weight("q", in_w)
        k_w = k_w + self._delta_weight("k", in_w)
        v_w = v_w + self._delta_weight("v", in_w)
        effective_in_w = torch.cat([q_w, k_w, v_w], dim=0)

        out_w = self.base.out_proj.weight
        effective_out_w = out_w + self._delta_weight("o", out_w)

        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.base.embed_dim,
            num_heads=self.base.num_heads,
            in_proj_weight=effective_in_w,
            in_proj_bias=self.base.in_proj_bias,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.base.dropout,
            out_proj_weight=effective_out_w,
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=kwargs.get("key_padding_mask", None),
            need_weights=kwargs.get("need_weights", True),
            attn_mask=kwargs.get("attn_mask", None),
            use_separate_proj_weight=False,
            average_attn_weights=kwargs.get("average_attn_weights", True),
            is_causal=kwargs.get("is_causal", False),
        )

        if batch_first and is_batched:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


@dataclass(frozen=True)
class LoraTargetPlan:
    target_lower: str
    enabled_parts: Tuple[str, ...]
    use_ffn: bool
    use_all_linear: bool
    use_structural: bool
    use_bridge: bool
    use_input_side: bool
    module_b_sites: str
    spatial_only: bool
    temporal_only: bool
    spatial_branch_target: bool
    temporal_branch_target: bool
    module_c_selected: Tuple[str, ...] = ()


def freeze_all_parameters(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = True


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _get_module_by_name(root: nn.Module, module_name: str) -> Optional[nn.Module]:
    if module_name == "":
        return root
    module = root
    try:
        for part in module_name.split("."):
            module = getattr(module, part)
        return module
    except AttributeError:
        return None


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, child_name = _get_parent_module(root, module_name)
    setattr(parent, child_name, new_module)


def _record_replacement(root: nn.Module, module_name: str, new_module: nn.Module, replaced: List[str]) -> nn.Module:
    _replace_module(root, module_name, new_module)
    replaced.append(module_name)
    return new_module


def _replace_with_lora_linear(
    root: nn.Module,
    module_name: str,
    module: nn.Linear,
    r: int,
    alpha: float,
    dropout: float,
    replaced: List[str],
) -> LoRALinear:
    return _record_replacement(
        root,
        module_name,
        LoRALinear(module, r=r, alpha=alpha, dropout=dropout),
        replaced,
    )


def _is_module_c_execution_target(lora_target: str) -> bool:
    return str(lora_target or "").lower() in ("module_c", "module_c_auto", "c_auto")


def _module_c_selected_ids(module_c_selected: Optional[str | Iterable[str]]) -> Tuple[str, ...]:
    return tuple(parse_module_ids(module_c_selected or ()))


def _target_to_enabled_parts(lora_target: str, module_c_selected: Optional[str | Iterable[str]] = None) -> List[str]:
    target = lora_target.lower()
    if _is_module_c_execution_target(target):
        selected = set(_module_c_selected_ids(module_c_selected))
        return ["q", "v"] if "E" in selected else []
    if target in ("qv", "attn_qv", "qv_ffn"):
        return ["q", "v"]
    if target in ("spatial_attn_ffn", "temporal_attn_ffn"):
        return ["q", "v"]
    if target in ("qkv", "attn_qkv"):
        return ["q", "k", "v"]
    if target in ("qkvo", "attn", "attention", "qkvo_ffn", "attn_ffn", "all_linear"):
        return ["q", "k", "v", "o"]
    if should_lora_structural_routing(target):
        # For structural/mixing LoRA, use q/v by default to adapt attention routing
        # without directly rewriting FFN semantic blocks.
        return ["q", "v"]
    if target in ("ffn", "mlp", "sem", "semantic", "fb_sem",
                "ffn_late", "ffn_tophalf", "ffn_last2", "ffn_last1", "bridge", "input_bridge", "front", "bridge_ffn", "bridge_ffn_last2", "bridge_last2ffn_pure") or is_module_b_target(target):
        return []
    return ["q", "v"]


def _should_lora_ffn(lora_target: str, module_c_selected: Optional[str | Iterable[str]] = None) -> bool:
    target = str(lora_target or "").lower()
    if _is_module_c_execution_target(lora_target):
        return "D" in set(_module_c_selected_ids(module_c_selected))
    if target in LEGACY_FFN_BASELINE_TARGETS:
        return True
    return should_lora_semantic_ffn(target)


def _should_lora_all_linear(lora_target: str) -> bool:
    return lora_target.lower() == "all_linear"


def _should_lora_structural(lora_target: str, module_c_selected: Optional[str | Iterable[str]] = None) -> bool:
    if _is_module_c_execution_target(lora_target):
        return "E" in set(_module_c_selected_ids(module_c_selected))
    return is_module_e_structural_mixing_target(lora_target)


def _should_lora_bridge(
    lora_target: str,
    module_c_selected: Optional[str | Iterable[str]] = None,
    module_b_sites: str = "both",
) -> bool:
    target = lora_target.lower()
    if _is_module_c_execution_target(target):
        return "B" in set(_module_c_selected_ids(module_c_selected)) and should_lora_bridge("signal_align", module_b_sites)
    return target in ("bridge", "input_bridge", "front",
                      "bridge_ffn", "bridge_ffn_last2", "bridge_last2ffn_pure") or should_lora_bridge(target, module_b_sites)


def _is_spatial_attn_only(lora_target: str) -> bool:
    return lora_target.lower() == "spatial_attn"


def _is_temporal_attn_only(lora_target: str) -> bool:
    return lora_target.lower() == "temporal_attn"


# 这两个名字只给旧 baseline 兼容，Module E 正式接口不再用它们。
def _is_legacy_spatial_attn_ffn(lora_target: str) -> bool:
    return str(lora_target or "").lower() == "spatial_attn_ffn"


def _is_legacy_temporal_attn_ffn(lora_target: str) -> bool:
    return str(lora_target or "").lower() == "temporal_attn_ffn"


def _ffn_layer_mode(lora_target: str) -> str:
    from .module_d_semantic_refinement import semantic_layer_mode

    return semantic_layer_mode(lora_target)


def _extract_layer_index(module_name: str) -> Optional[int]:
    """Best-effort parser for transformer layer indices in common EEGFM names."""
    from .module_d_semantic_refinement import extract_semantic_layer_index

    return extract_semantic_layer_index(module_name)


def _max_layer_index(model: nn.Module) -> Optional[int]:
    return max_semantic_layer_index(model)


def _layer_selected_for_ffn(module_name: str, lora_target: str, max_idx: Optional[int]) -> bool:
    target = "semantic" if _is_module_c_execution_target(lora_target) else str(lora_target or "").lower()
    if target in LEGACY_FFN_BASELINE_TARGETS:
        return True
    return layer_selected_for_semantic(module_name, target, max_idx)


def _should_lora_input_side(
    lora_target: str,
    module_c_selected: Optional[str | Iterable[str]] = None,
    module_b_sites: str = "both",
) -> bool:
    if _is_module_c_execution_target(lora_target):
        return "B" in set(_module_c_selected_ids(module_c_selected)) and should_lora_input_side("signal_align", module_b_sites)
    return should_lora_input_side(lora_target, module_b_sites)


def _install_input_side_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> List[str]:
    return install_input_side_lora(model, r=r, alpha=alpha, dropout=dropout)


def _lora_target_plan(
    lora_target: str,
    module_c_selected: Optional[str | Iterable[str]] = None,
    module_b_sites: str = "both",
) -> LoraTargetPlan:
    target_lower = str(lora_target or "").lower()
    selected = _module_c_selected_ids(module_c_selected) if _is_module_c_execution_target(target_lower) else tuple()
    return LoraTargetPlan(
        target_lower=target_lower,
        enabled_parts=tuple(_target_to_enabled_parts(target_lower, selected)),
        use_ffn=_should_lora_ffn(target_lower, selected),
        use_all_linear=_should_lora_all_linear(target_lower),
        use_structural=_should_lora_structural(target_lower, selected),
        use_bridge=_should_lora_bridge(target_lower, selected, module_b_sites),
        use_input_side=_should_lora_input_side(target_lower, selected, module_b_sites),
        module_b_sites=str(module_b_sites or "both"),
        spatial_only=_is_spatial_attn_only(target_lower),
        temporal_only=_is_temporal_attn_only(target_lower),
        spatial_branch_target=(
            is_module_e_spatial_attention_target(target_lower)
            or _is_legacy_spatial_attn_ffn(target_lower)
        ),
        temporal_branch_target=(
            is_module_e_temporal_attention_target(target_lower)
            or _is_legacy_temporal_attn_ffn(target_lower)
        ),
        module_c_selected=selected,
    )


def _apply_lora_to_biot_module(
    model: nn.Module,
    name: str,
    module: nn.Module,
    plan: LoraTargetPlan,
    lora_target: str,
    max_layer_idx: Optional[int],
    r: int,
    alpha: float,
    dropout: float,
    replaced: List[str],
) -> None:
    lower = name.lower()
    if isinstance(module, nn.Linear):
        should_replace = False
        if lower.endswith("to_q") and "q" in plan.enabled_parts:
            should_replace = True
        elif lower.endswith("to_k") and "k" in plan.enabled_parts:
            should_replace = True
        elif lower.endswith("to_v") and "v" in plan.enabled_parts:
            should_replace = True
        elif lower.endswith("to_out") and "o" in plan.enabled_parts:
            should_replace = True
        elif plan.use_all_linear and lower.startswith("main_model."):
            should_replace = True

        if should_replace:
            _replace_with_lora_linear(model, name, module, r, alpha, dropout, replaced)


def _apply_lora_to_cbramod_module(
    model: nn.Module,
    name: str,
    module: nn.Module,
    plan: LoraTargetPlan,
    lora_target: str,
    max_layer_idx: Optional[int],
    r: int,
    alpha: float,
    dropout: float,
    replaced: List[str],
) -> None:
    lower = name.lower()
    if isinstance(module, nn.MultiheadAttention) and ("self_attn_s" in lower or "self_attn_t" in lower):
        if plan.spatial_branch_target and "self_attn_s" not in lower:
            return
        if plan.temporal_branch_target and "self_attn_t" not in lower:
            return
        if plan.spatial_branch_target or plan.temporal_branch_target or len(plan.enabled_parts) > 0:
            _record_replacement(
                model,
                name,
                LoRAMultiheadAttention(module, r=r, alpha=alpha, dropout=dropout, enable_lora=plan.enabled_parts),
                replaced,
            )
    elif isinstance(module, nn.Linear):
        should_replace = False
        if plan.use_all_linear and lower.startswith("main_model."):
            should_replace = True
        if should_replace:
            _replace_with_lora_linear(model, name, module, r, alpha, dropout, replaced)


def _apply_lora_to_eegpt_labram_module(
    model: nn.Module,
    name: str,
    module: nn.Module,
    plan: LoraTargetPlan,
    lora_target: str,
    max_layer_idx: Optional[int],
    r: int,
    alpha: float,
    dropout: float,
    replaced: List[str],
) -> None:
    lower = name.lower()
    if isinstance(module, nn.Linear):
        should_replace = False
        new_module = None

        if lower.endswith("attn.qkv"):
            qkv_parts = [p for p in plan.enabled_parts if p in ("q", "k", "v")]
            if qkv_parts:
                should_replace = True
                new_module = LoRAMergedQKVLinear(
                    module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    enable_lora=qkv_parts,
                )
        elif lower.endswith("attn.proj") and "o" in plan.enabled_parts:
            should_replace = True
            new_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        elif plan.use_all_linear and lower.startswith("main_model."):
            should_replace = True
            new_module = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)

        if should_replace and new_module is not None:
            _record_replacement(model, name, new_module, replaced)


def _apply_lora_to_csbrain_module(
    model: nn.Module,
    name: str,
    module: nn.Module,
    plan: LoraTargetPlan,
    lora_target: str,
    max_layer_idx: Optional[int],
    r: int,
    alpha: float,
    dropout: float,
    replaced: List[str],
) -> None:
    lower = name.lower()
    if isinstance(module, nn.MultiheadAttention) and ("inter_window_attn" in lower or "inter_region_attn" in lower):
        if plan.spatial_branch_target and "inter_region_attn" not in lower:
            return
        if plan.temporal_branch_target and "inter_window_attn" not in lower:
            return
        if plan.spatial_branch_target or plan.temporal_branch_target or len(plan.enabled_parts) > 0:
            _record_replacement(
                model,
                name,
                LoRAMultiheadAttention(module, r=r, alpha=alpha, dropout=dropout, enable_lora=plan.enabled_parts),
                replaced,
            )
    elif isinstance(module, nn.Linear):
        should_replace = False
        if plan.use_all_linear and lower.startswith("main_model."):
            should_replace = True
        if should_replace:
            _replace_with_lora_linear(model, name, module, r, alpha, dropout, replaced)


def _apply_lora_to_gram_module(
    model: nn.Module,
    name: str,
    module: nn.Module,
    plan: LoraTargetPlan,
    lora_target: str,
    max_layer_idx: Optional[int],
    r: int,
    alpha: float,
    dropout: float,
    replaced: List[str],
) -> None:
    lower = name.lower()
    if not isinstance(module, nn.Linear):
        return

    skip = any(k in lower for k in ("vqgan", "quant", "codebook", "decoder", "tokenizer"))
    should_replace = False
    if not skip:
        if plan.use_structural:
            is_attn_linear = (".attn." in lower and (
                lower.endswith("key")
                or lower.endswith("query")
                or lower.endswith("value")
                or lower.endswith("proj")
            ))
            is_layer_fusion = ("proj_layers" in lower)
            should_replace = is_attn_linear or is_layer_fusion
        if not should_replace and plan.use_all_linear and lower.startswith("main_model."):
            should_replace = True

    if should_replace:
        _replace_with_lora_linear(model, name, module, r, alpha, dropout, replaced)


LORA_MODEL_MODULE_HANDLERS = {
    "biot": _apply_lora_to_biot_module,
    "cbramod": _apply_lora_to_cbramod_module,
    "eegpt": _apply_lora_to_eegpt_labram_module,
    "labram": _apply_lora_to_eegpt_labram_module,
    "csbrain": _apply_lora_to_csbrain_module,
    "gram": _apply_lora_to_gram_module,
}


def apply_lora_to_eegfm(
    model: nn.Module,
    model_name: str,
    lora_target: str = "qv",
    module_c_selected: Optional[str | Iterable[str]] = None,
    module_e_allowed_names: Optional[Iterable[str]] = None,
    module_b_sites: str = "both",
    r: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.1,
    verbose: bool = True,
    module_c_seed: int = 0,
) -> List[str]:
    """
    Inject LoRA into AdaBrain-Bench EEGFM wrappers.

    Supported model_name:
      - BIOT: separate to_q/to_k/to_v/to_out Linear layers.
      - CBraMod: nn.MultiheadAttention modules self_attn_s/self_attn_t.
      - EEGPT: merged attn.qkv Linear and attn.proj Linear.
      - LaBraM: merged attn.qkv Linear and attn.proj Linear.
      - CSBrain: inter-window/inter-region MultiheadAttention and FFN Linear layers.
      - Gram: explicit encoder-MLP B/D sites plus existing structural Linear routing.

    lora_target:
      - qv: first-round recommended baseline. Q/V only.
      - qkv: Q/K/V.
      - qkvo: Q/K/V/O.
      - ffn/mlp: FFN/MLP layers only.
      - qv_ffn: Q/V + FFN/MLP layers.
      - qkvo_ffn / attn_ffn: Q/K/V/O + FFN/MLP layers.
      - all_linear: broader ablation; not recommended as first run.
    """
    model_name = model_name.lower()
    is_module_c = _is_module_c_execution_target(lora_target)
    cpu_rng_state = torch.get_rng_state().clone() if is_module_c else None
    cuda_was_initialized = bool(is_module_c and torch.cuda.is_initialized())
    cuda_rng_state = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if cuda_was_initialized
        else None
    )
    try:
        plan = _lora_target_plan(
            lora_target,
            module_c_selected=module_c_selected,
            module_b_sites=module_b_sites,
        )
        max_layer_idx = _max_layer_index(model)
        handler = LORA_MODEL_MODULE_HANDLERS.get(model_name)

        replaced: List[str] = []

        bridge_paths = {
            site.module_path
            for site in resolve_backbone_bd_sites(model, model_name, "B")
        } if plan.use_bridge else set()
        semantic_paths = {
            site.module_path
            for site in resolve_backbone_bd_sites(model, model_name, "D")
        } if plan.use_ffn else set()

        if plan.use_input_side:
            replaced.extend(_install_input_side_lora(model, r=r, alpha=alpha, dropout=dropout))

        # Use a snapshot because we will replace modules during iteration.
        for name, module in list(model.named_modules()):
            if name == "":
                continue

            current_module = _get_module_by_name(model, name)
            if current_module is not module:
                continue

            if name in bridge_paths:
                _record_replacement(
                    model,
                    name,
                    LoRAConv1d1x1(module, r=r, alpha=alpha, dropout=dropout),
                    replaced,
                )
                continue

            if name in semantic_paths and _layer_selected_for_ffn(name, lora_target, max_layer_idx):
                _replace_with_lora_linear(model, name, module, r, alpha, dropout, replaced)
                continue

            if handler is not None:
                handler(model, name, module, plan, lora_target, max_layer_idx, r, alpha, dropout, replaced)

        if is_module_c:
            with torch.no_grad():
                for full_name, parameter in model.named_parameters():
                    name_parts = full_name.split(".")
                    if "lora_A" in name_parts:
                        seed_material = f"{int(module_c_seed)}\0{full_name}".encode("utf-8")
                        stable_seed = int.from_bytes(
                            hashlib.sha256(seed_material).digest()[:8], "big", signed=False
                        )
                        generator = torch.Generator(device=parameter.device)
                        generator.manual_seed(stable_seed)
                        nn.init.kaiming_uniform_(
                            parameter, a=math.sqrt(5), generator=generator
                        )
                    elif "lora_B" in name_parts:
                        nn.init.zeros_(parameter)
    finally:
        if is_module_c:
            torch.set_rng_state(cpu_rng_state)
            if cuda_was_initialized:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    if verbose:
        print(f"[LoRA] model={model_name}, target={lora_target}, r={r}, alpha={alpha}, dropout={dropout}")
        print(f"[LoRA] injected modules: {len(replaced)}")
        for n in replaced[:80]:
            print(f"  [LoRA] {n}")
        if len(replaced) > 80:
            print(f"  [LoRA] ... {len(replaced) - 80} more modules")

    return replaced


def mark_lora_and_selected_modules_trainable(
    model: nn.Module,
    train_task_head: bool = True,
    train_chan_conv: bool = False,
) -> None:
    """
    After freeze_all_parameters + LoRA injection, ensure LoRA params and selected
    non-backbone modules are trainable.
    """
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if train_task_head and hasattr(model, "task_head"):
        unfreeze_module(model.task_head)

    if train_chan_conv and hasattr(model, "chan_conv"):
        unfreeze_module(model.chan_conv)



def set_lora_runtime_scale(model: nn.Module, scale: float, verbose: bool = False) -> int:
    """Set inference-time multiplier for all LoRA branches.

    This does not change trained weights. It only changes how strongly LoRA
    deltas are used in forward(), enabling post-hoc adapter strength calibration.
    """
    n = 0
    for module in model.modules():
        if hasattr(module, "lora_runtime_scale"):
            module.lora_runtime_scale = float(scale)
            n += 1
    if verbose:
        print(f"[LoRA] runtime scale set to {float(scale):.4f} for {n} LoRA modules")
    return n


def get_lora_runtime_scale(model: nn.Module) -> float:
    for module in model.modules():
        if hasattr(module, "lora_runtime_scale"):
            return float(getattr(module, "lora_runtime_scale", 1.0))
    return 1.0


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def print_trainable_parameters(model: nn.Module, max_lines: int = 120) -> None:
    stats = count_trainable_parameters(model)
    ratio = 100.0 * stats["trainable"] / max(stats["total"], 1)
    print(
        f"[LoRA] trainable params: {stats['trainable']} / {stats['total']} "
        f"({ratio:.4f}%), frozen: {stats['frozen']}"
    )

    shown = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  [trainable] {name}: {tuple(p.shape)}")
            shown += 1
            if shown >= max_lines:
                remaining = sum(1 for _, pp in model.named_parameters() if pp.requires_grad) - shown
                if remaining > 0:
                    print(f"  [trainable] ... {remaining} more tensors")
                break
