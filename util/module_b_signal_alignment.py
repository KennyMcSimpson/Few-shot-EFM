# --------------------------------------------------------
# Module B: signal-alignment / input-front LoRA utilities.
#
# Module B is the input-front accessibility action in the current EEG
# framework. It keeps the EEG foundation model intact and learns a small
# low-rank correction before, or at, the channel-conversion front end.
# --------------------------------------------------------

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_B_TARGET = "signal_align"
MODULE_B_LEGACY_ALIASES = frozenset(
    (
        "sig",
        "signal",
        "front_align",
        "input_side",
        "channel_adapter",
    )
)
MODULE_B_TARGETS = frozenset((MODULE_B_TARGET, *MODULE_B_LEGACY_ALIASES))
MODULE_B_CURRENT = "signal_alignment_input_front"
MODULE_B_ROLE = "input_front_accessibility"
MODULE_B_SITES = frozenset(("both", "input", "bridge"))


def _module_c_selected_modules(args: Optional[Any]) -> set[str]:
    if args is None:
        return set()
    selected = getattr(args, "module_c_resolved_selected", "") or getattr(args, "module_c_selected", "")
    out = set()
    for token in str(selected or "").replace(";", ",").replace("|", ",").split(","):
        token = token.strip().upper()
        if token:
            out.add(token)
    return out


def normalize_lora_target(lora_target: Any) -> str:
    return str(lora_target or "").lower()


def is_module_b_target(lora_target: Any) -> bool:
    return normalize_lora_target(lora_target) in MODULE_B_TARGETS


def normalize_module_b_sites(module_b_sites: Any = "both") -> str:
    text = str(module_b_sites or "both").strip().lower()
    aliases = {
        "all": "both",
        "input_side": "input",
        "input-only": "input",
        "input_only": "input",
        "bridge-only": "bridge",
        "bridge_only": "bridge",
        "chan_conv": "bridge",
        "channel_bridge": "bridge",
    }
    text = aliases.get(text, text)
    if text not in MODULE_B_SITES:
        raise ValueError(f"Unknown module_b_sites={module_b_sites!r}; expected one of {sorted(MODULE_B_SITES)}")
    return text


def should_lora_input_side(lora_target: Any, module_b_sites: Any = "both") -> bool:
    """Whether this target should install the raw-input residual adapter."""
    return is_module_b_target(lora_target) and normalize_module_b_sites(module_b_sites) in ("both", "input")


def should_lora_bridge(lora_target: Any, module_b_sites: Any = "both") -> bool:
    """Whether this target should wrap the existing 1x1 channel bridge."""
    return is_module_b_target(lora_target) and normalize_module_b_sites(module_b_sites) in ("both", "bridge")


def module_b_metadata(
    args: Optional[Any] = None,
    lora_target: Optional[Any] = None,
    lora_base_update: Optional[Any] = None,
    module_b_sites: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return shared Module B metadata for logs, probes, and collected outputs."""
    target = lora_target
    base_update = lora_base_update
    sites = module_b_sites
    if args is not None:
        if target is None:
            target = getattr(args, "lora_target", "")
            if normalize_lora_target(target) in ("module_c", "module_c_auto", "c_auto") and "B" in _module_c_selected_modules(args):
                target = MODULE_B_TARGET
        if base_update is None:
            base_update = getattr(args, "lora_base_update", "")
        if sites is None:
            sites = getattr(args, "module_b_sites", "both")

    target = str(target or "")
    base_update = str(base_update or "")
    active = int(is_module_b_target(target))
    sites = normalize_module_b_sites(sites or "both") if active else ""
    pure = int(bool(active) and base_update.lower() == "freeze")

    return {
        "module_b_current": MODULE_B_CURRENT if active else "",
        "module_b_role": MODULE_B_ROLE if active else "",
        "module_b_is_active": active,
        "module_b_is_pure_isolation": pure,
        "module_b_sites": sites,
        "module_b_input_side_active": int(bool(active) and sites in ("both", "input")),
        "module_b_bridge_active": int(bool(active) and sites in ("both", "bridge")),
        "module_b_attribution_note": (
            "pure_frozen_b_isolation"
            if pure
            else (
                "full_ft_plus_lora_confounded"
                if active and base_update.lower() == "full"
                else ""
            )
        ),
        "adapter_target": target,
        "lora_base_update": base_update,
    }


class LoRAConv1d1x1(nn.Module):
    """
    LoRA wrapper for 1x1 Conv1d / Conv1dWithConstraint channel adapters.

    This is mainly used as an input-bridge LoRA for EEGPT / BIOT style
    channel-conversion front-ends:
        y = Conv1d_base(x) + scaling * B(A(x))
    where A and B are 1x1 convolutions represented by trainable parameters.
    """

    def __init__(self, base: nn.Conv1d, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Conv1d):
            raise TypeError(f"LoRAConv1d1x1 expects nn.Conv1d, got {type(base)}")
        if tuple(base.kernel_size) != (1,):
            raise ValueError(f"LoRAConv1d1x1 only supports kernel_size=1, got {base.kernel_size}")
        if tuple(base.groups if isinstance(base.groups, tuple) else (base.groups,)) != (1,):
            raise ValueError("LoRAConv1d1x1 currently supports groups=1 only.")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_runtime_scale = 1.0
        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_channels, 1))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, self.r, 1))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def effective_delta_weight(self):
        """Return the actual low-rank channel update W_delta = B @ A."""
        return (
            self.lora_B.squeeze(-1) @ self.lora_A.squeeze(-1)
        ) * self.scaling * float(getattr(self, "lora_runtime_scale", 1.0))

    def forward(self, x):
        base_out = self.base(x)
        hidden = F.conv1d(self.dropout(x), self.lora_A, bias=None, stride=1, padding=0, dilation=1, groups=1)
        lora_out = F.conv1d(hidden, self.lora_B, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return base_out + lora_out * self.scaling * float(getattr(self, "lora_runtime_scale", 1.0))


class InputSideLoRAResidual(nn.Module):
    """
    Input-side LoRA residual adapter:
        x' = x + scaling * B(A(dropout(x)))

    It is placed before the model forward, so it can adapt raw EEG channel
    distribution before the model's chan_conv/tokenization. This keeps the
    method inside the LoRA family while moving the adapter to a structural
    input-side location.
    """

    def __init__(self, channels: int, r: int = 8, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        if channels <= 0:
            raise ValueError("InputSideLoRAResidual needs positive channel count.")
        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")
        self.channels = int(channels)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.lora_runtime_scale = 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.r, self.channels, 1))
        self.lora_B = nn.Parameter(torch.zeros(self.channels, self.r, 1))
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def effective_delta_weight(self):
        """Return the actual input-channel update W_delta = B @ A."""
        return (
            self.lora_B.squeeze(-1) @ self.lora_A.squeeze(-1)
        ) * self.scaling * float(getattr(self, "lora_runtime_scale", 1.0))

    def delta(self, x):
        if x.dim() != 3:
            return torch.zeros_like(x)
        hidden = F.conv1d(self.dropout(x), self.lora_A, bias=None, stride=1, padding=0, dilation=1, groups=1)
        delta = F.conv1d(hidden, self.lora_B, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return delta * self.scaling * float(getattr(self, "lora_runtime_scale", 1.0))

    def forward(self, x):
        if x.dim() != 3:
            return x
        return x + self.delta(x)


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


def infer_input_channels(model: nn.Module) -> Optional[int]:
    """Infer raw EEG channel count for the input-side residual adapter."""
    channels = None

    # Preferred: top-level wrapper metadata injected by run_finetuning.get_models().
    if hasattr(model, "input_channels"):
        try:
            channels = int(getattr(model, "input_channels"))
        except Exception:
            channels = None

    # Common Ada wrappers: raw-channel converter before the backbone.
    if channels is None:
        chan_conv = _get_module_by_name(model, "chan_conv")
        if isinstance(chan_conv, nn.Conv1d):
            channels = int(chan_conv.in_channels)

    # Gram wrapper keeps channel names inside main_model.
    if channels is None and hasattr(model, "main_model") and hasattr(model.main_model, "ch_names"):
        try:
            channels = len(model.main_model.ch_names)
        except Exception:
            channels = None

    if channels is None and hasattr(model, "ch_names"):
        try:
            channels = len(model.ch_names)
        except Exception:
            channels = None

    if channels is None:
        # Fallback: first Conv1d input channel count.
        for _, m in model.named_modules():
            if isinstance(m, nn.Conv1d):
                channels = int(m.in_channels)
                break

    return channels


def install_input_side_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> List[str]:
    """Attach Module B's input-side residual before the model forward."""
    if hasattr(model, "input_side_lora"):
        return []

    channels = infer_input_channels(model)
    if channels is None:
        raise RuntimeError("Cannot infer input channels for input-side LoRA.")

    model.input_side_lora = InputSideLoRAResidual(channels=channels, r=r, alpha=alpha, dropout=dropout)

    if not hasattr(model, "_forward_without_input_side_lora"):
        original_forward = model.forward
        model._forward_without_input_side_lora = original_forward

        def _forward_with_input_side_lora(x, *args, **kwargs):
            x = model.input_side_lora(x)
            return original_forward(x, *args, **kwargs)

        model.forward = _forward_with_input_side_lora

    return ["input_side_lora"]
