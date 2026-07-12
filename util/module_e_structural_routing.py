# --------------------------------------------------------
# Module E: structural routing recalibration utilities.
#
# Module E is the structural / spatial-temporal / mixing adaptation action in
# the current EEG framework. It does not claim to cover every internal
# interaction path of every EEGFM. Instead, it exposes a coverage-audited
# structural routing interface and the SRP/SRR reference metrics.
# --------------------------------------------------------

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .fb_registry import classify_param_name


STRUCTURAL_ROUTING_BLOCKS: Tuple[str, str, str] = ("spatial", "temporal", "mixing")
NON_STRUCTURAL_CONFLICT_BLOCKS = frozenset(("input_front", "semantic", "readout", "restoration"))
EXPLICIT_STRUCTURAL_TOKENS = frozenset(
    (
        "attn",
        "attention",
        "qkv",
        "query",
        "key",
        "value",
        "fusion",
        "router",
        "gate",
        "self_attn",
        "inter_region",
        "inter_window",
        "proj_layers",
    )
)
MODULE_E_PURE_TARGETS = frozenset(
    (
        "str",
        "struct",
        "structural",
        "struct_mix",
        "mix",
        "spatial_attn",
        "temporal_attn",
    )
)

MODULE_E_CURRENT = "structural_routing_recalibration"
MODULE_E_ROLE = "spatial_temporal_mixing_route_calibration"
MODULE_E_METRIC_SRP = "structural_routing_pressure"
MODULE_E_METRIC_SRR = "structural_routing_release"
MODULE_E_METRIC_ESC = "e_structural_coverage"
MODULE_E_PRESSURE_PROXY_FILE = "module_e_structural_pressure.csv"
MODULE_E_DYNAMIC_PRESSURE_FILE = "module_e_dynamic_pressure.csv"
MODULE_E_LORA_INJECTION_AUDIT_FILE = "module_e_lora_injection_audit.csv"
MODULE_E_DYNAMIC_PRESSURE_DEFINITION = "online_lora_b_gradient_energy_mean_by_structural_branch"
MODULE_E_MODE_DYNAMIC = "dynamic_pressure_gate"
MODULE_E_MODES = (MODULE_E_MODE_DYNAMIC,)


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


def module_e_mode_from_args(args: Any) -> str:
    """Resolve Module E execution mode.

    Only the formal dynamic pressure-gated structural LoRA path is exposed.
    Unsupported labels fall back to this path so custom callers cannot trigger
    hidden Module E branches.
    """
    mode = str(getattr(args, "module_e_mode", "") or "").strip().lower()
    if mode:
        if mode not in MODULE_E_MODES:
            return MODULE_E_MODE_DYNAMIC
        return mode
    return MODULE_E_MODE_DYNAMIC


# Module E only accepts pure structural / spatial / temporal targets.
def is_module_e_pure_target(lora_target: Any) -> bool:
    return normalize_lora_target(lora_target) in MODULE_E_PURE_TARGETS


def is_module_e_composite_target(lora_target: Any) -> bool:
    return False


def is_module_e_target(lora_target: Any) -> bool:
    return is_module_e_pure_target(lora_target)


def is_module_e_structural_mixing_target(lora_target: Any) -> bool:
    return normalize_lora_target(lora_target) in ("str", "struct", "structural", "struct_mix", "mix")


def is_module_e_spatial_attention_target(lora_target: Any) -> bool:
    return normalize_lora_target(lora_target) == "spatial_attn"


def is_module_e_temporal_attention_target(lora_target: Any) -> bool:
    return normalize_lora_target(lora_target) == "temporal_attn"


def should_lora_structural_routing(lora_target: Any) -> bool:
    return is_module_e_target(lora_target)


def module_e_target_blocks(lora_target: Any) -> Tuple[str, ...]:
    target = normalize_lora_target(lora_target)
    if target == "spatial_attn":
        return ("spatial",)
    if target == "temporal_attn":
        return ("temporal",)
    if target in MODULE_E_PURE_TARGETS:
        return STRUCTURAL_ROUTING_BLOCKS
    return tuple()


def module_e_variant(lora_target: Any) -> str:
    target = normalize_lora_target(lora_target)
    if target in ("str", "struct", "structural", "struct_mix", "mix"):
        return "structural_mixing"
    if target == "spatial_attn":
        return "spatial_attention"
    if target == "temporal_attn":
        return "temporal_attention"
    return ""


def module_e_metadata(
    args: Optional[Any] = None,
    lora_target: Optional[Any] = None,
    lora_base_update: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return shared Module E metadata for logs, probes, and collected outputs."""
    target = lora_target
    base_update = lora_base_update
    if args is not None:
        if target is None:
            target = getattr(args, "lora_target", "")
            if normalize_lora_target(target) in ("module_c", "module_c_auto", "c_auto") and "E" in _module_c_selected_modules(args):
                target = "struct_mix"
        if base_update is None:
            base_update = getattr(args, "lora_base_update", "")

    target = str(target or "")
    base_update = str(base_update or "")
    target_norm = normalize_lora_target(target)
    active = int(is_module_e_target(target_norm))
    pure = int(bool(active) and base_update.lower() == "freeze")
    composite = 0

    if pure:
        note = "pure_frozen_e_isolation"
    elif active and base_update.lower() == "full":
        note = "full_ft_plus_lora_confounded"
    else:
        note = ""

    return {
        "module_e_current": MODULE_E_CURRENT if active else "",
        "module_e_role": MODULE_E_ROLE if active else "",
        "module_e_is_active": active,
        "module_e_is_pure_isolation": pure,
        "module_e_is_composite": composite,
        "module_e_variant": module_e_variant(target_norm),
        "module_e_target_blocks": ",".join(module_e_target_blocks(target_norm)),
        "module_e_reference_metrics": "SRP,SRR,ESC" if active else "",
        "module_e_attribution_note": note,
        "adapter_target": target,
        "lora_base_update": base_update,
    }


def _safe_positive(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(out):
        return 0.0
    return max(0.0, out)


def _as_name_tuple(names: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    if names is None:
        return tuple()
    out = []
    seen = set()
    for raw in names:
        name = str(raw or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _parse_name_list(names: Optional[Any]) -> Tuple[str, ...]:
    if names is None:
        return tuple()
    if isinstance(names, str):
        text = names.replace("\n", ";").replace(",", ";")
        return _as_name_tuple(part for part in text.split(";"))
    return _as_name_tuple(names)


def _name_matches(candidate_name: str, injected_names: Sequence[str]) -> bool:
    for injected in injected_names:
        if candidate_name == injected:
            return True
        if candidate_name.startswith(injected + "."):
            return True
        if injected.startswith(candidate_name + "."):
            return True
    return False


def _structural_block_for_name(model_name: str, name: str) -> Optional[str]:
    primary, hits = classify_param_name(model_name, name)
    lower_name = str(name or "").lower()
    has_conflict = any(block in hits for block in NON_STRUCTURAL_CONFLICT_BLOCKS)
    has_explicit_structural_token = any(token in lower_name for token in EXPLICIT_STRUCTURAL_TOKENS)
    if has_conflict and not has_explicit_structural_token:
        return None
    if primary in STRUCTURAL_ROUTING_BLOCKS:
        return primary
    for block in STRUCTURAL_ROUTING_BLOCKS:
        if block in hits:
            return block
    return None


def _is_lora_adapter_param(name: str) -> bool:
    lower = str(name or "").lower()
    return ".lora_a" in lower or ".lora_b" in lower or lower.endswith("lora_a") or lower.endswith("lora_b")


def _is_weight_param(name: str) -> bool:
    lower = str(name or "").lower()
    return lower.endswith("weight") or lower.endswith("in_proj_weight")


def _module_prefix_from_param_name(name: str) -> str:
    text = str(name or "")
    lower = text.lower()
    for marker in (".lora_a", ".lora_b"):
        idx = lower.find(marker)
        if idx >= 0:
            return text[:idx]
    for suffix in (".base.weight", ".weight", ".in_proj_weight"):
        if lower.endswith(suffix):
            return text[: -len(suffix)]
    return text


def module_e_module_prefix_from_name(name: Any) -> str:
    """Map a parameter or adapter tensor name to the module that E can wrap."""
    return _module_prefix_from_param_name(str(name or "").strip())


def _pressure_for_name(name: str, pressure_by_name: Mapping[str, Any]) -> float:
    pressure = _safe_positive(pressure_by_name.get(name, 0.0))
    prefix = module_e_module_prefix_from_name(name)
    if prefix and prefix != name:
        pressure += _safe_positive(pressure_by_name.get(prefix, 0.0))
    dotted_prefix = prefix + "."
    counted = {name, prefix}
    for pressure_name, value in pressure_by_name.items():
        key = str(pressure_name or "")
        if key in counted:
            continue
        if key and key.startswith(dotted_prefix):
            pressure += _safe_positive(value)
    return pressure


def structural_inventory_from_model(
    model_name: str,
    model: Any,
    include_lora_adapters: bool = False,
) -> Tuple[str, ...]:
    """Return weight tensors that form Module E's audited structural inventory.

    The inventory is intentionally conservative: LoRA adapter tensors are not
    counted as candidate structural surface, and bias terms are skipped because
    Module E coverage is about low-rank routing-weight coverage.
    """
    if model is None or not hasattr(model, "named_parameters"):
        return tuple()

    names = []
    seen = set()
    for raw_name, _param in model.named_parameters():
        name = str(raw_name or "")
        if not name or name in seen:
            continue
        if not include_lora_adapters and _is_lora_adapter_param(name):
            continue
        if not _is_weight_param(name):
            continue
        if _structural_block_for_name(model_name, name) is None:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def module_e_branch_from_lora_param_name(model_name: str, param_name: Any) -> Optional[str]:
    """Infer the Module E structural branch for a LoRA tensor name."""
    name = str(param_name or "")
    prefix = module_e_module_prefix_from_name(name)
    branch = _structural_block_for_name(model_name, prefix)
    if branch is not None:
        return branch
    return _structural_block_for_name(model_name, name)


def _is_lora_b_param_name(name: Any) -> bool:
    lower = str(name or "").lower()
    return ".lora_b" in lower or lower.endswith("lora_b")


def _tensor_grad_energy_and_count(grad: Any) -> Tuple[float, int]:
    if grad is None:
        return 0.0, 0
    try:
        detached = grad.detach().float()
        energy = float(detached.pow(2).sum().cpu().item())
        count = int(detached.numel())
    except Exception:
        return 0.0, 0
    if not math.isfinite(energy) or count <= 0:
        return 0.0, 0
    return max(0.0, energy), count
# Formal dynamic E observes LoRA-B pressure and controls branch optimizer LRs.
class ModuleEDynamicPressureController:
    """Online pressure-gated LoRA update controller for Module E.

    Pressure is the mean squared unscaled-gradient energy on LoRA-B tensors.
    The resulting allocation is applied temporarily to isolated optimizer
    groups, leaving the gradients themselves unchanged.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "",
        run_tag: str = "",
        beta: float = 0.95,  # 这边是压力 EMA 的平滑系数
        temperature: float = 1.0,
        gate_floor: float = 0.2,
        scale_min: float = 0.5,
        scale_max: float = 1.5,
        warmup_steps: int = 0,
        diag_freq: int = 1,
        branches: Sequence[str] = STRUCTURAL_ROUTING_BLOCKS,
    ):
        self.model_name = str(model_name or "")
        self.output_dir = str(output_dir or "")
        self.run_tag = str(run_tag or "")
        self.beta = max(0.0, min(0.9999, float(beta)))
        self.temperature = max(1e-6, float(temperature))
        self.gate_floor = max(0.0, min(0.95, float(gate_floor)))
        self.scale_min = max(0.0, float(scale_min))
        self.scale_max = max(self.scale_min, float(scale_max))
        self.warmup_steps = max(0, int(warmup_steps))
        self.diag_freq = max(1, int(diag_freq))
        self.branches = tuple(str(branch) for branch in branches if str(branch) in STRUCTURAL_ROUTING_BLOCKS)
        if not self.branches:
            self.branches = STRUCTURAL_ROUTING_BLOCKS

        self._ema_pressure: Dict[str, float] = {branch: 0.0 for branch in self.branches}
        self._raw_pressure: Dict[str, float] = {branch: 0.0 for branch in self.branches}
        self._branch_scale: Dict[str, float] = {branch: 1.0 for branch in self.branches}
        self._pending_energy: Dict[str, float] = {branch: 0.0 for branch in self.branches}
        self._pending_count: Dict[str, int] = {branch: 0 for branch in self.branches}
        self._pending_param_tensors: Dict[str, int] = {branch: 0 for branch in self.branches}
        self._branch_param_names: Dict[str, List[str]] = {branch: [] for branch in self.branches}
        self._branch_pressure_param_names: Dict[str, List[str]] = {branch: [] for branch in self.branches}
        self._param_to_branch: Dict[str, str] = {}
        self._param_is_pressure: Dict[str, bool] = {}
        self._param_objects: Dict[str, Any] = {}
        self._param_name_by_id: Dict[int, str] = {}
        self._hook_handles: List[Any] = []
        self._expected_replacement_names: Tuple[str, ...] = tuple()
        self._bound_optimizer: Optional[Any] = None
        self._prepared_optimizer: Optional[Any] = None
        self._stored_group_lrs: List[Tuple[Dict[str, Any], float]] = []
        self._parameter_snapshots: Dict[str, Any] = {}
        self._prepared_global_step: Optional[int] = None
        self._prepared_epoch: Optional[int] = None
        self._prepared_entropy = 0.0
        self._allocation_degenerate = 0
        self._steps_seen = 0

    def register_param(
        self,
        param_name: Any,
        branch: Any,
        param: Any = None,
        is_pressure_param: bool = False,
    ) -> bool:
        if isinstance(param, bool) and is_pressure_param is False:
            is_pressure_param = param
            param = None
        branch = str(branch or "")
        if branch not in self.branches:
            return False
        name = str(param_name or "")
        if not name:
            return False
        if name not in self._param_to_branch:
            self._branch_param_names[branch].append(name)
        self._param_to_branch[name] = branch
        self._param_is_pressure[name] = bool(is_pressure_param)
        if param is not None:
            existing = self._param_name_by_id.get(id(param))
            if existing is not None and existing != name:
                raise RuntimeError(
                    f"Module E controlled parameter is registered under multiple names: {existing}, {name}."
                )
            self._param_objects[name] = param
            self._param_name_by_id[id(param)] = name
        if is_pressure_param and name not in self._branch_pressure_param_names[branch]:
            self._branch_pressure_param_names[branch].append(name)
        return True

    def controlled_parameter_names(self) -> Tuple[str, ...]:
        return tuple(self._param_to_branch)

    def branch_for_parameter(self, param: Any) -> Optional[str]:
        name = self._param_name_by_id.get(id(param))
        if name is None:
            return None
        return self._param_to_branch.get(name)

    def optimizer_group_tag(self, param_name: Any) -> str:
        branch = self._param_to_branch.get(str(param_name or ""))
        return f"module_e:{branch}" if branch is not None else "non_module_e"

    def branch_param_count(self, branch: Any) -> int:
        return len(self._branch_param_names.get(str(branch or ""), ()))

    def total_param_count(self) -> int:
        return len(self._param_to_branch)

    def total_pressure_param_count(self) -> int:
        return sum(len(names) for names in self._branch_pressure_param_names.values())

    def scale_for_branch(self, branch: Any) -> float:
        return float(self._branch_scale.get(str(branch or ""), 1.0))

    def scale_for_param(self, param_name: Any) -> float:
        branch = self._param_to_branch.get(str(param_name or ""), "")
        return self.scale_for_branch(branch)

    def record_gradient(self, branch: Any, grad: Any, is_pressure_param: bool = True) -> None:
        branch = str(branch or "")
        if branch not in self.branches or not is_pressure_param:
            return
        energy, count = _tensor_grad_energy_and_count(grad)
        if count <= 0:
            return
        self._pending_energy[branch] += energy
        self._pending_count[branch] += count
        self._pending_param_tensors[branch] += 1

    def make_gradient_hook(self, param_name: Any):
        name = str(param_name or "")
        branch = self._param_to_branch.get(name, "")
        is_pressure_param = bool(self._param_is_pressure.get(name, False))

        def _hook(grad):
            self.record_gradient(branch, grad, is_pressure_param=is_pressure_param)
            return grad

        return _hook

    def _active_branches(self) -> Tuple[str, ...]:
        return tuple(branch for branch in self.branches if self._branch_param_names.get(branch))

    def _compute_next_scales(self) -> Tuple[Dict[str, float], float, int]:
        active = self._active_branches()
        scales = {branch: 1.0 for branch in self.branches}
        if len(active) <= 1:
            return scales, 0.0, 1
        if self._steps_seen < self.warmup_steps:
            return scales, 0.0, 0

        pressures = [max(0.0, self._ema_pressure.get(branch, 0.0)) for branch in active]
        if sum(pressures) <= 0.0:
            return scales, 0.0, 0

        logs = [math.log(value + 1e-12) / self.temperature for value in pressures]
        max_log = max(logs)
        exp_values = [math.exp(value - max_log) for value in logs]
        denom = sum(exp_values)
        if denom <= 0.0:
            return scales, 0.0, 0

        n = len(active)
        uniform = 1.0 / max(n, 1)
        weights = [value / denom for value in exp_values]
        gated = [
            self.gate_floor * uniform + (1.0 - self.gate_floor) * weight
            for weight in weights
        ]
        entropy = 0.0
        if n > 1:
            entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in gated) / math.log(n)

        for branch, share in zip(active, gated):
            raw_scale = share / uniform
            scales[branch] = min(self.scale_max, max(self.scale_min, float(raw_scale)))
        return scales, float(entropy), 0

    def _validate_expected_coverage(self) -> None:
        if not self._expected_replacement_names:
            raise RuntimeError("Module E requested but no E replacement names were provided.")
        missing = []
        controlled = tuple(self._param_to_branch)
        for replacement in self._expected_replacement_names:
            owned = tuple(name for name in controlled if name.startswith(replacement + "."))
            has_a = any(".lora_a" in name.lower() for name in owned)
            has_b = any(".lora_b" in name.lower() for name in owned)
            if not (has_a and has_b):
                missing.append(replacement)
        if missing:
            raise RuntimeError(
                "Module E structural LoRA coverage is incomplete; every E replacement requires LoRA A/B tensors: "
                + ", ".join(missing[:8])
            )

    def bind_optimizer(self, optimizer: Any) -> None:
        if optimizer is None or not hasattr(optimizer, "param_groups"):
            raise RuntimeError("Module E requires a concrete optimizer with parameter groups.")
        if not self._param_objects or len(self._param_objects) != len(self._param_to_branch):
            raise RuntimeError("Module E cannot bind because controlled parameter objects are incomplete.")

        controlled_ids = {id(param): name for name, param in self._param_objects.items()}
        occurrences = {param_id: 0 for param_id in controlled_ids}
        for group_index, group in enumerate(optimizer.param_groups):
            branches = set()
            non_e_count = 0
            for param in group.get("params", ()):
                param_id = id(param)
                name = controlled_ids.get(param_id)
                if name is None:
                    non_e_count += 1
                    continue
                occurrences[param_id] += 1
                branches.add(self._param_to_branch[name])
            if len(branches) > 1:
                raise RuntimeError(f"Module E optimizer group {group_index} mixes E branches: {sorted(branches)}.")
            if branches and non_e_count:
                raise RuntimeError(f"Module E optimizer group {group_index} mixes E and non-E parameters.")
            expected_tag = f"module_e:{next(iter(branches))}" if branches else "non_module_e"
            if str(group.get("param_group_tag", "")) != expected_tag:
                raise RuntimeError(
                    f"Module E optimizer group {group_index} has tag={group.get('param_group_tag')!r}; "
                    f"expected {expected_tag!r}."
                )

        missing = [controlled_ids[param_id] for param_id, count in occurrences.items() if count == 0]
        duplicated = [controlled_ids[param_id] for param_id, count in occurrences.items() if count != 1 and count > 0]
        if missing or duplicated:
            raise RuntimeError(
                f"Module E controlled parameters must appear exactly once; missing={missing[:5]}, duplicated={duplicated[:5]}."
            )
        for branch in self._active_branches():
            if not self._branch_pressure_param_names.get(branch):
                raise RuntimeError(f"Module E active branch {branch} has no LoRA-B pressure tensors.")
        if self._expected_replacement_names:
            self._validate_expected_coverage()
        self._bound_optimizer = optimizer

    def _read_unscaled_pressure(self) -> None:
        for branch in self.branches:
            self._pending_energy[branch] = 0.0
            self._pending_count[branch] = 0
            self._pending_param_tensors[branch] = 0
        for name, param in self._param_objects.items():
            if self._param_is_pressure.get(name, False):
                self.record_gradient(self._param_to_branch[name], getattr(param, "grad", None), True)

    def prepare_optimizer_step(
        self,
        optimizer: Any,
        global_step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        if optimizer is not self._bound_optimizer:
            raise RuntimeError("Module E optimizer must be bound before preparing a step.")
        if self._prepared_optimizer is not None:
            raise RuntimeError("Module E optimizer step was prepared twice without finishing.")

        stored_group_lrs = [
            (group, float(group["lr"])) for group in optimizer.param_groups
        ]
        pressure_state = {
            "ema": dict(self._ema_pressure),
            "raw": dict(self._raw_pressure),
            "scale": dict(self._branch_scale),
            "pending_energy": dict(self._pending_energy),
            "pending_count": dict(self._pending_count),
            "pending_param_tensors": dict(self._pending_param_tensors),
            "entropy": self._prepared_entropy,
            "degenerate": self._allocation_degenerate,
        }
        try:
            self._read_unscaled_pressure()
            for branch in self.branches:
                count = int(self._pending_count.get(branch, 0))
                raw = float(self._pending_energy.get(branch, 0.0) / count) if count > 0 else 0.0
                prev = float(self._ema_pressure.get(branch, 0.0))
                self._raw_pressure[branch] = raw
                self._ema_pressure[branch] = self.beta * prev + (1.0 - self.beta) * raw
            scales, entropy, degenerate = self._compute_next_scales()
            self._branch_scale.update(scales)

            self._parameter_snapshots = {
                name: param.detach().clone() for name, param in self._param_objects.items()
            }
            self._stored_group_lrs = stored_group_lrs
            self._prepared_optimizer = optimizer
            self._prepared_global_step = global_step
            self._prepared_epoch = epoch
            self._prepared_entropy = entropy
            self._allocation_degenerate = degenerate
            for group, base_lr in stored_group_lrs:
                tag = str(group.get("param_group_tag", ""))
                if tag.startswith("module_e:"):
                    branch = tag.split(":", 1)[1]
                    group["lr"] = base_lr * self.scale_for_branch(branch)
        except BaseException:
            for group, base_lr in stored_group_lrs:
                try:
                    group["lr"] = base_lr
                except BaseException:
                    pass
            self._ema_pressure = pressure_state["ema"]
            self._raw_pressure = pressure_state["raw"]
            self._branch_scale = pressure_state["scale"]
            self._pending_energy = pressure_state["pending_energy"]
            self._pending_count = pressure_state["pending_count"]
            self._pending_param_tensors = pressure_state["pending_param_tensors"]
            self._prepared_entropy = pressure_state["entropy"]
            self._allocation_degenerate = pressure_state["degenerate"]
            self._stored_group_lrs = []
            self._parameter_snapshots = {}
            self._prepared_optimizer = None
            self._prepared_global_step = None
            self._prepared_epoch = None
            raise

    def _diagnostic_path(self) -> str:
        if not self.output_dir:
            return ""
        return os.path.join(self.output_dir, "diagnostics", MODULE_E_DYNAMIC_PRESSURE_FILE)

    def _append_row(self, row: Mapping[str, Any]) -> None:
        path = self._diagnostic_path()
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = list(row.keys())
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not exists:
                writer.writeheader()
            writer.writerow(dict(row))

    def finish_optimizer_step(self, optimizer: Any, step_applied: bool = True) -> Dict[str, Any]:
        if optimizer is not self._prepared_optimizer:
            raise RuntimeError("Module E optimizer step must be prepared before it is finished.")
        actual_by_branch = {branch: 0.0 for branch in self.branches}
        try:
            if step_applied:
                squared = {branch: 0.0 for branch in self.branches}
                for name, before in self._parameter_snapshots.items():
                    param = self._param_objects[name]
                    delta = param.detach() - before.to(device=param.device, dtype=param.dtype)
                    squared[self._param_to_branch[name]] += float(delta.float().pow(2).sum().cpu().item())
                actual_by_branch = {branch: math.sqrt(max(0.0, value)) for branch, value in squared.items()}

            row: Dict[str, Any] = {
                "module_e_metric": "dynamic_structural_pressure_gate",
                "pressure_definition": MODULE_E_DYNAMIC_PRESSURE_DEFINITION,
                "pressure_source": "unscaled_lora_b_gradient_before_optimizer_step",
                "pressure_controls_optimizer_lr": 1,
                "optimizer_control": "temporary_branch_group_lr_multiplier",
                "pressure_param_scope": "lora_b_only",
                "controlled_param_scope": "all_lora_params_in_same_structural_branch",
                "model_name": self.model_name,
                "run_tag": self.run_tag,
                "epoch": "" if self._prepared_epoch is None else int(self._prepared_epoch),
                "global_step": self._steps_seen if self._prepared_global_step is None else int(self._prepared_global_step),
                "controller_step": self._steps_seen,
                "step_applied": int(bool(step_applied)),
                "beta": self.beta,
                "temperature": self.temperature,
                "gate_floor": self.gate_floor,
                "scale_min": self.scale_min,
                "scale_max": self.scale_max,
                "warmup_steps": self.warmup_steps,
                "allocation_entropy": self._prepared_entropy,
                "allocation_degenerate": int(self._allocation_degenerate),
                "registered_lora_param_count": self.total_param_count(),
                "registered_pressure_param_count": self.total_pressure_param_count(),
                "actual_controlled_update_norm": math.sqrt(
                    sum(value * value for value in actual_by_branch.values())
                ),
            }
            for branch in self.branches:
                row[f"raw_pressure_{branch}"] = self._raw_pressure.get(branch, 0.0)
                row[f"ema_pressure_{branch}"] = self._ema_pressure.get(branch, 0.0)
                row[f"optimizer_lr_multiplier_{branch}"] = self._branch_scale.get(branch, 1.0)
                row[f"actual_update_norm_{branch}"] = actual_by_branch.get(branch, 0.0)
                row[f"lora_param_count_{branch}"] = len(self._branch_param_names.get(branch, ()))
                row[f"pressure_param_count_{branch}"] = int(self._pending_param_tensors.get(branch, 0))
            if self._steps_seen % self.diag_freq == 0:
                self._append_row(row)
            return row
        finally:
            for group, base_lr in self._stored_group_lrs:
                group["lr"] = base_lr
            for branch in self.branches:
                self._pending_energy[branch] = 0.0
                self._pending_count[branch] = 0
                self._pending_param_tensors[branch] = 0
            self._stored_group_lrs = []
            self._parameter_snapshots = {}
            self._prepared_optimizer = None
            self._prepared_global_step = None
            self._prepared_epoch = None
            self._steps_seen += 1


def attach_module_e_dynamic_pressure_controller(args: Any, model: Any) -> ModuleEDynamicPressureController:
    """Attach gradient hooks for dynamic Module E pressure-gated LoRA."""
    if args is None or model is None or not hasattr(model, "named_parameters"):
        raise RuntimeError("Module E requested without a model exposing named parameters.")

    existing = getattr(model, "_module_e_dynamic_pressure_controller", None)
    if existing is not None:
        return existing

    controller = ModuleEDynamicPressureController(
        model_name=str(getattr(args, "model_name", "") or ""),
        output_dir=str(getattr(args, "output_dir", "") or ""),
        run_tag=str(getattr(args, "run_tag", "") or ""),
        beta=float(getattr(args, "module_e_pressure_beta", 0.95)),
        temperature=float(getattr(args, "module_e_gate_temperature", 1.0)),
        gate_floor=float(getattr(args, "module_e_gate_floor", 0.2)),
        scale_min=float(getattr(args, "module_e_scale_min", 0.5)),
        scale_max=float(getattr(args, "module_e_scale_max", 1.5)),
        warmup_steps=int(getattr(args, "module_e_warmup_steps", 0)),
        diag_freq=int(getattr(args, "module_e_diag_freq", getattr(args, "diag_freq", 1))),
    )
    controller._expected_replacement_names = _parse_name_list(
        getattr(args, "module_e_injected_names", "")
    )
    if not controller._expected_replacement_names:
        raise RuntimeError("Module E requested but no E replacement names were provided.")

    hook_count = 0
    try:
        for name, param in model.named_parameters():
            if not bool(getattr(param, "requires_grad", False)):
                continue
            if not _is_lora_adapter_param(name):
                continue
            branch = module_e_branch_from_lora_param_name(controller.model_name, name)
            if branch is None:
                continue
            if not controller.register_param(
                name, branch, param, is_pressure_param=_is_lora_b_param_name(name)
            ):
                continue
            handle = param.register_hook(controller.make_gradient_hook(name))
            controller._hook_handles.append(handle)
            hook_count += 1
        controller._validate_expected_coverage()
        for branch in controller._active_branches():
            if not controller._branch_pressure_param_names.get(branch):
                raise RuntimeError(f"Module E active branch {branch} has no LoRA-B pressure tensors.")
    except Exception:
        for handle in controller._hook_handles:
            handle.remove()
        raise

    setattr(model, "_module_e_dynamic_pressure_controller", controller)
    if args is not None:
        setattr(args, "module_e_dynamic_pressure_enabled", True)
        setattr(args, "module_e_dynamic_pressure_file", MODULE_E_DYNAMIC_PRESSURE_FILE)
        setattr(
            args,
            "module_e_dynamic_pressure_branches",
            ";".join(branch for branch in controller.branches if controller.branch_param_count(branch) > 0),
        )
    print(
        f"[ModuleE] dynamic pressure gate hooks registered: lora_tensors={hook_count}, "
        f"pressure_tensors={controller.total_pressure_param_count()}, branches={getattr(args, 'module_e_dynamic_pressure_branches', '')}"
    )
    return controller


def _infer_injected_names_from_model(model_name: str, model: Any) -> Tuple[str, ...]:
    if model is None or not hasattr(model, "named_parameters"):
        return tuple()

    names = []
    seen = set()
    for raw_name, param in model.named_parameters():
        name = str(raw_name or "")
        if not name or not bool(getattr(param, "requires_grad", False)):
            continue
        if not _is_lora_adapter_param(name):
            continue
        prefix = _module_prefix_from_param_name(name)
        if prefix in seen:
            continue
        if _structural_block_for_name(model_name, prefix) is None and _structural_block_for_name(model_name, name) is None:
            continue
        seen.add(prefix)
        names.append(prefix)
    return tuple(names)


def _module_by_name(model: Any, module_name: str) -> Optional[Any]:
    if model is None or not hasattr(model, "named_modules"):
        return None
    for name, module in model.named_modules():
        if name == module_name:
            return module
    return None


def _module_enabled_parts(module: Any) -> str:
    parts = getattr(module, "enable_lora", "")
    if isinstance(parts, (list, tuple, set)):
        return ",".join(str(part) for part in parts)
    return str(parts or "")


def _module_e_matched_rule(model_name: str, module_name: str, branch: str) -> str:
    lower_model = str(model_name or "").lower()
    lower_name = str(module_name or "").lower()
    if lower_model == "cbramod":
        if "self_attn_s" in lower_name:
            return "CBraMod.self_attn_s->spatial"
        if "self_attn_t" in lower_name:
            return "CBraMod.self_attn_t->temporal"
    if lower_model in ("eegpt", "labram") and ".attn." in lower_name:
        return "ViT.attn.qkv/proj->mixing"
    if lower_model == "csbrain":
        if "inter_region_attn" in lower_name:
            return "CSBrain.inter_region_attn->spatial"
        if "inter_window_attn" in lower_name:
            return "CSBrain.inter_window_attn->temporal"
    if lower_model == "biot" and lower_name.endswith(("to_q", "to_k", "to_v", "to_out")):
        return "BIOT.attention_projection->mixing"
    if lower_model == "gram":
        if ".attn." in lower_name:
            return "Gram.attn_projection->mixing"
        if "proj_layers" in lower_name:
            return "Gram.proj_layers->mixing"
    return f"fb_registry->{branch or 'unknown'}"


def _module_e_param_counts(model: Any, module_name: str) -> Tuple[int, int]:
    if model is None or not hasattr(model, "named_parameters"):
        return 0, 0
    lora_count = 0
    pressure_count = 0
    prefix = str(module_name or "")
    dotted = prefix + "."
    for raw_name, _param in model.named_parameters():
        name = str(raw_name or "")
        if not (name == prefix or name.startswith(dotted)):
            continue
        if not _is_lora_adapter_param(name):
            continue
        lora_count += 1
        if _is_lora_b_param_name(name):
            pressure_count += 1
    return lora_count, pressure_count


def save_module_e_lora_injection_audit(
    args: Any,
    model: Any,
    injected_names: Optional[Iterable[Any]] = None,
) -> Optional[str]:
    """Save per-module audit rows for the actual Module E LoRA insertion.

    This file explains where E touched the backbone. It is diagnostic metadata:
    the dynamic pressure controller still decides update strength online.
    """
    if args is None or not bool(getattr(args, "fb_enable", False)):
        return None

    meta = module_e_metadata(args=args)
    if not int(meta.get("module_e_is_active", 0)):
        return None

    output_dir = str(getattr(args, "output_dir", "") or "")
    if not output_dir:
        return None

    model_name = str(getattr(args, "model_name", "") or "")
    parsed_injected = _parse_name_list(injected_names)
    if not parsed_injected:
        parsed_injected = _parse_name_list(getattr(args, "module_e_injected_names", ""))
    if not parsed_injected:
        parsed_injected = _infer_injected_names_from_model(model_name, model)
    if not parsed_injected:
        return None

    controlled_branches = set(_parse_name_list(getattr(args, "module_e_dynamic_pressure_branches", "")))
    rows = []
    for module_name in parsed_injected:
        module_name = str(module_name or "")
        branch = _structural_block_for_name(model_name, module_name) or ""
        if not branch:
            for raw_param_name, _param in getattr(model, "named_parameters", lambda: [])():
                param_name = str(raw_param_name or "")
                if param_name.startswith(module_name + "."):
                    branch = _structural_block_for_name(model_name, param_name) or ""
                    if branch:
                        break
        if not branch:
            continue

        module = _module_by_name(model, module_name)
        lora_count, pressure_count = _module_e_param_counts(model, module_name)
        rows.append(
            {
                "module_e_metric": "e_lora_injection_audit",
                "model_name": model_name,
                "run_tag": str(getattr(args, "run_tag", "") or ""),
                "module_name": module_name,
                "structural_branch": branch,
                "matched_rule": _module_e_matched_rule(model_name, module_name, branch),
                "wrapper_type": type(module).__name__ if module is not None else "",
                "enabled_lora_parts": _module_enabled_parts(module),
                "lora_param_count": lora_count,
                "pressure_param_count": pressure_count,
                "pressure_param_scope": "lora_b_only",
                "controlled_param_scope": "all_lora_params_in_same_structural_branch",
                "pressure_reason": "lora_b_is_zero_initialized_output_side_adapter",
                "dynamic_pressure_controlled": int(
                    bool(getattr(args, "module_e_dynamic_pressure_enabled", False))
                    and branch in controlled_branches
                    and lora_count > 0
                ),
            }
        )

    if not rows:
        return None

    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, MODULE_E_LORA_INJECTION_AUDIT_FILE)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[ModuleE] LoRA injection audit saved to: {path}")
    return path


def save_module_e_coverage_audit(
    args: Any,
    model: Any,
    injected_names: Optional[Iterable[Any]] = None,
    pressure_by_name: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Save Module E structural coverage audit to diagnostics/.

    This is a code-level audit, not a claim that Module E covers all possible
    hidden interaction mechanisms in every EEGFM. ESC is defined over the
    structural inventory recognized by the local registry.
    """
    if args is None or not bool(getattr(args, "fb_enable", False)):
        return None

    meta = module_e_metadata(args=args)
    if not int(meta.get("module_e_is_active", 0)):
        return None

    output_dir = str(getattr(args, "output_dir", "") or "")
    if not output_dir:
        return None

    model_name = str(getattr(args, "model_name", "") or "")
    parsed_injected = _parse_name_list(injected_names)
    if not parsed_injected:
        parsed_injected = _parse_name_list(getattr(args, "module_e_injected_names", ""))
    if not parsed_injected:
        parsed_injected = _infer_injected_names_from_model(model_name, model)

    candidate_names = structural_inventory_from_model(model_name, model)
    audit = structural_coverage_from_names(
        model_name=model_name,
        injected_names=parsed_injected,
        candidate_names=candidate_names,
        pressure_by_name=pressure_by_name,
    )

    row = dict(meta)
    row.update(audit.as_dict())
    row["injected_names"] = ";".join(parsed_injected)

    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, "module_e_coverage_audit.csv")
    keys = list(row.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow(row)
    print(f"[ModuleE] structural coverage audit saved to: {path}")
    return path


# The helpers below are post-hoc diagnostics; they do not control LoRA updates.
def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def structural_pressure_rows_from_block_delta(
    block_delta_rows: Iterable[Mapping[str, Any]],
    model_name: str = "",
    run_tag: str = "",
) -> Tuple[Dict[str, Any], ...]:
    """Compute a landed Module E pressure proxy from block delta diagnostics.

    This is not pre-E gradient/Fisher pressure. It is a post-hoc proxy from the
    already recorded block update energy, so downstream analysis must not treat
    it as pressure-guided target selection evidence.
    """
    grouped: Dict[float, Dict[str, float]] = {}
    for row in block_delta_rows:
        epoch = _safe_float(row.get("epoch"))
        if epoch is None:
            continue
        block = str(row.get("block", "") or "").lower()
        delta = _safe_float(row.get("delta_norm_l2"))
        if delta is None:
            continue
        grouped.setdefault(epoch, {})
        grouped[epoch][block] = grouped[epoch].get(block, 0.0) + float(delta * delta)

    out = []
    for epoch in sorted(grouped):
        energy = grouped[epoch]
        structural_energy = sum(energy.get(block, 0.0) for block in STRUCTURAL_ROUTING_BLOCKS)
        reference_energy = energy.get("input_front", 0.0) + energy.get("semantic", 0.0)
        total_energy = sum(max(0.0, value) for value in energy.values())
        srp_denom = structural_energy + reference_energy
        out.append(
            {
                "module_e_metric": "structural_routing_pressure_proxy",
                "pressure_source": "fb_block_delta_summary.delta_norm_l2_squared",
                "pressure_definition": "posthoc_structural_update_energy_share_not_pre_e_gradient_pressure",
                "pressure_is_pre_e": 0,
                "model_name": str(model_name or ""),
                "run_tag": str(run_tag or ""),
                "epoch": int(epoch) if float(epoch).is_integer() else float(epoch),
                "srp_proxy": float(structural_energy / srp_denom) if srp_denom > 0.0 else 0.0,
                "structural_delta_share_all": float(structural_energy / total_energy) if total_energy > 0.0 else 0.0,
                "structural_energy": float(structural_energy),
                "reference_energy": float(reference_energy),
                "total_energy": float(total_energy),
                "spatial_energy": float(energy.get("spatial", 0.0)),
                "temporal_energy": float(energy.get("temporal", 0.0)),
                "mixing_energy": float(energy.get("mixing", 0.0)),
                "input_front_energy": float(energy.get("input_front", 0.0)),
                "semantic_energy": float(energy.get("semantic", 0.0)),
            }
        )
    return tuple(out)


def save_module_e_structural_pressure_proxy(
    args: Any,
    block_delta_csv_path: Optional[str] = None,
) -> Optional[str]:
    """Save the currently implemented Module E pressure proxy.

    The proxy is deliberately named and documented as post-hoc block-delta
    pressure, not the future pre-E gradient/Fisher pressure probe.
    """
    if args is None or not bool(getattr(args, "fb_enable", False)):
        return None
    meta = module_e_metadata(args=args)
    if not int(meta.get("module_e_is_active", 0)):
        return None
    output_dir = str(getattr(args, "output_dir", "") or "")
    if not output_dir:
        return None

    diag_dir = os.path.join(output_dir, "diagnostics")
    path_in = block_delta_csv_path or os.path.join(diag_dir, "fb_block_delta_summary.csv")
    rows = structural_pressure_rows_from_block_delta(
        _read_csv_rows(path_in),
        model_name=str(getattr(args, "model_name", "") or ""),
        run_tag=str(getattr(args, "run_tag", "") or ""),
    )
    if not rows:
        return None

    os.makedirs(diag_dir, exist_ok=True)
    path_out = os.path.join(diag_dir, MODULE_E_PRESSURE_PROXY_FILE)
    keys = list(rows[0].keys())
    with open(path_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[ModuleE] structural pressure proxy saved to: {path_out}")
    return path_out

# ESC is a registry-defined structural coverage audit.
@dataclass(frozen=True)
class StructuralCoverageAudit:
    """ESC = how much audited structural routing surface Module E covers."""

    model_name: str
    candidate_count: int
    covered_count: int
    esc: float
    pressure_weighted_esc: float
    candidate_names: Tuple[str, ...]
    covered_names: Tuple[str, ...]
    block_counts: Mapping[str, int]
    covered_block_counts: Mapping[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module_e_metric": MODULE_E_METRIC_ESC,
            "model_name": self.model_name,
            "candidate_count": self.candidate_count,
            "covered_count": self.covered_count,
            "esc": self.esc,
            "pressure_weighted_esc": self.pressure_weighted_esc,
            "candidate_names": ";".join(self.candidate_names),
            "covered_names": ";".join(self.covered_names),
            "block_counts": dict(self.block_counts),
            "covered_block_counts": dict(self.covered_block_counts),
        }


def structural_coverage_from_names(
    model_name: str,
    injected_names: Iterable[Any],
    candidate_names: Optional[Iterable[Any]] = None,
    pressure_by_name: Optional[Mapping[str, Any]] = None,
) -> StructuralCoverageAudit:
    """Compute Module E structural coverage from audited module/parameter names.

    ``candidate_names`` should be the model's structural routing inventory.
    When it is omitted, the injected names become the inventory, which is useful
    for smoke tests but not for paper-level coverage claims.
    """
    injected = _as_name_tuple(injected_names)
    candidates = _as_name_tuple(candidate_names) or injected
    pressure_by_name = pressure_by_name or {}

    structural_candidates = []
    block_counts = {block: 0 for block in STRUCTURAL_ROUTING_BLOCKS}
    for name in candidates:
        block = _structural_block_for_name(model_name, name)
        if block is None:
            continue
        structural_candidates.append(name)
        block_counts[block] += 1

    covered = []
    covered_block_counts = {block: 0 for block in STRUCTURAL_ROUTING_BLOCKS}
    for name in structural_candidates:
        if not _name_matches(name, injected):
            continue
        covered.append(name)
        block = _structural_block_for_name(model_name, name)
        if block is not None:
            covered_block_counts[block] += 1

    candidate_count = len(structural_candidates)
    covered_count = len(covered)
    esc = float(covered_count / candidate_count) if candidate_count else 0.0

    if pressure_by_name:
        denom = sum(_pressure_for_name(name, pressure_by_name) for name in structural_candidates)
        numer = sum(_pressure_for_name(name, pressure_by_name) for name in covered)
        pressure_weighted = float(numer / denom) if denom > 0.0 else 0.0
    else:
        pressure_weighted = esc

    return StructuralCoverageAudit(
        model_name=str(model_name or ""),
        candidate_count=candidate_count,
        covered_count=covered_count,
        esc=esc,
        pressure_weighted_esc=pressure_weighted,
        candidate_names=tuple(structural_candidates),
        covered_names=tuple(covered),
        block_counts=block_counts,
        covered_block_counts=covered_block_counts,
    )


@dataclass(frozen=True)
class StructuralRoutingPressure:
    """SRP = positive structural probe gain share among reference blocks."""

    srp: float
    structural_gain: float
    reference_gain: float
    total_gain: float
    gains_by_block: Mapping[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module_e_metric": MODULE_E_METRIC_SRP,
            "srp": self.srp,
            "structural_gain": self.structural_gain,
            "reference_gain": self.reference_gain,
            "total_gain": self.total_gain,
        }


def structural_routing_pressure(
    gains_by_block: Mapping[str, Any],
    structural_blocks: Sequence[str] = STRUCTURAL_ROUTING_BLOCKS,
    reference_blocks: Sequence[str] = ("input_front", "semantic"),
) -> StructuralRoutingPressure:
    structural_gain = sum(_safe_positive(gains_by_block.get(block, 0.0)) for block in structural_blocks)
    reference_gain = sum(_safe_positive(gains_by_block.get(block, 0.0)) for block in reference_blocks)
    total = structural_gain + reference_gain
    srp = float(structural_gain / total) if total > 0.0 else 0.0
    return StructuralRoutingPressure(
        srp=srp,
        structural_gain=float(structural_gain),
        reference_gain=float(reference_gain),
        total_gain=float(total),
        gains_by_block={str(k): _safe_positive(v) for k, v in gains_by_block.items()},
    )


@dataclass(frozen=True)
class StructuralRoutingRelease:
    """SRR = fraction of pre-E structural pressure released after E training."""

    srr: float
    released_gain: float
    before_structural_gain: float
    after_structural_gain: float
    released_by_block: Mapping[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module_e_metric": MODULE_E_METRIC_SRR,
            "srr": self.srr,
            "released_gain": self.released_gain,
            "before_structural_gain": self.before_structural_gain,
            "after_structural_gain": self.after_structural_gain,
        }


def structural_routing_release(
    before_gains_by_block: Mapping[str, Any],
    after_gains_by_block: Mapping[str, Any],
    structural_blocks: Sequence[str] = STRUCTURAL_ROUTING_BLOCKS,
) -> StructuralRoutingRelease:
    released_by_block: Dict[str, float] = {}
    before_total = 0.0
    after_total = 0.0
    released_total = 0.0
    for block in structural_blocks:
        before = _safe_positive(before_gains_by_block.get(block, 0.0))
        after = _safe_positive(after_gains_by_block.get(block, 0.0))
        released = max(0.0, before - after)
        before_total += before
        after_total += after
        released_total += released
        released_by_block[str(block)] = float(released)
    srr = float(released_total / before_total) if before_total > 0.0 else 0.0
    return StructuralRoutingRelease(
        srr=srr,
        released_gain=float(released_total),
        before_structural_gain=float(before_total),
        after_structural_gain=float(after_total),
        released_by_block=released_by_block,
    )


def validation_test_gap_delta(
    base_val_metric: Any,
    base_test_metric: Any,
    adapted_val_metric: Any,
    adapted_test_metric: Any,
) -> float:
    """Positive values mean Module E enlarged the validation-test gap."""
    base_gap = float(base_val_metric) - float(base_test_metric)
    adapted_gap = float(adapted_val_metric) - float(adapted_test_metric)
    return float(adapted_gap - base_gap)


def class_tradeoff_guard(
    reference_recall: Sequence[Any],
    adapted_recall: Sequence[Any],
) -> Dict[str, Any]:
    """Compact guard for recall rescue vs class sacrifice."""
    if len(reference_recall) != len(adapted_recall):
        raise ValueError("reference_recall and adapted_recall must have the same length.")
    deltas = [float(adapted) - float(ref) for ref, adapted in zip(reference_recall, adapted_recall)]
    return {
        "delta_worst_recall": min(deltas) if deltas else 0.0,
        "max_class_gain": max(deltas) if deltas else 0.0,
        "min_class_delta": min(deltas) if deltas else 0.0,
        "mean_class_delta": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        "deltas": tuple(deltas),
    }
