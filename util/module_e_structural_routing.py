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
MODULE_E_PRESSURE_TARGETS_FILE = "module_e_pressure_targets.csv"
MODULE_E_DYNAMIC_PRESSURE_FILE = "module_e_dynamic_pressure.csv"
MODULE_E_PRE_LORA_PRESSURE_DEFINITION = "pre_lora_gradient_energy_on_structural_candidate_weight"
MODULE_E_DYNAMIC_PRESSURE_DEFINITION = "online_lora_b_gradient_energy_mean_by_structural_branch"
MODULE_E_MODE_DYNAMIC = "dynamic_pressure_gate"
MODULE_E_MODE_STATIC = "static_pressure_topk"
MODULE_E_MODE_LEGACY = "legacy_all_structural"
MODULE_E_MODES = (MODULE_E_MODE_DYNAMIC, MODULE_E_MODE_STATIC, MODULE_E_MODE_LEGACY)


def normalize_lora_target(lora_target: Any) -> str:
    return str(lora_target or "").lower()


def module_e_mode_from_args(args: Any) -> str:
    """Resolve Module E execution mode while keeping old flags as aliases."""
    mode = str(getattr(args, "module_e_mode", "") or "").strip().lower()
    if mode:
        if mode not in MODULE_E_MODES:
            return MODULE_E_MODE_DYNAMIC
        return mode

    legacy_flag = getattr(args, "module_e_pressure_guided", None)
    if legacy_flag is True:
        return MODULE_E_MODE_STATIC
    if legacy_flag is False:
        return MODULE_E_MODE_LEGACY
    return MODULE_E_MODE_DYNAMIC


# 这边是 E 模块的目标接口：正式 E 只认结构/空间/时间注意力，不再接收带 FFN 的复合 target。
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


def select_module_e_pressure_targets(
    model_name: str,
    candidate_names: Iterable[Any],
    pressure_by_name: Mapping[str, Any],
    top_k: Optional[int] = None,
    min_pressure: float = 0.0,
) -> Tuple[str, ...]:
    """Select the structural module prefixes where Module E is allowed to act.

    Pressure is defined on candidate structural parameters. For LoRA insertion
    we aggregate those parameter pressures back to the module prefix that can
    actually be wrapped, then keep the strongest prefixes.
    """
    module_pressure: Dict[str, float] = {}
    module_order: Dict[str, int] = {}
    for idx, raw_name in enumerate(_as_name_tuple(candidate_names)):
        name = str(raw_name or "")
        if _structural_block_for_name(model_name, name) is None:
            continue
        prefix = module_e_module_prefix_from_name(name)
        if not prefix:
            continue
        pressure = _pressure_for_name(name, pressure_by_name)
        module_pressure[prefix] = module_pressure.get(prefix, 0.0) + pressure
        module_order.setdefault(prefix, idx)

    threshold = max(0.0, float(min_pressure or 0.0))
    ranked = [
        (prefix, pressure)
        for prefix, pressure in module_pressure.items()
        if pressure > 0.0 and pressure >= threshold
    ]
    ranked.sort(key=lambda item: (-item[1], module_order.get(item[0], 10**9), item[0]))

    if top_k is not None and int(top_k) > 0:
        ranked = ranked[: int(top_k)]
    return tuple(prefix for prefix, _pressure in ranked)


def load_module_e_pressure_csv(
    path: str,
    name_col: str = "",
    value_col: str = "",
) -> Dict[str, float]:
    """Load an external Module E pressure file.

    Accepted defaults cover the diagnostics written by this module and simple
    user-provided CSV files with name/module/parameter and pressure/score
    columns.
    """
    rows = _read_csv_rows(str(path or ""))
    if not rows:
        return {}

    name_candidates = [
        name_col,
        "candidate_name",
        "param_name",
        "parameter_name",
        "module_name",
        "target_name",
        "name",
    ]
    value_candidates = [
        value_col,
        "pressure",
        "grad_energy",
        "module_pressure",
        "score",
        "value",
    ]

    out: Dict[str, float] = {}
    for row in rows:
        name = ""
        for col in name_candidates:
            if col and row.get(col):
                name = str(row.get(col, "") or "").strip()
                break
        if not name:
            continue

        value = None
        for col in value_candidates:
            if col and row.get(col) not in (None, ""):
                value = _safe_float(row.get(col))
                if value is not None:
                    break
        if value is None:
            continue
        out[name] = out.get(name, 0.0) + _safe_positive(value)
    return out
# 前面全是什么数据输入啊，什么名字清洗啊，没啥重要的
# 下面是旧版的，没删掉主要是有一些模块还有效
def save_module_e_pressure_targets(
    args: Any,
    candidate_names: Iterable[Any],
    pressure_by_name: Mapping[str, Any],
    selected_names: Iterable[Any],
    pressure_source: str,
) -> Optional[str]:
    """Persist the pre-insertion pressure map and selected E action surface."""
    if args is None:
        return None
    output_dir = str(getattr(args, "output_dir", "") or "")
    if not output_dir:
        return None

    model_name = str(getattr(args, "model_name", "") or "")
    selected = set(_as_name_tuple(selected_names))
    rows = []
    for raw_name in _as_name_tuple(candidate_names):
        candidate = str(raw_name or "")
        if _structural_block_for_name(model_name, candidate) is None:
            continue
        target = module_e_module_prefix_from_name(candidate)
        pressure = _pressure_for_name(candidate, pressure_by_name)
        rows.append(
            {
                "module_e_metric": MODULE_E_METRIC_SRP,
                "pressure_definition": MODULE_E_PRE_LORA_PRESSURE_DEFINITION,
                "pressure_source": str(pressure_source or ""),
                "pressure_is_pre_e": 1,
                "model_name": model_name,
                "run_tag": str(getattr(args, "run_tag", "") or ""),
                "candidate_name": candidate,
                "target_name": target,
                "structural_block": _structural_block_for_name(model_name, candidate) or "",
                "pressure": float(pressure),
                "selected_for_lora": int(target in selected),
            }
        )
    rows.sort(key=lambda row: (-float(row["pressure"]), row["target_name"], row["candidate_name"]))
    if not rows:
        return None

    diag_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, MODULE_E_PRESSURE_TARGETS_FILE)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[ModuleE] pressure-guided target selection saved to: {path}")
    return path
# 到此

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
# 这边往上就是抓原始 EEGFM 的结构权重，后面静态旧模式会用到；正式动态 E 主要看 LoRA-B 梯度。

# 这块是 Module E 的核心控制器：算 pressure，再把 pressure 变成 LoRA 梯度倍率。
class ModuleEDynamicPressureController:
    """Online pressure-gated LoRA update controller for Module E.

    Pressure is the mean squared gradient energy on LoRA-B tensors grouped by
    structural branch. The EMA pressure controls the next optimizer update via
    branch-specific gradient multipliers. This keeps insertion broad while
    making Module E's actual update allocation pressure-dependent.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "",
        run_tag: str = "",
        beta: float = 0.95, # 平滑系数
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
        self._hook_handles: List[Any] = []
        self._steps_seen = 0

    def register_param(self, param_name: Any, branch: Any, is_pressure_param: bool = False) -> bool:
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
        if is_pressure_param and name not in self._branch_pressure_param_names[branch]:
            self._branch_pressure_param_names[branch].append(name)
        return True
# 下面主要是记录参数怎么变化的
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
            scale = self.scale_for_branch(branch)
            if scale == 1.0:
                return grad
            return grad * scale

        return _hook

    def _compute_next_scales(self) -> Tuple[Dict[str, float], float]:
        if self._steps_seen < self.warmup_steps:
            return {branch: 1.0 for branch in self.branches}, 0.0

        pressures = [max(0.0, self._ema_pressure.get(branch, 0.0)) for branch in self.branches]
        if sum(pressures) <= 0.0:
            return {branch: 1.0 for branch in self.branches}, 0.0

        logs = [math.log(value + 1e-12) / self.temperature for value in pressures]
        max_log = max(logs)
        exp_values = [math.exp(value - max_log) for value in logs]
        denom = sum(exp_values)
        if denom <= 0.0:
            return {branch: 1.0 for branch in self.branches}, 0.0

        n = len(self.branches)
        uniform = 1.0 / max(n, 1)
        weights = [value / denom for value in exp_values]
        gated = [
            self.gate_floor * uniform + (1.0 - self.gate_floor) * weight
            for weight in weights
        ]
        entropy = 0.0
        if n > 1:
            entropy = -sum(weight * math.log(max(weight, 1e-12)) for weight in gated) / math.log(n)

        scales = {}
        for branch, share in zip(self.branches, gated):
            raw_scale = share / uniform
            scales[branch] = min(self.scale_max, max(self.scale_min, float(raw_scale)))
        return scales, float(entropy)

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

    def finish_step(self, global_step: Optional[int] = None, epoch: Optional[int] = None) -> Dict[str, Any]:
        raw_pressure = {}
        for branch in self.branches:
            count = int(self._pending_count.get(branch, 0))
            raw = float(self._pending_energy.get(branch, 0.0) / count) if count > 0 else 0.0
            raw_pressure[branch] = raw
            prev = float(self._ema_pressure.get(branch, 0.0))
            self._ema_pressure[branch] = self.beta * prev + (1.0 - self.beta) * raw
            self._raw_pressure[branch] = raw

        scales, entropy = self._compute_next_scales()
        self._branch_scale.update(scales)

        row: Dict[str, Any] = {
            "module_e_metric": "dynamic_structural_pressure_gate",
            "pressure_definition": MODULE_E_DYNAMIC_PRESSURE_DEFINITION,
            "pressure_source": "online_lora_gradient_hook",
            "pressure_is_pre_e": 0,
            "pressure_controls_lora_update": 1,
            "model_name": self.model_name,
            "run_tag": self.run_tag,
            "epoch": "" if epoch is None else int(epoch),
            "global_step": self._steps_seen if global_step is None else int(global_step),
            "controller_step": self._steps_seen,
            "beta": self.beta,
            "temperature": self.temperature,
            "gate_floor": self.gate_floor,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "warmup_steps": self.warmup_steps,
            "gate_entropy": entropy,
            "registered_lora_param_count": self.total_param_count(),
            "registered_pressure_param_count": self.total_pressure_param_count(),
        }
        for branch in self.branches:
            row[f"raw_pressure_{branch}"] = raw_pressure.get(branch, 0.0)
            row[f"ema_pressure_{branch}"] = self._ema_pressure.get(branch, 0.0)
            row[f"scale_{branch}"] = self._branch_scale.get(branch, 1.0)
            row[f"lora_param_count_{branch}"] = len(self._branch_param_names.get(branch, ()))
            row[f"pressure_param_count_{branch}"] = int(self._pending_param_tensors.get(branch, 0))

        if self._steps_seen % self.diag_freq == 0:
            self._append_row(row)

        for branch in self.branches:
            self._pending_energy[branch] = 0.0
            self._pending_count[branch] = 0
            self._pending_param_tensors[branch] = 0
        self._steps_seen += 1
        return row


def attach_module_e_dynamic_pressure_controller(args: Any, model: Any) -> Optional[ModuleEDynamicPressureController]:
    """Attach gradient hooks for dynamic Module E pressure-gated LoRA."""
    if args is None or model is None or not hasattr(model, "named_parameters"):
        return None

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

    hook_count = 0
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        if not _is_lora_adapter_param(name):
            continue
        branch = module_e_branch_from_lora_param_name(controller.model_name, name)
        if branch is None:
            continue
        if not controller.register_param(name, branch, is_pressure_param=_is_lora_b_param_name(name)):
            continue
        handle = param.register_hook(controller.make_gradient_hook(name))
        controller._hook_handles.append(handle)
        hook_count += 1

    if controller.total_param_count() <= 0:
        print("[ModuleE][WARN] dynamic pressure gate requested but found no structural LoRA tensors to control.")
        return None

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


# 这块往下主要是训练后的记录和诊断，不直接决定 LoRA 怎么更新。
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

# 往下是怎么定义ESC，也就是Module E structural coverage的，主要是计算有多少结构路由表面被审计过了
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
