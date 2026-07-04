# --------------------------------------------------------
# Module C-v2: Residual-Guided Functional Search (RGFS).
#
# RGFS selects adapter actions by residual-burden coverage. The primary
# residuals are class burdens from validation loss. Optional functional
# residuals let a module expose a non-class failure mode, such as Module E's
# structural routing imbalance, without bypassing the class no-harm gate.
# Complexity is only a tie-breaker.
# --------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _module_id(value: Any) -> str:
    return str(value or "").strip().upper()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _positive(value: Any) -> float:
    return max(0.0, _safe_float(value, 0.0))


def _normalize_burden(class_ids: Sequence[int], burden: Mapping[Any, Any]) -> Dict[int, float]:
    classes = [int(c) for c in class_ids]
    raw = {c: _positive(burden.get(c, burden.get(str(c), 0.0))) for c in classes}
    total = sum(raw.values())
    if total <= 0.0 and classes:
        return {c: 1.0 / float(len(classes)) for c in classes}
    if total <= 0.0:
        return {}
    return {c: raw[c] / total for c in classes}


def _normalize_functional_burden(burden: Optional[Mapping[Any, Any]]) -> Dict[str, float]:
    if not burden:
        return {}
    out: Dict[str, float] = {}
    for key, value in burden.items():
        residual_id = str(key or "").strip()
        amount = _positive(value)
        if residual_id and amount > 0.0:
            out[residual_id] = amount
    return dict(sorted(out.items(), key=lambda item: item[0]))


@dataclass(frozen=True)
class RGFSConfig:
    """Search thresholds for RGFS.

    Defaults are chosen to avoid hidden method constants: any positive residual
    marginal can be accepted, exact ties can use complexity, positive reliable
    harm vetoes a module, and focus classes are those whose burden is at least
    the uniform class burden.
    """

    min_marginal_gain: float = 0.0
    tie_tolerance: float = 0.0
    harm_veto_threshold: float = 0.0
    focus_burden_ratio: float = 1.0
    max_subset_size: Optional[int] = None
    allow_empty: bool = True


@dataclass(frozen=True)
class RGFSDecision:
    selected_modules: Tuple[str, ...]
    selected_score: float
    class_coverage: Dict[int, float]
    class_burden: Dict[int, float]
    functional_coverage: Dict[str, float]
    functional_burden: Dict[str, float]
    focus_classes: Tuple[int, ...]
    candidate_decisions: Dict[str, Dict[str, Any]]
    search_steps: Tuple[Dict[str, Any], ...]
    reason: str
    config: Dict[str, Any] = field(default_factory=dict)


def rgfs_config_dict(config: Optional[RGFSConfig] = None) -> Dict[str, Any]:
    config = config or RGFSConfig()
    return {
        "policy": "residual_guided_functional_search",
        "min_marginal_gain": float(config.min_marginal_gain),
        "tie_tolerance": float(config.tie_tolerance),
        "harm_veto_threshold": float(config.harm_veto_threshold),
        "focus_burden_ratio": float(config.focus_burden_ratio),
        "max_subset_size": config.max_subset_size,
        "allow_empty": bool(config.allow_empty),
        "complexity_role": "tie_break_only",
        "residual_space": "class_plus_optional_functional",
        "focus_rule": "class_burden_at_or_above_uniform_when_ratio_is_1",
        "harm_rule": "positive_reliable_harm_on_focus_class_blocks_candidate_when_threshold_is_0",
    }


def coverage_value(
    class_ids: Sequence[int],
    burden: Mapping[int, float],
    coverage: Mapping[int, float],
) -> float:
    return float(sum(float(burden.get(int(c), 0.0)) * float(coverage.get(int(c), 0.0)) for c in class_ids))


def functional_coverage_value(
    residual_ids: Sequence[str],
    burden: Mapping[str, float],
    coverage: Mapping[str, float],
) -> float:
    return float(sum(float(burden.get(str(r), 0.0)) * float(coverage.get(str(r), 0.0)) for r in residual_ids))


def _focus_classes(class_ids: Sequence[int], burden: Mapping[int, float], config: RGFSConfig) -> Tuple[int, ...]:
    if not class_ids:
        return tuple()
    threshold = max(0.0, float(config.focus_burden_ratio)) / float(len(class_ids))
    focused = [int(c) for c in class_ids if float(burden.get(int(c), 0.0)) >= threshold]
    if focused:
        return tuple(focused)
    best = max((int(c) for c in class_ids), key=lambda c: float(burden.get(int(c), 0.0)))
    return (best,)


def _module_relief(relief_lcb: Mapping[str, Mapping[Any, Any]], module_id: str, cls: int) -> float:
    by_class = relief_lcb.get(module_id, relief_lcb.get(module_id.lower(), {}))
    return _positive(by_class.get(cls, by_class.get(str(cls), 0.0)))


def _module_harm(harm_lcb: Mapping[str, Mapping[Any, Any]], module_id: str, cls: int) -> float:
    by_class = harm_lcb.get(module_id, harm_lcb.get(module_id.lower(), {}))
    return _positive(by_class.get(cls, by_class.get(str(cls), 0.0)))


def _module_functional_relief(
    functional_relief_lcb: Mapping[str, Mapping[Any, Any]],
    module_id: str,
    residual_id: str,
) -> float:
    by_residual = functional_relief_lcb.get(module_id, functional_relief_lcb.get(module_id.lower(), {}))
    return _positive(by_residual.get(residual_id, by_residual.get(str(residual_id), 0.0)))


def _candidate_record(
    module_id: str,
    class_ids: Sequence[int],
    focus_classes: Sequence[int],
    burden: Mapping[int, float],
    coverage: Mapping[int, float],
    functional_ids: Sequence[str],
    functional_burden: Mapping[str, float],
    functional_coverage: Mapping[str, float],
    relief_lcb: Mapping[str, Mapping[Any, Any]],
    harm_lcb: Mapping[str, Mapping[Any, Any]],
    functional_relief_lcb: Mapping[str, Mapping[Any, Any]],
    complexity: Mapping[str, Any],
    config: RGFSConfig,
) -> Dict[str, Any]:
    relief = {int(c): _module_relief(relief_lcb, module_id, int(c)) for c in class_ids}
    harm = {int(c): _module_harm(harm_lcb, module_id, int(c)) for c in class_ids}
    functional_relief = {
        str(r): _module_functional_relief(functional_relief_lcb, module_id, str(r))
        for r in functional_ids
    }
    reliable_harm = [
        int(c)
        for c in focus_classes
        if harm.get(int(c), 0.0) > float(config.harm_veto_threshold)
        and relief.get(int(c), 0.0) <= float(coverage.get(int(c), 0.0))
    ]
    marginal_by_class = {
        int(c): max(0.0, relief.get(int(c), 0.0) - float(coverage.get(int(c), 0.0)))
        for c in class_ids
    }
    functional_marginal_by_residual = {
        str(r): max(0.0, functional_relief.get(str(r), 0.0) - float(functional_coverage.get(str(r), 0.0)))
        for r in functional_ids
    }
    class_marginal = sum(float(burden.get(int(c), 0.0)) * marginal_by_class[int(c)] for c in class_ids)
    focus_class_marginal = sum(float(burden.get(int(c), 0.0)) * marginal_by_class[int(c)] for c in focus_classes)
    functional_marginal = sum(
        float(functional_burden.get(str(r), 0.0)) * functional_marginal_by_residual[str(r)]
        for r in functional_ids
    )
    marginal = class_marginal + functional_marginal
    focus_marginal = focus_class_marginal + functional_marginal
    gate = "pass"
    if reliable_harm:
        gate = "blocked_harm_high_burden"
    elif focus_marginal <= float(config.min_marginal_gain):
        gate = "blocked_no_focus_marginal"
    return {
        "module_id": module_id,
        "gate": gate,
        "marginal_gain": float(marginal),
        "class_marginal_gain": float(class_marginal),
        "functional_marginal_gain": float(functional_marginal),
        "focus_marginal_gain": float(focus_marginal),
        "focus_class_marginal_gain": float(focus_class_marginal),
        "complexity": _positive(complexity.get(module_id, complexity.get(module_id.lower(), 1.0))) or 1.0,
        "relief_lcb_by_class": relief,
        "harm_lcb_by_class": harm,
        "marginal_by_class": marginal_by_class,
        "functional_relief_lcb_by_residual": functional_relief,
        "functional_marginal_by_residual": functional_marginal_by_residual,
        "blocked_harm_classes": reliable_harm,
    }


def _choose_best(records: Sequence[Mapping[str, Any]], config: RGFSConfig) -> Optional[Mapping[str, Any]]:
    passing = [r for r in records if r.get("gate") == "pass"]
    if not passing:
        return None
    passing.sort(key=lambda r: (-float(r.get("focus_marginal_gain", 0.0)), float(r.get("complexity", 1.0)), str(r.get("module_id", ""))))
    best = passing[0]
    for record in passing[1:]:
        if float(best.get("focus_marginal_gain", 0.0)) - float(record.get("focus_marginal_gain", 0.0)) > float(config.tie_tolerance):
            break
        if float(record.get("complexity", 1.0)) < float(best.get("complexity", 1.0)):
            best = record
    return best


def _blocked_harm_score(record: Mapping[str, Any]) -> Tuple[int, float, float]:
    blocked = record.get("blocked_harm_classes", ()) or ()
    harms = record.get("harm_lcb_by_class", {}) or {}
    harm_values = [_positive(v) for v in harms.values()]
    return (
        len(tuple(blocked)),
        max(harm_values, default=0.0),
        sum(harm_values),
    )


def _choose_forced_nonempty(records: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick a transparent fallback when a formal C run disallows empty LoRA."""
    candidates = [dict(r) for r in records if str(r.get("module_id", "")).strip()]
    if not candidates:
        return None
    candidates.sort(
        key=lambda r: (
            0 if float(r.get("focus_marginal_gain", 0.0)) > 0.0 else 1,
            -float(r.get("focus_marginal_gain", 0.0)),
            *_blocked_harm_score(r),
            -float(r.get("marginal_gain", 0.0)),
            float(r.get("complexity", 1.0)),
            str(r.get("module_id", "")),
        )
    )
    chosen = dict(candidates[0])
    original_gate = str(chosen.get("gate", "") or "no_pass")
    chosen["gate"] = f"forced_nonempty_from_{original_gate}"
    chosen["forced_nonempty"] = True
    chosen["forced_nonempty_reason"] = "formal_lora_search_requires_at_least_one_module"
    return chosen


def select_rgfs_subset(
    module_ids: Iterable[Any],
    class_ids: Iterable[Any],
    burden: Mapping[Any, Any],
    relief_lcb: Mapping[str, Mapping[Any, Any]],
    harm_lcb: Optional[Mapping[str, Mapping[Any, Any]]] = None,
    complexity: Optional[Mapping[str, Any]] = None,
    config: Optional[RGFSConfig] = None,
    functional_burden: Optional[Mapping[Any, Any]] = None,
    functional_relief_lcb: Optional[Mapping[str, Mapping[Any, Any]]] = None,
) -> RGFSDecision:
    """Greedily select modules that cover class and optional functional burden."""
    config = config or RGFSConfig()
    modules: List[str] = []
    for raw in module_ids:
        module_id = _module_id(raw)
        if module_id and module_id not in modules:
            modules.append(module_id)
    classes = [int(c) for c in class_ids]
    class_burden = _normalize_burden(classes, burden)
    focus_classes = _focus_classes(classes, class_burden, config)
    harm_lcb = harm_lcb or {}
    complexity = complexity or {}
    functional_burden_norm = _normalize_functional_burden(functional_burden)
    functional_ids = tuple(functional_burden_norm.keys())
    functional_relief_lcb = functional_relief_lcb or {}

    selected: List[str] = []
    coverage = {int(c): 0.0 for c in classes}
    functional_coverage = {str(r): 0.0 for r in functional_ids}
    candidate_decisions: Dict[str, Dict[str, Any]] = {}
    search_steps: List[Dict[str, Any]] = []
    max_size = int(config.max_subset_size) if config.max_subset_size is not None else len(modules)
    max_size = max(0, min(max_size, len(modules)))

    while len(selected) < max_size:
        records = []
        for module_id in modules:
            if module_id in selected:
                continue
            record = _candidate_record(
                module_id=module_id,
                class_ids=classes,
                focus_classes=focus_classes,
                burden=class_burden,
                coverage=coverage,
                functional_ids=functional_ids,
                functional_burden=functional_burden_norm,
                functional_coverage=functional_coverage,
                relief_lcb=relief_lcb,
                harm_lcb=harm_lcb,
                functional_relief_lcb=functional_relief_lcb,
                complexity=complexity,
                config=config,
            )
            records.append(record)
            candidate_decisions[module_id] = dict(record)

        chosen = _choose_best(records, config)
        forced_nonempty = False
        if chosen is None and not selected and not bool(config.allow_empty):
            chosen = _choose_forced_nonempty(records)
            if chosen is not None:
                forced_nonempty = True
                forced_id = str(chosen.get("module_id", ""))
                candidate_decisions[forced_id] = dict(chosen)
                records = [
                    dict(chosen) if str(r.get("module_id", "")) == forced_id else dict(r)
                    for r in records
                ]
        search_steps.append(
            {
                "step": len(selected) + 1,
                "current_selected": list(selected),
                "current_coverage": dict(coverage),
                "current_functional_coverage": dict(functional_coverage),
                "candidates": [dict(r) for r in records],
                "chosen_module": "" if chosen is None else str(chosen.get("module_id", "")),
                "forced_nonempty": bool(forced_nonempty),
            }
        )
        if chosen is None:
            break

        chosen_id = str(chosen["module_id"])
        selected.append(chosen_id)
        for cls in classes:
            coverage[int(cls)] = max(coverage[int(cls)], float(chosen["relief_lcb_by_class"].get(int(cls), 0.0)))
        for residual_id in functional_ids:
            functional_coverage[str(residual_id)] = max(
                functional_coverage[str(residual_id)],
                float(chosen["functional_relief_lcb_by_residual"].get(str(residual_id), 0.0)),
            )

    selected_score = coverage_value(classes, class_burden, coverage) + functional_coverage_value(
        functional_ids,
        functional_burden_norm,
        functional_coverage,
    )
    forced = any(bool(candidate_decisions.get(m, {}).get("forced_nonempty", False)) for m in selected)
    if selected and forced:
        reason = "forced non-empty module because formal LoRA search disallows empty selection"
    elif selected:
        reason = "selected modules by residual burden coverage"
    elif config.allow_empty:
        reason = "no reliable positive marginal relief"
    else:
        reason = "empty selection disallowed"
    return RGFSDecision(
        selected_modules=tuple(selected),
        selected_score=float(selected_score),
        class_coverage={int(k): float(v) for k, v in coverage.items()},
        class_burden={int(k): float(v) for k, v in class_burden.items()},
        functional_coverage={str(k): float(v) for k, v in functional_coverage.items()},
        functional_burden={str(k): float(v) for k, v in functional_burden_norm.items()},
        focus_classes=tuple(int(c) for c in focus_classes),
        candidate_decisions=candidate_decisions,
        search_steps=tuple(search_steps),
        reason=reason,
        config=rgfs_config_dict(config),
    )
