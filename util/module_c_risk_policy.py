"""Parameter-free B/D/E subset selection from class-wise validation effects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


MODULE_C_ACTIONS = ("B", "D", "E")
_EPSILON = 1e-12


@dataclass(frozen=True)
class ValidationRiskDecision:
    selected_modules: Tuple[str, ...]
    per_class_effect: Dict[int, float]
    overall_effect: float
    worst_class_effect: float
    candidate_decisions: Dict[str, Dict[str, Any]]
    search_steps: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    forced_nonempty: bool = False
    reason: str = ""


def _normalize_effects(module_effects: Mapping[Any, Mapping[Any, Any]]) -> Dict[str, Dict[int, float]]:
    normalized: Dict[str, Dict[int, float]] = {}
    class_ids = None
    for raw_module, raw_effects in module_effects.items():
        module_id = str(raw_module or "").strip().upper()
        if module_id not in MODULE_C_ACTIONS:
            raise ValueError(f"Module C accepts only B, D, and E; got {module_id!r}.")
        effects = {int(cls): float(value) for cls, value in raw_effects.items()}
        if not effects:
            raise ValueError(f"Module C received no class-wise validation effects for {module_id}.")
        if class_ids is None:
            class_ids = tuple(sorted(effects))
        elif tuple(sorted(effects)) != class_ids:
            raise ValueError("All Module C candidates must cover the same validation classes.")
        normalized[module_id] = effects
    if len(normalized) == 0:
        raise ValueError("Module C requires at least one B/D/E candidate.")
    if class_ids is None or len(class_ids) < 3:
        raise ValueError("Module C unified-risk selection is defined only for multi-class tasks with at least three classes.")
    return normalized


def _metrics(per_class_effect: Mapping[int, float]) -> Tuple[float, float]:
    values = tuple(float(value) for _, value in sorted(per_class_effect.items()))
    return sum(values) / float(len(values)), min(values)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_gain = float(left["overall_effect"])
    left_worst = float(left["worst_class_effect"])
    right_gain = float(right["overall_effect"])
    right_worst = float(right["worst_class_effect"])
    return (
        left_gain >= right_gain - _EPSILON
        and left_worst >= right_worst - _EPSILON
        and (left_gain > right_gain + _EPSILON or left_worst > right_worst + _EPSILON)
    )


def _is_safe(record: Mapping[str, Any]) -> bool:
    return float(record["overall_effect"]) > _EPSILON and float(record["worst_class_effect"]) >= -_EPSILON


def _safe_rank(record: Mapping[str, Any]) -> Tuple[float, float, int, str]:
    return (
        -float(record["overall_effect"]),
        -float(record["worst_class_effect"]),
        int(record["parameter_count"]),
        str(record["module_id"]),
    )


def _forced_rank(record: Mapping[str, Any]) -> Tuple[float, float, int, str]:
    return (
        -float(record["worst_class_effect"]),
        -float(record["overall_effect"]),
        int(record["parameter_count"]),
        str(record["module_id"]),
    )


def _combined_effect(
    current: Mapping[int, float],
    addition: Mapping[int, float],
) -> Dict[int, float]:
    return {int(cls): float(current[int(cls)]) + float(addition[int(cls)]) for cls in current}


def select_validation_risk_subset(
    module_effects: Mapping[Any, Mapping[Any, Any]],
    parameter_counts: Mapping[Any, Any],
    allow_empty: bool = False,
) -> ValidationRiskDecision:
    """Select a nonempty B/D/E subset from common class-wise loss effects.

    Positive effects lower validation loss. A primary action must improve the
    class-balanced mean without predicting harm to any class. Later actions
    are accepted only when they improve the mean and do not lower the current
    worst-class effect.
    """

    if allow_empty:
        raise ValueError("Module C never permits empty selection; its candidates are only B, D, and E.")

    effects = _normalize_effects(module_effects)
    records: Dict[str, Dict[str, Any]] = {}
    for module_id, by_class in effects.items():
        overall, worst = _metrics(by_class)
        records[module_id] = {
            "module_id": module_id,
            "per_class_effect": dict(by_class),
            "overall_effect": float(overall),
            "worst_class_effect": float(worst),
            "parameter_count": int(parameter_counts.get(module_id, 0)),
            "dominated_by": [],
            "gate": "",
        }

    for module_id, record in records.items():
        record["dominated_by"] = [
            other_id
            for other_id, other in records.items()
            if other_id != module_id and _dominates(other, record)
        ]
        if not _is_safe(record):
            record["gate"] = "unsafe_class_harm" if record["worst_class_effect"] < -_EPSILON else "no_positive_mean_effect"
        elif record["dominated_by"]:
            record["gate"] = "dominated"
        else:
            record["gate"] = "safe_candidate"

    viable = [record for record in records.values() if record["gate"] == "safe_candidate"]
    forced_nonempty = False
    if viable:
        primary = min(viable, key=_safe_rank)
        reason = "selected_safe_validation_effect"
    else:
        primary = min(records.values(), key=_forced_rank)
        forced_nonempty = True
        reason = "forced_nonempty_least_harm"
        primary["gate"] = "forced_nonempty_least_harm"

    selected = [str(primary["module_id"])]
    current_effect = dict(primary["per_class_effect"])
    current_gain, current_worst = _metrics(current_effect)
    steps = [
        {
            "step": 1,
            "selected_modules": list(selected),
            "overall_effect": float(current_gain),
            "worst_class_effect": float(current_worst),
            "forced_nonempty": bool(forced_nonempty),
        }
    ]

    while len(selected) < len(effects):
        additions = []
        for module_id, by_class in effects.items():
            if module_id in selected:
                continue
            combined = _combined_effect(current_effect, by_class)
            combined_gain, combined_worst = _metrics(combined)
            if combined_gain > current_gain + _EPSILON and combined_worst >= current_worst - _EPSILON:
                additions.append(
                    {
                        "module_id": module_id,
                        "per_class_effect": combined,
                        "overall_effect": combined_gain,
                        "worst_class_effect": combined_worst,
                        "parameter_count": sum(int(records[m]["parameter_count"]) for m in (*selected, module_id)),
                    }
                )
        if not additions:
            break
        chosen = min(additions, key=_safe_rank)
        selected.append(str(chosen["module_id"]))
        current_effect = dict(chosen["per_class_effect"])
        current_gain = float(chosen["overall_effect"])
        current_worst = float(chosen["worst_class_effect"])
        steps.append(
            {
                "step": len(selected),
                "selected_modules": list(selected),
                "overall_effect": current_gain,
                "worst_class_effect": current_worst,
                "forced_nonempty": False,
            }
        )

    for module_id, record in records.items():
        record["selected"] = int(module_id in selected)

    return ValidationRiskDecision(
        selected_modules=tuple(selected),
        per_class_effect={int(cls): float(value) for cls, value in current_effect.items()},
        overall_effect=float(current_gain),
        worst_class_effect=float(current_worst),
        candidate_decisions=records,
        search_steps=tuple(steps),
        forced_nonempty=bool(forced_nonempty),
        reason=reason,
    )
