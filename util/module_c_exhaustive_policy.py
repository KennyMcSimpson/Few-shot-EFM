"""Deterministic exhaustive subset policy for Module C.

The policy consumes already measured branch risks.  It never trains a branch,
prunes a path, or performs a post-selection deletion.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SubsetRisk:
    subset: Tuple[str, ...]
    macro_loss: float
    micro_loss: float
    per_class_loss: Mapping[int, float]
    adapter_parameter_count: int


@dataclass(frozen=True)
class ExhaustiveDecision:
    selected_subset: Tuple[str, ...]
    selection_status: str
    runner_up_subset: Tuple[str, ...]
    selection_gap: float
    observed_gain: float
    ranked_subsets: Tuple[Tuple[str, ...], ...]
    conditional_contributions: Mapping[str, float]
    pair_interactions: Mapping[str, float]
    triple_interaction: Optional[float]
    per_class_gain: Mapping[str, Mapping[int, float]]


def enumerate_action_subsets(
    candidate_order: Sequence[str],
) -> Tuple[Tuple[str, ...], ...]:
    candidates = tuple(str(action) for action in candidate_order)
    if not candidates:
        raise ValueError("Module C requires at least one candidate action.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Module C candidate actions must be unique.")
    return tuple(
        subset
        for size in range(1, len(candidates) + 1)
        for subset in itertools.combinations(candidates, size)
    )


def _subset_label(subset: Sequence[str]) -> str:
    return "+".join(subset) if subset else "EMPTY"


def _validate_risk(key: Tuple[str, ...], risk: SubsetRisk) -> None:
    if tuple(risk.subset) != key:
        raise ValueError(
            f"Module C branch key {key!r} disagrees with its recorded subset {risk.subset!r}."
        )
    if not math.isfinite(float(risk.macro_loss)) or not math.isfinite(float(risk.micro_loss)):
        raise ValueError(f"Module C branch {_subset_label(key)} has a non-finite validation loss.")
    if int(risk.adapter_parameter_count) < 0:
        raise ValueError(f"Module C branch {_subset_label(key)} has a negative parameter count.")
    if not risk.per_class_loss:
        raise ValueError(f"Module C branch {_subset_label(key)} has no per-class validation loss.")
    if any(not math.isfinite(float(value)) for value in risk.per_class_loss.values()):
        raise ValueError(f"Module C branch {_subset_label(key)} has a non-finite class loss.")


def select_exhaustive_subset(
    branches: Mapping[Tuple[str, ...], SubsetRisk],
    candidate_order: Sequence[str],
) -> ExhaustiveDecision:
    candidates = tuple(str(action) for action in candidate_order)
    nonempty = enumerate_action_subsets(candidates)
    expected = ((), *nonempty)
    normalized = {tuple(key): value for key, value in branches.items()}
    missing = [subset for subset in expected if subset not in normalized]
    if missing:
        labels = ", ".join(_subset_label(subset) for subset in missing)
        raise ValueError(f"Module C is missing exhaustive branches: {labels}.")
    unexpected = [subset for subset in normalized if subset not in expected]
    if unexpected:
        labels = ", ".join(_subset_label(subset) for subset in unexpected)
        raise ValueError(f"Module C has unexpected exhaustive branches: {labels}.")

    empty = normalized[()]
    _validate_risk((), empty)
    expected_classes = set(int(class_id) for class_id in empty.per_class_loss)
    order_index = {action: index for index, action in enumerate(candidates)}

    for subset in nonempty:
        risk = normalized[subset]
        _validate_risk(subset, risk)
        observed_classes = set(int(class_id) for class_id in risk.per_class_loss)
        if observed_classes != expected_classes:
            raise ValueError(
                f"Module C branch {_subset_label(subset)} has classes {sorted(observed_classes)}; "
                f"expected {sorted(expected_classes)}."
            )

    def rank_key(subset: Tuple[str, ...]):
        risk = normalized[subset]
        return (
            float(risk.macro_loss),
            len(subset),
            int(risk.adapter_parameter_count),
            tuple(order_index[action] for action in subset),
        )

    ranked = tuple(sorted(nonempty, key=rank_key))
    selected = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else ranked[0]
    selected_risk = normalized[selected]
    observed_gain = float(empty.macro_loss) - float(selected_risk.macro_loss)
    selection_status = (
        "positive_gain" if observed_gain > 0.0 else "forced_nonempty_best_observed"
    )

    gains = {
        subset: float(empty.macro_loss) - float(normalized[subset].macro_loss)
        for subset in nonempty
    }
    per_class_gain: Dict[str, Dict[int, float]] = {}
    for subset in nonempty:
        per_class_gain[_subset_label(subset)] = {
            int(class_id): float(empty.per_class_loss[class_id])
            - float(normalized[subset].per_class_loss[class_id])
            for class_id in sorted(expected_classes)
        }

    conditional_contributions = {
        action: float(normalized[tuple(item for item in selected if item != action)].macro_loss)
        - float(selected_risk.macro_loss)
        for action in selected
    }

    pair_interactions: Dict[str, float] = {}
    for left, right in itertools.combinations(candidates, 2):
        pair = (left, right)
        pair_interactions[_subset_label(pair)] = (
            gains[pair] - gains[(left,)] - gains[(right,)]
        )

    triple_interaction: Optional[float] = None
    if len(candidates) == 3:
        triple = tuple(candidates)
        triple_interaction = gains[triple]
        triple_interaction -= sum(gains[(action,)] for action in candidates)
        triple_interaction -= sum(pair_interactions.values())

    return ExhaustiveDecision(
        selected_subset=selected,
        selection_status=selection_status,
        runner_up_subset=runner_up,
        selection_gap=float(normalized[runner_up].macro_loss)
        - float(selected_risk.macro_loss),
        observed_gain=observed_gain,
        ranked_subsets=ranked,
        conditional_contributions=conditional_contributions,
        pair_interactions=pair_interactions,
        triple_interaction=triple_interaction,
        per_class_gain=per_class_gain,
    )
