"""Paired, class-balanced validation-risk evidence for Module C.

The policy is intentionally model agnostic.  It receives paired validation
log-loss reductions from matched branches and decides whether an action has
enough downstream evidence to enter the B/D/E subset.  Action-specific Module
B, D, and E diagnostics never alter this common score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple

from scipy.stats import t as student_t


MODULE_C_ALPHA = 0.05


@dataclass(frozen=True)
class PairedRiskEvidence:
    """Subject-clustered evidence for one matched branch comparison.

    Every value is in validation log-loss reduction units.  Positive values
    mean that the candidate branch has lower loss than its matched reference.
    """

    subject_class_gain: Dict[str, Dict[int, float]]
    class_gain: Dict[int, float]
    overall_gain: float
    worst_class_gain: float
    cluster_count: int
    class_cluster_counts: Dict[int, int]
    overall_standard_error: float
    class_standard_error: Dict[int, float]
    overall_gain_p_value: float
    class_harm_p_values: Dict[int, float]
    confidence_status: str


@dataclass(frozen=True)
class ActionTrial:
    """One measured reference-subset versus candidate-subset comparison."""

    label: str
    base_subset: Tuple[str, ...]
    candidate_subset: Tuple[str, ...]
    added_actions: Tuple[str, ...]
    parameter_count: int
    evidence: PairedRiskEvidence


@dataclass(frozen=True)
class SearchDecision:
    """Result of one forward, rescue, or floating-deletion search stage."""

    selected_trial: Optional[ActionTrial]
    reason: str
    evidence_strength: str
    trial_diagnostics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def selected_subset(self) -> Tuple[str, ...]:
        if self.selected_trial is None:
            return tuple()
        return tuple(self.selected_trial.candidate_subset)


def _validate_subject_class_windows(
    subject_class_windows: Mapping[Any, Mapping[Any, Sequence[Any]]],
) -> Dict[str, Dict[int, Tuple[float, ...]]]:
    normalized: Dict[str, Dict[int, Tuple[float, ...]]] = {}
    for raw_subject, raw_classes in subject_class_windows.items():
        subject_id = str(raw_subject)
        class_map: Dict[int, Tuple[float, ...]] = {}
        for raw_class, raw_values in raw_classes.items():
            class_id = int(raw_class)
            values = tuple(float(value) for value in raw_values)
            if not values:
                continue
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Module C paired validation losses must be finite.")
            class_map[class_id] = values
        if class_map:
            normalized[subject_id] = class_map
    if not normalized:
        raise ValueError("Module C received no subject-class paired validation losses.")
    class_ids = sorted({class_id for class_map in normalized.values() for class_id in class_map})
    if len(class_ids) < 2:
        raise ValueError("Module C requires paired validation losses for at least two observed classes.")
    return normalized


def _standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(stdev(values) / math.sqrt(float(len(values))))


def _one_sided_p_value(estimate: float, standard_error: float, degrees_of_freedom: int, direction: str) -> float:
    if degrees_of_freedom < 1 or not math.isfinite(standard_error):
        return 1.0
    if standard_error == 0.0:
        if estimate == 0.0:
            return 0.5
        if direction == "greater":
            return 0.0 if estimate > 0.0 else 1.0
        if direction == "less":
            return 0.0 if estimate < 0.0 else 1.0
        raise ValueError(f"Unknown one-sided direction: {direction}")
    statistic = float(estimate / standard_error)
    if direction == "greater":
        return float(student_t.sf(statistic, degrees_of_freedom))
    if direction == "less":
        return float(student_t.cdf(statistic, degrees_of_freedom))
    raise ValueError(f"Unknown one-sided direction: {direction}")


def cluster_jackknife_evidence(
    subject_class_windows: Mapping[Any, Mapping[Any, Sequence[Any]]],
) -> PairedRiskEvidence:
    """Aggregate window gains and estimate uncertainty at the subject level.

    The aggregation order is fixed: windows within ``(subject, class)``, then
    subjects within class, then classes.  Delete-one-subject jackknife
    pseudo-values prevent densely windowed recordings from masquerading as
    independent evidence.
    """

    windows = _validate_subject_class_windows(subject_class_windows)
    subject_class_gain = {
        subject_id: {class_id: float(mean(values)) for class_id, values in class_map.items()}
        for subject_id, class_map in windows.items()
    }
    class_ids = tuple(sorted({class_id for class_map in subject_class_gain.values() for class_id in class_map}))
    class_values = {
        class_id: [
            class_map[class_id]
            for class_map in subject_class_gain.values()
            if class_id in class_map
        ]
        for class_id in class_ids
    }
    class_gain = {class_id: float(mean(values)) for class_id, values in class_values.items()}
    overall_gain = float(mean(class_gain.values()))
    worst_class_gain = float(min(class_gain.values()))

    subjects = tuple(sorted(subject_class_gain))
    overall_pseudo_values = []
    for held_out in subjects:
        leave_one_class_gain: Dict[int, float] = {}
        for class_id in class_ids:
            remaining = [
                class_map[class_id]
                for subject_id, class_map in subject_class_gain.items()
                if subject_id != held_out and class_id in class_map
            ]
            if not remaining:
                leave_one_class_gain = {}
                break
            leave_one_class_gain[class_id] = float(mean(remaining))
        if not leave_one_class_gain:
            overall_pseudo_values = []
            break
        leave_one_overall = float(mean(leave_one_class_gain.values()))
        n_subjects = len(subjects)
        overall_pseudo_values.append(float(n_subjects * overall_gain - (n_subjects - 1) * leave_one_overall))

    overall_standard_error = _standard_error(overall_pseudo_values)
    class_standard_error: Dict[int, float] = {}
    class_harm_p_values: Dict[int, float] = {}
    class_cluster_counts: Dict[int, int] = {}
    for class_id, values in class_values.items():
        count = len(values)
        class_cluster_counts[class_id] = count
        standard_error = _standard_error(values)
        class_standard_error[class_id] = standard_error
        class_harm_p_values[class_id] = _one_sided_p_value(
            class_gain[class_id], standard_error, count - 1, "less"
        )

    confidence_available = (
        len(subjects) >= 3
        and len(overall_pseudo_values) == len(subjects)
        and all(count >= 2 for count in class_cluster_counts.values())
        and math.isfinite(overall_standard_error)
    )
    confidence_status = "cluster_jackknife" if confidence_available else "insufficient_subject_clusters"
    overall_gain_p_value = (
        _one_sided_p_value(overall_gain, overall_standard_error, len(subjects) - 1, "greater")
        if confidence_available
        else 1.0
    )
    if not confidence_available:
        class_harm_p_values = {class_id: 1.0 for class_id in class_ids}

    return PairedRiskEvidence(
        subject_class_gain=subject_class_gain,
        class_gain=class_gain,
        overall_gain=overall_gain,
        worst_class_gain=worst_class_gain,
        cluster_count=len(subjects),
        class_cluster_counts=class_cluster_counts,
        overall_standard_error=overall_standard_error,
        class_standard_error=class_standard_error,
        overall_gain_p_value=overall_gain_p_value,
        class_harm_p_values=class_harm_p_values,
        confidence_status=confidence_status,
    )


def holm_adjust(p_values: Mapping[Hashable, Any]) -> Dict[Hashable, float]:
    """Return Holm-adjusted p-values for one explicitly defined test family."""

    if not p_values:
        return {}
    ordered = sorted(
        ((key, min(1.0, max(0.0, float(value)))) for key, value in p_values.items()),
        key=lambda item: (item[1], str(item[0])),
    )
    adjusted: Dict[Hashable, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        corrected = min(1.0, float((family_size - rank) * value))
        running_max = max(running_max, corrected)
        adjusted[key] = running_max
    return adjusted


def _trial_rank(trial: ActionTrial) -> Tuple[float, float, int, Tuple[str, ...]]:
    return (
        -float(trial.evidence.overall_gain),
        -float(trial.evidence.worst_class_gain),
        int(trial.parameter_count),
        tuple(trial.candidate_subset),
    )


def _minimax_rank(trial: ActionTrial) -> Tuple[float, float, int, Tuple[str, ...]]:
    return (
        -float(trial.evidence.worst_class_gain),
        -float(trial.evidence.overall_gain),
        int(trial.parameter_count),
        tuple(trial.candidate_subset),
    )


def _stage_diagnostics(
    trials: Sequence[ActionTrial],
    alpha: float,
) -> Dict[str, Dict[str, Any]]:
    labels = [trial.label for trial in trials]
    if len(labels) != len(set(labels)):
        raise ValueError("Module C trial labels must be unique within a search stage.")
    gain_adjusted = holm_adjust({trial.label: trial.evidence.overall_gain_p_value for trial in trials})
    harm_adjusted_flat = holm_adjust(
        {
            (trial.label, class_id): p_value
            for trial in trials
            for class_id, p_value in trial.evidence.class_harm_p_values.items()
        }
    )
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for trial in trials:
        adjusted_harm = {
            int(class_id): float(harm_adjusted_flat[(trial.label, class_id)])
            for class_id in trial.evidence.class_harm_p_values
        }
        supported_harm = tuple(sorted(class_id for class_id, value in adjusted_harm.items() if value < alpha))
        supported_gain = (
            trial.evidence.confidence_status == "cluster_jackknife"
            and trial.evidence.overall_gain > 0.0
            and float(gain_adjusted[trial.label]) < alpha
        )
        diagnostics[trial.label] = {
            "base_subset": list(trial.base_subset),
            "candidate_subset": list(trial.candidate_subset),
            "added_actions": list(trial.added_actions),
            "parameter_count": int(trial.parameter_count),
            "overall_gain": float(trial.evidence.overall_gain),
            "worst_class_gain": float(trial.evidence.worst_class_gain),
            "class_gain": dict(trial.evidence.class_gain),
            "confidence_status": trial.evidence.confidence_status,
            "gain_p_value": float(trial.evidence.overall_gain_p_value),
            "gain_p_value_holm": float(gain_adjusted[trial.label]),
            "class_harm_p_value_holm": adjusted_harm,
            "supported_gain": bool(supported_gain),
            "supported_harm_classes": list(supported_harm),
            "gate": "supported_gain_no_supported_class_harm" if supported_gain and not supported_harm else (
                "supported_class_harm" if supported_harm else "gain_not_supported"
            ),
        }
    return diagnostics


def choose_action(
    trials: Sequence[ActionTrial],
    require_nonempty: bool,
    alpha: float = MODULE_C_ALPHA,
) -> SearchDecision:
    """Choose one measured branch without combining independent action scores."""

    trials = tuple(trials)
    if not trials:
        raise ValueError("Module C requires at least one measured trial in a search stage.")
    diagnostics = _stage_diagnostics(trials, alpha=float(alpha))
    safe = [
        trial
        for trial in trials
        if diagnostics[trial.label]["supported_gain"]
        and not diagnostics[trial.label]["supported_harm_classes"]
    ]
    if safe:
        chosen = min(safe, key=_trial_rank)
        reason = "supported_primary_gain" if not chosen.base_subset else "supported_conditional_gain"
        return SearchDecision(chosen, reason, "supported", diagnostics)

    if not require_nonempty:
        return SearchDecision(None, "no_supported_safe_gain", "none", diagnostics)

    no_supported_harm = [
        trial for trial in trials if not diagnostics[trial.label]["supported_harm_classes"]
    ]
    if no_supported_harm:
        chosen = min(no_supported_harm, key=_trial_rank)
        return SearchDecision(
            chosen,
            "nonempty_weak_best_observed_gain",
            "weak",
            diagnostics,
        )

    chosen = min(trials, key=_minimax_rank)
    return SearchDecision(
        chosen,
        "nonempty_mandatory_minimax_harm",
        "mandatory",
        diagnostics,
    )


def choose_floating_deletion(
    trials: Sequence[ActionTrial],
    alpha: float = MODULE_C_ALPHA,
) -> SearchDecision:
    """Prefer a smaller measured subset only when observed risk is nonworse.

    This is a parsimony rule, not an equivalence claim: the smaller subset must
    have a nonnegative paired class-balanced point gain and no supported class
    harm.  No arbitrary noninferiority margin is introduced.
    """

    trials = tuple(trials)
    if not trials:
        return SearchDecision(None, "no_floating_deletion_trial", "none", {})
    diagnostics = _stage_diagnostics(trials, alpha=float(alpha))
    eligible = [
        trial
        for trial in trials
        if trial.evidence.overall_gain >= 0.0
        and not diagnostics[trial.label]["supported_harm_classes"]
    ]
    if not eligible:
        return SearchDecision(None, "floating_delete_rejected", "none", diagnostics)
    chosen = min(eligible, key=_trial_rank)
    return SearchDecision(
        chosen,
        "floating_delete_observed_nonworse",
        "parsimony",
        diagnostics,
    )
