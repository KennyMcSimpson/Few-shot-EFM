# --------------------------------------------------------
# Module C: LoRA search / policy utilities.
#
# Module C is a pre-training adapter-module subset selector. It does not
# inject LoRA modules by itself. Instead, it scores candidate adaptation
# modules such as B/D/E and decides which subset is worth enabling.
# --------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MODULE_C_CURRENT = "lora_module_subset_search"
MODULE_C_ROLE = "adapter_module_subset_selector"


DEFAULT_CANDIDATE_MODULES: Dict[str, Dict[str, Any]] = {
    "B": {
        "name": "signal_alignment",
        "role": "input_front_correction",
        "lora_target": "signal_align",
        "fb_recipe": "sig_align",
        "blocks": ("input_front",),
    },
    "D": {
        "name": "semantic_ffn_adaptation",
        "role": "semantic_ffn_refinement",
        "lora_target": "semantic",
        "fb_recipe": "sem_lif",
        "blocks": ("semantic",),
    },
    "E": {
        "name": "structural_mixing_adaptation",
        "role": "structural_mixing_spatiotemporal_routing",
        "lora_target": "struct_mix",
        "fb_recipe": "str_mix",
        "blocks": ("mixing", "spatial", "temporal"),
    },
}

MODULE_C_EXCLUDED_BASELINES = frozenset(
    (
        "QV",
        "Q_V",
        "QKVO",
        "QV_FFN",
        "QVFFN",
        "QKVO_FFN",
        "QKVOFFN",
        "ATTN_FFN",
        "ATTNFFN",
        "LORA_BASELINE",
        "BASELINE",
        "CONTROL",
    )
)


def normalize_module_id(module_id: Any) -> str:
    return str(module_id or "").strip().upper()


def _canonical_module_token(module_id: Any) -> str:
    token = normalize_module_id(module_id)
    for sep in ("-", " ", "/", ".", ":"):
        token = token.replace(sep, "_")
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")


def is_module_c_baseline_candidate(module_id: Any) -> bool:
    """Return whether a candidate id is a qv-style baseline/control, not C."""
    return _canonical_module_token(module_id) in MODULE_C_EXCLUDED_BASELINES


def parse_module_ids(
    modules: Optional[str | Iterable[Any]],
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    exclude_baselines: bool = True,
) -> List[str]:
    """Parse a comma/semicolon separated module list without hard-coding B/D/E."""
    registry = registry or DEFAULT_CANDIDATE_MODULES
    if modules is None:
        raw: List[str] = list(registry.keys())
    elif isinstance(modules, str):
        raw = [x.strip() for x in modules.replace(";", ",").split(",")]
    else:
        raw = [str(x).strip() for x in modules]

    out: List[str] = []
    for item in raw:
        if not item:
            continue
        module_id = normalize_module_id(item)
        if exclude_baselines and is_module_c_baseline_candidate(module_id):
            continue
        if module_id not in out:
            out.append(module_id)
    return out


def _pair_key(left: Any, right: Any) -> Tuple[str, str]:
    a, b = sorted((normalize_module_id(left), normalize_module_id(right)))
    return a, b


@dataclass(frozen=True)
class ModuleCPolicyConfig:
    """Weights for a conservative train-before-free Module C selector."""

    pressure_weight: float = 0.45
    hard_class_weight: float = 0.20
    stability_weight: float = 0.10
    class_conflict_weight: float = 0.20
    val_test_risk_weight: float = 0.15
    complexity_weight: float = 0.05
    subset_size_weight: float = 0.02
    marginal_margin: float = 0.03
    min_module_score: float = 0.0
    max_subset_size: Optional[int] = None
    allow_empty: bool = False


@dataclass(frozen=True)
class ModuleCScore:
    """
    Precomputed diagnostic values for one candidate module.

    These values are intentionally generic so B/D/E can evolve independently.
    Future modules only need to produce the same fields.
    """

    module_id: str
    pressure: float = 0.0
    low_rank_fit: float = 1.0
    hard_class_leverage: float = 0.0
    class_conflict: float = 0.0
    stability: float = 1.0
    val_test_risk: float = 0.0
    complexity: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_id(self) -> str:
        return normalize_module_id(self.module_id)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _clip01(value: Any, default: float = 0.0) -> float:
    out = _safe_float(value)
    if out is None:
        return float(default)
    return float(max(0.0, min(1.0, out)))


def _positive_delta01(value: Any, scale: float = 0.10) -> float:
    out = _safe_float(value)
    if out is None or scale <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, out / scale)))


def _negative_delta01(value: Any, scale: float = 0.10) -> float:
    out = _safe_float(value)
    if out is None or scale <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, -out / scale)))


def _positive_float(value: Any, default: float = 0.0) -> float:
    out = _safe_float(value)
    if out is None:
        return float(default)
    return float(max(0.0, out))


def _diagnostics_mapping(diagnostics: Any) -> Dict[str, Any]:
    if diagnostics is None:
        return {}
    if hasattr(diagnostics, "as_dict") and callable(getattr(diagnostics, "as_dict")):
        diagnostics = diagnostics.as_dict()
    if not isinstance(diagnostics, Mapping):
        return {}
    return {str(k).strip().lower(): v for k, v in diagnostics.items()}


def _first_number(diagnostics: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = diagnostics.get(key.lower())
        out = _safe_float(value)
        if out is not None:
            return out
    return None


def _max_present(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None]
    return max(finite) if finite else None


def _default_complexity(module_id: str) -> float:
    if module_id == "E":
        return 1.15
    if module_id == "D":
        return 0.95
    return 1.0


def module_c_score_from_diagnostics(
    module_id: Any,
    diagnostics: Any,
    config: Optional[ModuleCPolicyConfig] = None,
) -> ModuleCScore:
    """Translate finalized module diagnostics into the generic C score space.

    The translation is deliberately conservative. Positive validation-only
    reference gains raise pressure/leverage; validation-test gaps, class
    sacrifice, and negative recovery become risks rather than hidden boosts.
    """
    del config
    normalized_id = normalize_module_id(module_id)
    diag = _diagnostics_mapping(diagnostics)

    pressure_raw = _first_number(diag, "pressure", "module_pressure")
    low_rank_raw = _first_number(diag, "low_rank_fit", "lora_fit", "rank_fit")
    hard_raw = _first_number(diag, "hard_class_leverage", "hard_leverage")
    conflict_raw = _first_number(diag, "class_conflict", "class_tradeoff", "recall_conflict")
    stability_raw = _first_number(diag, "stability", "selection_stability")
    risk_raw = _first_number(diag, "val_test_risk", "risk", "overfit_risk")
    complexity_raw = _first_number(diag, "complexity", "adapter_complexity")

    pressure = _clip01(pressure_raw, default=0.0)
    low_rank_fit = _clip01(low_rank_raw, default=1.0)
    hard_class_leverage = _clip01(hard_raw, default=0.0)
    class_conflict = _clip01(conflict_raw, default=0.0)
    stability = _clip01(stability_raw, default=1.0)
    val_test_risk = _clip01(risk_raw, default=0.0)
    complexity = _positive_float(complexity_raw, default=_default_complexity(normalized_id))

    generic_bacc_delta = _first_number(diag, "bacc_delta", "balanced_accuracy_delta", "val_bacc_delta")
    generic_gap = _first_number(diag, "val_test_gap", "validation_test_gap", "overfit_gap")
    hard_class_leverage = max(hard_class_leverage, _positive_delta01(generic_bacc_delta))
    class_conflict = max(class_conflict, _negative_delta01(generic_bacc_delta))
    val_test_risk = max(val_test_risk, _positive_delta01(generic_gap))

    if normalized_id == "B":
        fcr = _first_number(diag, "fcr", "module_b_fcr", "functional_compatibility_retention")
        retention = _first_number(diag, "compatibility_retention", "accessibility_retention")
        retention_for_fit = min([v for v in (fcr, retention) if v is not None], default=None)
        retention_for_stability = retention if retention is not None else fcr
        if retention_for_fit is not None:
            low_rank_fit = min(low_rank_fit, _clip01(retention_for_fit))
            if retention_for_fit < 0.50:
                val_test_risk = max(val_test_risk, _clip01(1.0 - retention_for_fit))
        if pressure_raw is None:
            accessibility_loss = _first_number(diag, "accessibility_loss", "fewshot_accessibility_loss")
            if accessibility_loss is not None:
                pressure = max(pressure, _clip01(accessibility_loss))
            elif retention is not None:
                pressure = max(pressure, _clip01(1.0 - retention))
        recovery = _first_number(diag, "module_b_recovery", "recovery", "recovery_of_lost_retention")
        accessibility_gain = _first_number(diag, "accessibility_gain", "input_front_gain")
        b_delta = _max_present((recovery, accessibility_gain, generic_bacc_delta))
        hard_class_leverage = max(hard_class_leverage, _positive_delta01(b_delta))
        class_conflict = max(class_conflict, _negative_delta01(recovery), _negative_delta01(accessibility_gain), _negative_delta01(generic_bacc_delta))
        if stability_raw is None and retention_for_stability is not None:
            stability = min(stability, _clip01(retention_for_stability))

    elif normalized_id == "D":
        sbr = _first_number(diag, "sbr", "semantic_boundary_refinement_score", "module_d_sbr")
        hard_gain = _first_number(diag, "hard_gain", "hard_class_gain")
        stable_loss = _first_number(diag, "stable_loss", "stable_class_loss")
        if pressure_raw is None:
            pressure = max(
                pressure,
                _positive_delta01(sbr),
                _positive_delta01(hard_gain),
                _positive_delta01(generic_bacc_delta),
            )
        hard_class_leverage = max(
            hard_class_leverage,
            _positive_delta01(sbr),
            _positive_delta01(hard_gain),
            _positive_delta01(generic_bacc_delta),
        )
        class_conflict = max(class_conflict, _positive_delta01(stable_loss), _negative_delta01(generic_bacc_delta))
        if stability_raw is None:
            stability = max(0.0, min(stability, 1.0 - 0.30 * class_conflict - 0.20 * val_test_risk))

    elif normalized_id == "E":
        srp = _first_number(diag, "srp", "structural_routing_pressure")
        srr = _first_number(diag, "srr", "structural_routing_release")
        esc = _first_number(diag, "esc", "e_structural_coverage")
        pressure_weighted_esc = _first_number(diag, "pressure_weighted_esc", "weighted_esc")
        coverage_fit = _max_present((_clip01(esc) if esc is not None else None, _clip01(pressure_weighted_esc) if pressure_weighted_esc is not None else None))
        if pressure_raw is None:
            pressure = max(pressure, _clip01(srp), _clip01(srr), _positive_delta01(generic_bacc_delta))
        if coverage_fit is not None:
            low_rank_fit = coverage_fit if low_rank_raw is None else min(low_rank_fit, coverage_fit)
        delta_worst = _first_number(diag, "delta_worst_recall", "worst_recall_delta", "min_class_delta")
        mean_delta = _first_number(diag, "mean_class_delta")
        hard_class_leverage = max(hard_class_leverage, _positive_delta01(delta_worst), _positive_delta01(mean_delta), _positive_delta01(generic_bacc_delta))
        class_conflict = max(class_conflict, _negative_delta01(delta_worst), _negative_delta01(mean_delta), _negative_delta01(generic_bacc_delta))
        gap_delta = _first_number(diag, "val_test_gap_delta", "validation_test_gap_delta")
        val_test_risk = max(val_test_risk, _positive_delta01(gap_delta))
        if stability_raw is None and coverage_fit is not None:
            stability = max(0.0, min(stability, coverage_fit, 1.0 - 0.20 * val_test_risk))

    return ModuleCScore(
        module_id=normalized_id,
        pressure=float(pressure),
        low_rank_fit=float(low_rank_fit),
        hard_class_leverage=float(hard_class_leverage),
        class_conflict=float(class_conflict),
        stability=float(stability),
        val_test_risk=float(val_test_risk),
        complexity=float(complexity),
        metadata={"diagnostic_keys": tuple(sorted(diag.keys()))},
    )


@dataclass(frozen=True)
class ModuleCDecision:
    selected_modules: Tuple[str, ...]
    selected_score: float
    module_scores: Dict[str, float]
    subset_scores: Dict[Tuple[str, ...], float]
    reason: str
    recipe: Dict[str, Any]
    module_utility_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    subset_score_breakdown: Dict[Tuple[str, ...], Dict[str, Any]] = field(default_factory=dict)


def module_c_policy_config_dict(config: Optional[ModuleCPolicyConfig] = None) -> Dict[str, Any]:
    """Return the fixed conservative priority weights used by Module C."""
    config = config or ModuleCPolicyConfig()
    return {
        "pressure_weight": float(config.pressure_weight),
        "hard_class_weight": float(config.hard_class_weight),
        "train_val_agreement_weight": float(config.stability_weight),
        "class_conflict_weight": float(config.class_conflict_weight),
        "generalization_risk_weight": float(config.val_test_risk_weight),
        "complexity_weight": float(config.complexity_weight),
        "subset_size_weight": float(config.subset_size_weight),
        "marginal_margin": float(config.marginal_margin),
        "min_module_score": float(config.min_module_score),
        "max_subset_size": config.max_subset_size,
        "allow_empty": bool(config.allow_empty),
        "weight_source": "fixed_conservative_priority_not_literature_derived",
    }


def module_utility_breakdown(
    score: ModuleCScore,
    config: Optional[ModuleCPolicyConfig] = None,
) -> Dict[str, float]:
    """Explain how one module utility is assembled without changing selection."""
    config = config or ModuleCPolicyConfig()
    pressure = float(score.pressure)
    low_rank_fit = float(score.low_rank_fit)
    hard_class_leverage = float(score.hard_class_leverage)
    train_val_agreement = float(score.stability)
    class_conflict = max(0.0, float(score.class_conflict))
    generalization_risk = max(0.0, float(score.val_test_risk))
    complexity = max(0.0, float(score.complexity))

    pressure_low_rank_product = pressure * low_rank_fit
    pressure_low_rank_contribution = config.pressure_weight * pressure_low_rank_product
    hard_class_contribution = config.hard_class_weight * hard_class_leverage
    train_val_agreement_contribution = config.stability_weight * train_val_agreement
    class_conflict_penalty = -config.class_conflict_weight * class_conflict
    generalization_risk_penalty = -config.val_test_risk_weight * generalization_risk
    complexity_penalty = -config.complexity_weight * complexity
    utility = (
        pressure_low_rank_contribution
        + hard_class_contribution
        + train_val_agreement_contribution
        + class_conflict_penalty
        + generalization_risk_penalty
        + complexity_penalty
    )
    return {
        "pressure": pressure,
        "low_rank_fit": low_rank_fit,
        "pressure_low_rank_product": pressure_low_rank_product,
        "hard_class_leverage": hard_class_leverage,
        "train_val_agreement": train_val_agreement,
        "class_conflict": class_conflict,
        "generalization_risk": generalization_risk,
        "val_test_risk_legacy_name": generalization_risk,
        "complexity": complexity,
        "pressure_weight": float(config.pressure_weight),
        "hard_class_weight": float(config.hard_class_weight),
        "train_val_agreement_weight": float(config.stability_weight),
        "class_conflict_weight": float(config.class_conflict_weight),
        "generalization_risk_weight": float(config.val_test_risk_weight),
        "complexity_weight": float(config.complexity_weight),
        "pressure_low_rank_contribution": pressure_low_rank_contribution,
        "hard_class_contribution": hard_class_contribution,
        "train_val_agreement_contribution": train_val_agreement_contribution,
        "class_conflict_penalty": class_conflict_penalty,
        "generalization_risk_penalty": generalization_risk_penalty,
        "complexity_penalty": complexity_penalty,
        "utility": float(utility),
    }


def module_utility(score: ModuleCScore, config: Optional[ModuleCPolicyConfig] = None) -> float:
    """Score a single module from pressure, LoRA fit, class leverage, and risks."""
    return float(module_utility_breakdown(score, config)["utility"])


def normalized_interaction_scores(
    interaction_scores: Optional[Mapping[Tuple[Any, Any], float]] = None,
) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for key, value in (interaction_scores or {}).items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Module C interaction key must be a pair, got {key!r}")
        left = normalize_module_id(key[0])
        right = normalize_module_id(key[1])
        if is_module_c_baseline_candidate(left) or is_module_c_baseline_candidate(right):
            continue
        out[_pair_key(left, right)] = float(value)
    return out


def score_subset(
    subset: Sequence[str],
    module_scores: Mapping[str, float],
    interaction_scores: Optional[Mapping[Tuple[Any, Any], float]] = None,
    config: Optional[ModuleCPolicyConfig] = None,
) -> float:
    config = config or ModuleCPolicyConfig()
    normalized_subset = tuple(normalize_module_id(x) for x in subset)
    interactions = normalized_interaction_scores(interaction_scores)

    score = sum(float(module_scores[m]) for m in normalized_subset)
    for left, right in combinations(normalized_subset, 2):
        score += interactions.get(_pair_key(left, right), 0.0)
    if len(normalized_subset) > 1:
        score -= config.subset_size_weight * float(len(normalized_subset) - 1)
    return float(score)


def score_subset_breakdown(
    subset: Sequence[str],
    module_scores: Mapping[str, float],
    interaction_scores: Optional[Mapping[Tuple[Any, Any], float]] = None,
    config: Optional[ModuleCPolicyConfig] = None,
) -> Dict[str, float]:
    """Explain a subset score as utility sum, interaction term, and size penalty."""
    config = config or ModuleCPolicyConfig()
    normalized_subset = tuple(normalize_module_id(x) for x in subset)
    interactions = normalized_interaction_scores(interaction_scores)
    module_utility_sum = sum(float(module_scores[m]) for m in normalized_subset)
    interaction_total = 0.0
    for left, right in combinations(normalized_subset, 2):
        interaction_total += interactions.get(_pair_key(left, right), 0.0)
    subset_size_penalty = (
        config.subset_size_weight * float(len(normalized_subset) - 1)
        if len(normalized_subset) > 1
        else 0.0
    )
    final_score = module_utility_sum + interaction_total - subset_size_penalty
    return {
        "module_utility_sum": float(module_utility_sum),
        "interaction_total": float(interaction_total),
        "subset_size_penalty": float(subset_size_penalty),
        "final_subset_score": float(final_score),
    }


def _candidate_subsets(module_ids: Sequence[str], config: ModuleCPolicyConfig) -> Iterable[Tuple[str, ...]]:
    max_size = config.max_subset_size or len(module_ids)
    min_size = 0 if config.allow_empty else 1
    for size in range(min_size, min(max_size, len(module_ids)) + 1):
        if size == 0:
            yield tuple()
            continue
        for subset in combinations(module_ids, size):
            yield tuple(subset)


def _best_proper_subset_score(
    subset: Tuple[str, ...],
    subset_scores: Mapping[Tuple[str, ...], float],
) -> Optional[float]:
    if len(subset) <= 1:
        return None
    best: Optional[float] = None
    for size in range(1, len(subset)):
        for smaller in combinations(subset, size):
            value = subset_scores.get(tuple(smaller))
            if value is not None and (best is None or value > best):
                best = value
    return best


def select_module_subset(
    module_scores: Iterable[ModuleCScore],
    interaction_scores: Optional[Mapping[Tuple[Any, Any], float]] = None,
    config: Optional[ModuleCPolicyConfig] = None,
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ModuleCDecision:
    """Select a module subset using marginal-gain guarded subset scoring."""
    config = config or ModuleCPolicyConfig()
    registry = registry or DEFAULT_CANDIDATE_MODULES
    interactions = normalized_interaction_scores(interaction_scores)

    raw_scores: Dict[str, ModuleCScore] = {}
    for score in module_scores:
        module_id = score.normalized_id()
        if not module_id or is_module_c_baseline_candidate(module_id):
            continue
        raw_scores[module_id] = score
    utilities = {module_id: module_utility(score, config) for module_id, score in raw_scores.items()}
    utility_breakdown = {
        module_id: module_utility_breakdown(score, config)
        for module_id, score in raw_scores.items()
    }
    candidate_modules = sorted(raw_scores.keys())
    eligible = [m for m, score in utilities.items() if score >= config.min_module_score]
    if not eligible:
        recipe = build_module_c_recipe(tuple(), registry=registry, candidate_modules=candidate_modules, module_scores=utilities)
        return ModuleCDecision(
            selected_modules=tuple(),
            selected_score=0.0,
            module_scores=utilities,
            subset_scores={tuple(): 0.0},
            reason="no module passed min_module_score",
            recipe=recipe,
            module_utility_breakdown=utility_breakdown,
            subset_score_breakdown={tuple(): {"module_utility_sum": 0.0, "interaction_total": 0.0, "subset_size_penalty": 0.0, "final_subset_score": 0.0}},
        )

    subset_scores: Dict[Tuple[str, ...], float] = {}
    for subset in _candidate_subsets(sorted(eligible), config):
        subset_scores[subset] = score_subset(subset, utilities, interactions, config)
    subset_breakdown: Dict[Tuple[str, ...], Dict[str, Any]] = {
        subset: score_subset_breakdown(subset, utilities, interactions, config)
        for subset in subset_scores.keys()
    }

    ranked = sorted(subset_scores.items(), key=lambda item: (item[1], -len(item[0])), reverse=True)
    selected, selected_score = ranked[0]
    rejected_by_margin: List[Tuple[Tuple[str, ...], float, float]] = []
    for subset, value in ranked:
        proper_best = _best_proper_subset_score(subset, subset_scores)
        if subset in subset_breakdown:
            subset_breakdown[subset]["best_proper_subset_score"] = (
                float(proper_best) if proper_best is not None else None
            )
            subset_breakdown[subset]["marginal_gain_over_best_proper_subset"] = (
                float(value - proper_best) if proper_best is not None else None
            )
            subset_breakdown[subset]["accepted_by_margin"] = (
                proper_best is None or value > proper_best + config.marginal_margin
            )
        if proper_best is None or value > proper_best + config.marginal_margin:
            selected, selected_score = subset, value
            break
        rejected_by_margin.append((subset, value, proper_best))
    for subset, value in subset_scores.items():
        if "best_proper_subset_score" in subset_breakdown[subset]:
            continue
        proper_best = _best_proper_subset_score(subset, subset_scores)
        subset_breakdown[subset]["best_proper_subset_score"] = (
            float(proper_best) if proper_best is not None else None
        )
        subset_breakdown[subset]["marginal_gain_over_best_proper_subset"] = (
            float(value - proper_best) if proper_best is not None else None
        )
        subset_breakdown[subset]["accepted_by_margin"] = (
            proper_best is None or value > proper_best + config.marginal_margin
        )

    recipe = build_module_c_recipe(selected, registry=registry, candidate_modules=candidate_modules, module_scores=utilities)
    reason = _selection_reason(selected, selected_score, interactions, rejected_by_margin)
    return ModuleCDecision(
        selected_modules=tuple(selected),
        selected_score=float(selected_score),
        module_scores=utilities,
        subset_scores=subset_scores,
        reason=reason,
        recipe=recipe,
        module_utility_breakdown=utility_breakdown,
        subset_score_breakdown=subset_breakdown,
    )


def select_from_module_diagnostics(
    diagnostics_by_module: Mapping[Any, Any],
    interaction_scores: Optional[Mapping[Tuple[Any, Any], float]] = None,
    config: Optional[ModuleCPolicyConfig] = None,
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ModuleCDecision:
    """Build Module C scores from B/D/E-style diagnostics and select a subset."""
    scores: List[ModuleCScore] = []
    for module_id, diagnostics in (diagnostics_by_module or {}).items():
        normalized_id = normalize_module_id(module_id)
        if not normalized_id or is_module_c_baseline_candidate(normalized_id):
            continue
        scores.append(module_c_score_from_diagnostics(normalized_id, diagnostics, config=config))
    return select_module_subset(
        scores,
        interaction_scores=interaction_scores,
        config=config,
        registry=registry,
    )


def _selection_reason(
    selected: Tuple[str, ...],
    selected_score: float,
    interactions: Mapping[Tuple[str, str], float],
    rejected_by_margin: Sequence[Tuple[Tuple[str, ...], float, float]],
) -> str:
    pieces = [f"selected={'+'.join(selected) if selected else 'none'}", f"score={selected_score:.4f}"]
    if rejected_by_margin:
        pieces.append("marginal guard rejected higher-order subset")
    if any(v < 0.0 for v in interactions.values()):
        pieces.append("negative interaction considered")
    return "; ".join(pieces)


def build_module_c_recipe(
    selected_modules: Sequence[Any],
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    candidate_modules: Optional[Sequence[Any]] = None,
    module_scores: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable recipe. It records module actions, not a qv fallback."""
    registry = registry or DEFAULT_CANDIDATE_MODULES
    selected = tuple(
        normalize_module_id(m)
        for m in selected_modules
        if normalize_module_id(m) and not is_module_c_baseline_candidate(m)
    )
    candidates = parse_module_ids(candidate_modules, registry=registry) if candidate_modules is not None else parse_module_ids(selected)
    actions: Dict[str, Dict[str, Any]] = {}
    for module_id in selected:
        meta = dict(registry.get(module_id, {}))
        blocks = meta.get("blocks", ())
        actions[module_id] = {
            "enabled": True,
            "name": meta.get("name", ""),
            "role": meta.get("role", ""),
            "lora_target": meta.get("lora_target", ""),
            "fb_recipe": meta.get("fb_recipe", ""),
            "blocks": list(blocks) if not isinstance(blocks, str) else [blocks],
        }
    score_summary: Dict[str, float] = {}
    for raw_key, raw_value in (module_scores or {}).items():
        module_id = normalize_module_id(raw_key)
        if not module_id or is_module_c_baseline_candidate(module_id):
            continue
        value = _safe_float(raw_value)
        if value is not None:
            score_summary[module_id] = float(value)
    return {
        "module_c_current": MODULE_C_CURRENT,
        "module_c_role": MODULE_C_ROLE,
        "candidate_modules": list(candidates),
        "selected_modules": list(selected),
        "module_actions": actions,
        "module_score_summary": score_summary,
        "selection_scope": "module_subset_no_qv_baseline",
        "test_used_for_selection": 0,
    }


def module_c_metadata(
    args: Optional[Any] = None,
    selected_modules: Optional[Sequence[Any]] = None,
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return shared Module C metadata for logs, probes, and future runners."""
    registry = registry or DEFAULT_CANDIDATE_MODULES
    enabled = bool(getattr(args, "module_c_enable", False)) if args is not None else bool(selected_modules)
    candidates = getattr(args, "module_c_candidates", "") if args is not None else ""
    selection_rule = (
        str(getattr(args, "module_c_selection_rule", "") or "").strip()
        if args is not None
        else ""
    )
    if not selection_rule:
        selection_rule = "diagnostic_subset_selection_no_test"
    resolved_candidates = parse_module_ids(candidates, registry=registry)
    selected = tuple(parse_module_ids(selected_modules or (), registry=registry))
    recipe = build_module_c_recipe(selected, registry=registry, candidate_modules=resolved_candidates)
    recipe["selection_rule"] = selection_rule
    return {
        "module_c_current": MODULE_C_CURRENT if enabled else "",
        "module_c_role": MODULE_C_ROLE if enabled else "",
        "module_c_is_active": int(enabled),
        "module_c_candidates": ",".join(resolved_candidates),
        "module_c_selected_modules": ",".join(selected),
        "module_c_selection_rule": selection_rule,
        "module_c_no_qv_baseline": 1,
        "module_c_recipe": recipe,
    }
