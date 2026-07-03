# --------------------------------------------------------
# Module C: adapter action registry and metadata.
#
# The active selector is Module C-v2 RGFS, implemented in
# util.module_c_rgfs_policy and called by module_c_preflight_policy. This file
# intentionally no longer contains the old weighted subset selector; it keeps
# only shared parsing/recipe utilities used by lora.py, fb_policy.py, and logs.
# --------------------------------------------------------

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


MODULE_C_CURRENT = "rgfs_residual_guided_functional_search"
MODULE_C_ROLE = "zero_update_adapter_action_selector"
MODULE_C_SELECTION_SCOPE = "functional_action_subset_no_qv_baseline"


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
        raw = [x.strip() for x in modules.replace(";", ",").replace("|", ",").split(",")]
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


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def build_module_c_recipe(
    selected_modules: Sequence[Any],
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    candidate_modules: Optional[Sequence[Any]] = None,
    module_scores: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable recipe for the final training run."""
    registry = registry or DEFAULT_CANDIDATE_MODULES
    selected = tuple(
        normalize_module_id(m)
        for m in selected_modules
        if normalize_module_id(m) and not is_module_c_baseline_candidate(m)
    )
    if candidate_modules is None:
        candidates = parse_module_ids(registry.keys(), registry=registry)
    else:
        candidates = parse_module_ids(candidate_modules, registry=registry)

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
        "selection_scope": MODULE_C_SELECTION_SCOPE,
        "selection_rule": "rgfs_zero_update_preflight",
        "qv_qvffn_role": "baseline_control_only_not_candidate",
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
        selection_rule = "rgfs_zero_update_preflight"
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
