"""Module C B/D/E action registry, validation, and serializable metadata."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


MODULE_C_CURRENT = "exhaustive_low_fidelity_selection"
MODULE_C_ROLE = "nonempty_bde_adapter_action_selector"
MODULE_C_SELECTION_SCOPE = "bde_only_nonempty_subset"
MODULE_C_SELECTION_RULE = "exhaustive_validation_macro_log_loss"


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


def normalize_module_id(module_id: Any) -> str:
    return str(module_id or "").strip().upper()


def parse_module_ids(
    modules: Optional[str | Iterable[Any]],
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[str]:
    """Parse a Module C candidate list and reject anything outside its registry."""

    registry = registry or DEFAULT_CANDIDATE_MODULES
    allowed_order = tuple(normalize_module_id(module_id) for module_id in registry)
    allowed = set(allowed_order)
    if modules is None:
        raw: List[str] = list(allowed_order)
    elif isinstance(modules, str):
        raw = [item.strip() for item in modules.replace(";", ",").replace("|", ",").split(",")]
    else:
        raw = [str(item).strip() for item in modules]

    parsed: List[str] = []
    invalid: List[str] = []
    for item in raw:
        if not item:
            continue
        module_id = normalize_module_id(item)
        if module_id not in allowed:
            invalid.append(module_id)
            continue
        if module_id not in parsed:
            parsed.append(module_id)
    if invalid:
        allowed_text = ",".join(sorted(allowed))
        invalid_text = ",".join(invalid)
        raise ValueError(f"Module C accepts only {allowed_text}; got {invalid_text}.")
    return parsed


def build_module_c_recipe(
    selected_modules: Sequence[Any],
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    candidate_modules: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Build final-training metadata without introducing non-B/D/E controls."""

    registry = registry or DEFAULT_CANDIDATE_MODULES
    selected = tuple(parse_module_ids(selected_modules, registry=registry))
    candidates = (
        parse_module_ids(candidate_modules, registry=registry)
        if candidate_modules is not None
        else parse_module_ids(None, registry=registry)
    )
    if any(module_id not in candidates for module_id in selected):
        raise ValueError("Module C selected modules must be contained in its B/D/E candidates.")

    actions: Dict[str, Dict[str, Any]] = {}
    for module_id in selected:
        meta = dict(registry[module_id])
        blocks = meta.get("blocks", ())
        actions[module_id] = {
            "enabled": True,
            "name": meta.get("name", ""),
            "role": meta.get("role", ""),
            "lora_target": meta.get("lora_target", ""),
            "fb_recipe": meta.get("fb_recipe", ""),
            "blocks": list(blocks) if not isinstance(blocks, str) else [blocks],
        }

    return {
        "module_c_current": MODULE_C_CURRENT,
        "module_c_role": MODULE_C_ROLE,
        "candidate_modules": list(candidates),
        "selected_modules": list(selected),
        "module_actions": actions,
        "selection_scope": MODULE_C_SELECTION_SCOPE,
        "selection_rule": MODULE_C_SELECTION_RULE,
        "requires_nonempty_selection": True,
        "test_used_for_selection": 0,
    }


def module_c_metadata(
    args: Optional[Any] = None,
    selected_modules: Optional[Sequence[Any]] = None,
    registry: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return Module C B/D/E metadata for logs and final-training artifacts."""

    registry = registry or DEFAULT_CANDIDATE_MODULES
    enabled = bool(getattr(args, "module_c_enable", False)) if args is not None else bool(selected_modules)
    raw_candidates = getattr(args, "module_c_candidates", "") if args is not None else None
    candidates = parse_module_ids(raw_candidates, registry=registry) if raw_candidates else parse_module_ids(None, registry=registry)
    selected = tuple(parse_module_ids(selected_modules or (), registry=registry))
    selection_rule = str(getattr(args, "module_c_selection_rule", "") or MODULE_C_SELECTION_RULE) if args is not None else MODULE_C_SELECTION_RULE
    recipe = build_module_c_recipe(selected, registry=registry, candidate_modules=candidates)
    recipe["selection_rule"] = selection_rule
    return {
        "module_c_current": MODULE_C_CURRENT if enabled else "",
        "module_c_role": MODULE_C_ROLE if enabled else "",
        "module_c_is_active": int(enabled),
        "module_c_candidates": ",".join(candidates),
        "module_c_selected_modules": ",".join(selected),
        "module_c_selection_rule": selection_rule,
        "module_c_nonempty_bde_only": int(enabled),
        "module_c_recipe": recipe,
    }
