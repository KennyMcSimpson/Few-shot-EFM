"""Declarative Module B/D placement contracts for supported EEG backbones.

The contract records model structure only. It never depends on a dataset,
random seed, validation metric, or test result. Module E remains owned by its
existing structural-routing implementation.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch.nn as nn


ModuleKind = Literal["linear", "conv1d"]
BDAction = Literal["B", "D"]


class BackboneContractError(RuntimeError):
    """Raised when a declared backbone capability cannot be resolved safely."""


@dataclass(frozen=True)
class ModuleSiteSpec:
    path_glob: str
    module_kind: ModuleKind
    required: bool = True
    allow_identity: bool = False


@dataclass(frozen=True)
class BackboneBDContract:
    model_name: str
    canonical_head_paths: Tuple[str, ...]
    raw_input_site: bool
    bridge_sites: Tuple[ModuleSiteSpec, ...]
    semantic_ffn_sites: Tuple[ModuleSiteSpec, ...]


@dataclass(frozen=True)
class ResolvedBDSite:
    action: BDAction
    module_path: str
    module: nn.Module
    module_kind: ModuleKind


def _linear(path_glob: str) -> ModuleSiteSpec:
    return ModuleSiteSpec(path_glob=path_glob, module_kind="linear")


def _bridge(*, required: bool = True, allow_identity: bool = False) -> ModuleSiteSpec:
    return ModuleSiteSpec(
        path_glob="chan_conv",
        module_kind="conv1d",
        required=required,
        allow_identity=allow_identity,
    )


_CONTRACTS = {
    "BIOT": BackboneBDContract(
        model_name="BIOT",
        canonical_head_paths=("task_head",),
        raw_input_site=True,
        bridge_sites=(_bridge(),),
        semantic_ffn_sites=(_linear("*.w1"), _linear("*.w2")),
    ),
    "EEGPT": BackboneBDContract(
        model_name="EEGPT",
        canonical_head_paths=("task_head",),
        raw_input_site=True,
        bridge_sites=(_bridge(),),
        semantic_ffn_sites=(_linear("*.mlp.fc1"), _linear("*.mlp.fc2")),
    ),
    "LaBraM": BackboneBDContract(
        model_name="LaBraM",
        canonical_head_paths=("task_head",),
        raw_input_site=True,
        bridge_sites=(),
        semantic_ffn_sites=(_linear("*.mlp.fc1"), _linear("*.mlp.fc2")),
    ),
    "CBraMod": BackboneBDContract(
        model_name="CBraMod",
        canonical_head_paths=("task_head",),
        raw_input_site=True,
        bridge_sites=(),
        semantic_ffn_sites=(_linear("*.linear1"), _linear("*.linear2")),
    ),
    "CSBrain": BackboneBDContract(
        model_name="CSBrain",
        canonical_head_paths=("task_head",),
        raw_input_site=True,
        bridge_sites=(_bridge(required=False, allow_identity=True),),
        semantic_ffn_sites=(_linear("*.linear1"), _linear("*.linear2")),
    ),
    "Gram": BackboneBDContract(
        model_name="Gram",
        canonical_head_paths=("main_model.model.cls_head",),
        raw_input_site=True,
        bridge_sites=(),
        semantic_ffn_sites=(
            _linear("main_model.model.blocks.*.mlp.fc1"),
            _linear("main_model.model.blocks.*.mlp.fc2"),
        ),
    ),
}

SUPPORTED_BD_BACKBONES = tuple(_CONTRACTS)


def _canonical_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    for known in SUPPORTED_BD_BACKBONES:
        if normalized == known.lower():
            return known
    raise BackboneContractError(
        f"Unsupported B/D backbone {model_name!r}; expected one of {SUPPORTED_BD_BACKBONES}."
    )


def get_backbone_bd_contract(model_name: str) -> BackboneBDContract:
    return _CONTRACTS[_canonical_model_name(model_name)]


def _module_by_path(model: nn.Module, module_path: str):
    module = model
    try:
        for part in module_path.split("."):
            module = getattr(module, part)
    except AttributeError:
        return None
    return module


def _matches_kind(module: nn.Module, module_kind: ModuleKind) -> bool:
    if module_kind == "linear":
        return isinstance(module, nn.Linear)
    if module_kind == "conv1d":
        return isinstance(module, nn.Conv1d)
    raise BackboneContractError(f"Unknown module kind {module_kind!r} in B/D contract.")


def _resolve_specs(
    model: nn.Module,
    contract: BackboneBDContract,
    action: BDAction,
    specs: Tuple[ModuleSiteSpec, ...],
) -> Tuple[ResolvedBDSite, ...]:
    named_modules = tuple((name, module) for name, module in model.named_modules() if name)
    resolved = []
    seen_paths = set()

    for spec in specs:
        path_matches = [
            (name, module)
            for name, module in named_modules
            if fnmatch.fnmatchcase(name, spec.path_glob)
        ]
        valid_matches = []
        invalid_matches = []
        for name, module in path_matches:
            if _matches_kind(module, spec.module_kind):
                valid_matches.append((name, module))
            elif spec.allow_identity and isinstance(module, nn.Identity):
                continue
            else:
                invalid_matches.append((name, type(module).__name__))

        if invalid_matches:
            details = ", ".join(f"{name} ({actual})" for name, actual in invalid_matches)
            raise BackboneContractError(
                f"{contract.model_name} {action} site {spec.path_glob!r} expected "
                f"{spec.module_kind}, found {details}."
            )
        if spec.required and not valid_matches:
            role = "bridge" if action == "B" else "semantic"
            raise BackboneContractError(
                f"{contract.model_name} required {role} {action} site "
                f"{spec.path_glob!r} ({spec.module_kind}) was not found."
            )

        for name, module in valid_matches:
            if name in seen_paths:
                raise BackboneContractError(
                    f"{contract.model_name} {action} site {name!r} matched more than one rule."
                )
            seen_paths.add(name)
            resolved.append(
                ResolvedBDSite(
                    action=action,
                    module_path=name,
                    module=module,
                    module_kind=spec.module_kind,
                )
            )

    return tuple(resolved)


def resolve_backbone_bd_sites(
    model: nn.Module,
    model_name: str,
    action: BDAction,
) -> Tuple[ResolvedBDSite, ...]:
    contract = get_backbone_bd_contract(model_name)
    normalized_action = str(action or "").strip().upper()
    if normalized_action == "B":
        return _resolve_specs(model, contract, "B", contract.bridge_sites)
    if normalized_action == "D":
        return _resolve_specs(model, contract, "D", contract.semantic_ffn_sites)
    raise BackboneContractError(f"B/D contract does not resolve action {action!r}; expected 'B' or 'D'.")


def resolve_canonical_head(model: nn.Module, model_name: str) -> nn.Module:
    contract = get_backbone_bd_contract(model_name)
    for module_path in contract.canonical_head_paths:
        module = _module_by_path(model, module_path)
        if isinstance(module, nn.Module):
            return module
    raise BackboneContractError(
        f"{contract.model_name} canonical head was not found at "
        f"{contract.canonical_head_paths}."
    )


def backbone_bd_contract_hash(model_name: str) -> str:
    contract = get_backbone_bd_contract(model_name)
    payload = {
        "model_name": contract.model_name,
        "canonical_head_paths": contract.canonical_head_paths,
        "raw_input_site": contract.raw_input_site,
        "bridge_sites": [site.__dict__ for site in contract.bridge_sites],
        "semantic_ffn_sites": [site.__dict__ for site in contract.semantic_ffn_sites],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_backbone_bd_contract_audit(args, model: nn.Module):
    """Persist the B/D contract and the sites realized by the common injector."""
    payload = getattr(model, "_backbone_bd_contract_audit", None)
    if not isinstance(payload, dict):
        return None

    output_dir = str(getattr(args, "output_dir", "") or "").strip()
    if not output_dir:
        raise BackboneContractError("B/D contract audit requires args.output_dir.")

    diagnostics_dir = Path(output_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / "backbone_bd_contract.json"
    temporary_path = diagnostics_dir / "backbone_bd_contract.json.tmp"
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return str(path)
