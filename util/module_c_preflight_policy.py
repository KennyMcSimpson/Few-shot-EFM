"""Module C B/D/E selection from first-order validation-loss effects.

The disposable preflight model is never reused for final training. It measures
one full support-set gradient, derives one virtual optimizer update per action,
and evaluates the corresponding class-wise validation-loss effect. This keeps
the selector independent of test labels and avoids training B/D/E combinations
just to choose one.
"""

from __future__ import annotations

import csv
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lora import apply_lora_to_eegfm, freeze_all_parameters
from .module_c_lora_search import DEFAULT_CANDIDATE_MODULES, build_module_c_recipe, parse_module_ids
from .module_c_risk_policy import ValidationRiskDecision, select_validation_risk_subset
from .module_e_structural_routing import (
    STRUCTURAL_ROUTING_BLOCKS,
    module_e_branch_from_lora_param_name,
    module_e_module_prefix_from_name,
    structural_inventory_from_model,
)
from .optim_factory import LayerDecayValueAssigner, create_optimizer


MODULE_C_PREFLIGHT_SELECTION_RULE = "validation_risk_first_order_preflight"
MODULE_C_PREFLIGHT_SCORE_FILE = "module_c_preflight_scores.csv"
MODULE_C_PREFLIGHT_DECISION_FILE = "module_c_preflight_decision.json"
_NUMERICAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ModuleCPreflightResult:
    decision: ValidationRiskDecision
    diagnostics_by_module: Mapping[str, Mapping[str, Any]]
    replaced_modules: Tuple[str, ...]
    score_csv_path: str = ""
    decision_json_path: str = ""


def _is_module_c_execution_target(target: Any) -> bool:
    return str(target or "").lower() in ("module_c", "module_c_auto", "c_auto")


def module_c_preflight_requested(args: Any) -> bool:
    """Return whether Module C must automatically resolve a nonempty B/D/E set."""

    if args is None:
        return False
    if str(getattr(args, "finetune_mod", "") or "").lower() != "lora":
        return False
    if not _is_module_c_execution_target(getattr(args, "lora_target", "")):
        return False
    if not bool(getattr(args, "module_c_preflight", True)):
        return False
    selected = parse_module_ids(getattr(args, "module_c_selected", ""))
    resolved = parse_module_ids(getattr(args, "module_c_resolved_selected", ""))
    return len(selected) == 0 and len(resolved) == 0


def _prepare_batch(args: Any, batch: Sequence[Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = batch[0]
    labels_raw = batch[1]
    if str(getattr(args, "norm_method", "")) == "mv":
        samples = samples.float().to(device, non_blocking=True) * float(getattr(args, "mv_norm_value", 0.01))
    else:
        samples = samples.float().to(device, non_blocking=True)

    labels = labels_raw.to(device, non_blocking=True)
    if str(getattr(args, "task_mod", "")) == "Regression":
        targets = labels.float()
    elif int(getattr(args, "nb_classes", 0)) == 1:
        targets = labels.float().view(-1, 1)
    else:
        targets = labels.int().long()
    return samples, targets, labels


def _classification_labels(labels: torch.Tensor, args: Any) -> Optional[torch.Tensor]:
    if str(getattr(args, "task_mod", "")) != "Classification":
        return None
    if int(getattr(args, "nb_classes", 0)) == 1:
        return labels.detach().view(-1).long().clamp(min=0, max=1)
    return labels.detach().view(-1).long()


def _default_criterion(args: Any, device: torch.device) -> nn.Module:
    del device
    if str(getattr(args, "task_mod", "")) == "Regression":
        return nn.MSELoss()
    if int(getattr(args, "nb_classes", 0)) == 1:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _per_sample_loss(output: torch.Tensor, targets: torch.Tensor, args: Any) -> torch.Tensor:
    if str(getattr(args, "task_mod", "")) == "Regression":
        loss = (output.view_as(targets.float()) - targets.float()).pow(2)
        return loss.view(loss.shape[0], -1).mean(dim=1)
    if int(getattr(args, "nb_classes", 0)) == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            output.view(-1), targets.float().view(-1), reduction="none"
        ).view(-1)
    return torch.nn.functional.cross_entropy(output, targets.long(), reduction="none")


def _forward_output(model: nn.Module, samples: torch.Tensor) -> torch.Tensor:
    output = model(samples)
    return output[0] if isinstance(output, (list, tuple)) else output


def _forward_loss(model: nn.Module, samples: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    return criterion(_forward_output(model, samples), targets)


def _collect_probe_batches(data_loader: Any, max_batches: int) -> List[Sequence[Any]]:
    """Collect the full split by default; positive caps are explicit debug controls."""

    batches: List[Sequence[Any]] = []
    if data_loader is None:
        return batches
    limit = int(max_batches)
    for batch_index, batch in enumerate(data_loader):
        if limit > 0 and batch_index >= limit:
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            batches.append(batch)
    return batches


def _strip_lora_param_suffix(param_name: str) -> str:
    name = str(param_name or "")
    lower = name.lower()
    for marker in (".lora_a.", ".lora_b.", ".lora_a", ".lora_b"):
        index = lower.find(marker)
        if index >= 0:
            return name[:index]
    return name


def _prefix_match(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".") or prefix.startswith(name + ".")


def _replacement_owner(
    replacement: str,
    candidate_modules: Sequence[str],
    structural_prefixes: Sequence[str],
) -> str:
    name = str(replacement or "")
    lower = name.lower()
    candidates = set(candidate_modules)
    if "B" in candidates and (name == "input_side_lora" or lower.endswith("chan_conv") or ".chan_conv" in lower):
        return "B"
    if "E" in candidates and any(prefix and _prefix_match(name, prefix) for prefix in structural_prefixes):
        return "E"
    return "D" if "D" in candidates else ""


def _is_adapter_parameter(name: str) -> bool:
    lower = str(name or "").lower()
    return ".lora_a" in lower or ".lora_b" in lower or lower.endswith("lora_a") or lower.endswith("lora_b")


def _build_adapter_param_to_module(
    model: nn.Module,
    model_name: str,
    candidate_modules: Sequence[str],
    replaced_modules: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, int]]:
    structural_prefixes = tuple(
        sorted(
            {
                module_e_module_prefix_from_name(name)
                for name in structural_inventory_from_model(str(model_name), model)
                if module_e_module_prefix_from_name(name)
            },
            key=len,
            reverse=True,
        )
    )
    replacement_owner = {
        str(replacement): _replacement_owner(str(replacement), candidate_modules, structural_prefixes)
        for replacement in replaced_modules
    }
    replacement_owner = {name: module_id for name, module_id in replacement_owner.items() if module_id}

    param_to_module: Dict[str, str] = {}
    parameter_counts = {module_id: 0 for module_id in candidate_modules}
    for name, param in model.named_parameters():
        if not _is_adapter_parameter(name):
            continue
        module_prefix = _strip_lora_param_suffix(name)
        owner = ""
        best_length = -1
        for replacement, module_id in replacement_owner.items():
            if _prefix_match(module_prefix, replacement) and len(replacement) > best_length:
                owner = module_id
                best_length = len(replacement)
        if not owner:
            continue
        param_to_module[name] = owner
        parameter_counts[owner] = parameter_counts.get(owner, 0) + int(param.numel())
    return param_to_module, parameter_counts


def _configure_adapter_trainability(model: nn.Module, adapter_param_to_module: Mapping[str, str]) -> None:
    freeze_all_parameters(model)
    for name, parameter in model.named_parameters():
        if name in adapter_param_to_module:
            parameter.requires_grad_(True)


def _named_adapter_parameters(model: nn.Module, adapter_param_to_module: Mapping[str, str]) -> Dict[str, nn.Parameter]:
    named = dict(model.named_parameters())
    missing = sorted(set(adapter_param_to_module) - set(named))
    if missing:
        raise RuntimeError(f"Module C lost probe adapter parameters: {missing[:3]}")
    return {name: named[name] for name in adapter_param_to_module}


def _full_support_gradients(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
    criterion: nn.Module,
    named_parameters: Mapping[str, nn.Parameter],
) -> Dict[str, torch.Tensor]:
    """Return the gradient of the mean support loss for every probe adapter."""

    gradients = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in named_parameters.items()}
    total_examples = 0
    original_training = bool(model.training)
    try:
        model.eval()
        for batch in batches:
            samples, targets, _labels = _prepare_batch(args, batch, device)
            batch_size = int(samples.shape[0])
            if batch_size <= 0:
                continue
            loss = _forward_loss(model, samples, targets, criterion)
            if loss.ndim != 0:
                loss = loss.mean()
            if not bool(torch.isfinite(loss.detach()).item()):
                raise RuntimeError("Module C support gradient scan produced a non-finite loss.")
            values = torch.autograd.grad(
                loss * float(batch_size),
                tuple(named_parameters.values()),
                allow_unused=True,
            )
            for name, value in zip(named_parameters, values):
                if value is not None:
                    gradients[name].add_(value.detach().to(device="cpu", dtype=gradients[name].dtype))
            total_examples += batch_size
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_training)

    if total_examples <= 0:
        raise RuntimeError("Module C support gradient scan saw no usable examples.")
    for gradient in gradients.values():
        gradient.div_(float(total_examples))
    return gradients


def _optimizer_layer_decay_callbacks(args: Any, model: nn.Module) -> Tuple[Optional[Callable[[str], int]], Optional[Callable[[int], float]]]:
    """Mirror the layer-wise learning-rate scales used by final fine-tuning."""

    model_name = str(getattr(args, "model_name", "") or "")
    main_model = getattr(model, "main_model", None)
    layer_count: Optional[int] = None
    if model_name == "LaBraM" and main_model is not None and hasattr(main_model, "get_num_layers"):
        layer_count = int(main_model.get_num_layers())
    elif model_name == "CBraMod" and main_model is not None and hasattr(main_model, "encoder"):
        layers = getattr(main_model.encoder, "layers", None)
        if layers is not None:
            layer_count = len(layers)
    if layer_count is None or layer_count < 0:
        return None, None
    layer_decay = float(getattr(args, "layer_decay", 1.0))
    assigner = LayerDecayValueAssigner([layer_decay ** (layer_count + 1 - index) for index in range(layer_count + 2)])
    return assigner.get_layer_id, assigner.get_scale


def _optimizer_step_from_gradients(
    args: Any,
    model: nn.Module,
    named_parameters: Mapping[str, nn.Parameter],
    adapter_param_to_module: Mapping[str, str],
    gradients: Mapping[str, torch.Tensor],
    selected_modules: Sequence[str],
    restore_parameters: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Apply one configured optimizer step from cached support gradients.

    No additional support forward/backward pass is needed. ``restore_parameters``
    is used for isolated virtual updates; the confirmation update keeps its
    parameters so validation can inspect the resulting subset.
    """

    selected = set(selected_modules)
    if not selected:
        raise ValueError("Module C optimizer step requires at least one B/D/E module.")
    original_requires_grad = {name: bool(parameter.requires_grad) for name, parameter in named_parameters.items()}
    before: Dict[str, torch.Tensor] = {}
    try:
        for name, parameter in named_parameters.items():
            active = adapter_param_to_module[name] in selected
            parameter.requires_grad_(active)
            if active:
                before[name] = parameter.detach().clone()
                gradient = gradients.get(name)
                parameter.grad = None if gradient is None else gradient.to(parameter.device, dtype=parameter.dtype).clone()
            else:
                parameter.grad = None

        get_num_layer, get_layer_scale = _optimizer_layer_decay_callbacks(args, model)
        optimizer = create_optimizer(
            args,
            model,
            skip_list=[],
            get_num_layer=get_num_layer,
            get_layer_scale=get_layer_scale,
        )
        for group in optimizer.param_groups:
            group["lr"] = float(getattr(args, "lr", 0.0)) * float(group.get("lr_scale", 1.0))
        active_parameters = [parameter for name, parameter in named_parameters.items() if adapter_param_to_module[name] in selected]
        clip_grad = getattr(args, "clip_grad", None)
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(active_parameters, float(clip_grad))
        optimizer.step()

        updates: Dict[str, Dict[str, torch.Tensor]] = {module_id: {} for module_id in selected}
        for name, parameter in named_parameters.items():
            module_id = adapter_param_to_module[name]
            if module_id in selected:
                updates[module_id][name] = (parameter.detach() - before[name]).to(device="cpu").clone()
        return updates
    finally:
        if restore_parameters:
            with torch.no_grad():
                for name, value in before.items():
                    named_parameters[name].copy_(value)
        for name, parameter in named_parameters.items():
            parameter.grad = None
            parameter.requires_grad_(original_requires_grad[name])
        model.zero_grad(set_to_none=True)


def _class_grouped_validation_gradients(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
    named_parameters: Mapping[str, nn.Parameter],
    adapter_param_to_module: Mapping[str, str],
    candidate_modules: Sequence[str],
) -> Tuple[Dict[str, Dict[int, Dict[str, torch.Tensor]]], Dict[int, int]]:
    """Collect one validation forward per batch and one backward per present class."""

    gradients: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {module_id: {} for module_id in candidate_modules}
    counts: Dict[int, int] = {}
    original_training = bool(model.training)
    try:
        model.eval()
        for batch in batches:
            samples, targets, labels_raw = _prepare_batch(args, batch, device)
            labels = _classification_labels(labels_raw, args)
            if labels is None:
                raise ValueError("Module C validation-risk selection requires multi-class classification labels.")
            output = _forward_output(model, samples)
            losses = _per_sample_loss(output, targets, args)
            class_ids = [int(value) for value in torch.unique(labels).detach().cpu().tolist()]
            for class_index, class_id in enumerate(class_ids):
                mask = labels == class_id
                class_count = int(mask.sum().item())
                if class_count <= 0:
                    continue
                class_loss_sum = losses[mask].sum()
                values = torch.autograd.grad(
                    class_loss_sum,
                    tuple(named_parameters.values()),
                    retain_graph=class_index < len(class_ids) - 1,
                    allow_unused=True,
                )
                counts[class_id] = counts.get(class_id, 0) + class_count
                for name, value in zip(named_parameters, values):
                    if value is None:
                        continue
                    module_id = adapter_param_to_module[name]
                    class_map = gradients[module_id].setdefault(class_id, {})
                    detached = value.detach().to(device="cpu")
                    if name in class_map:
                        class_map[name].add_(detached)
                    else:
                        class_map[name] = detached.clone()
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_training)

    class_ids = tuple(sorted(counts))
    if len(class_ids) < 3:
        raise ValueError("Module C validation-risk selection is defined only for classification tasks with at least three validation classes.")
    for module_id in candidate_modules:
        for class_id in class_ids:
            vectors = gradients[module_id].setdefault(class_id, {})
            for name, parameter in named_parameters.items():
                if adapter_param_to_module[name] != module_id:
                    continue
                if name not in vectors:
                    vectors[name] = torch.zeros_like(parameter, device="cpu")
                vectors[name].div_(float(counts[class_id]))
    return gradients, counts


def first_order_effects_from_snapshots(
    validation_gradients: Mapping[Any, Mapping[Any, Mapping[str, torch.Tensor]]],
    virtual_updates: Mapping[Any, Mapping[str, torch.Tensor]],
) -> Dict[str, Dict[int, float]]:
    """Estimate class-wise validation-loss reduction ``-<g_val, delta>``.

    The result is positive when the virtual support update is predicted to
    lower that class's validation loss. It uses the same adapter tensors and
    optimizer-produced parameter displacement as final LoRA training.
    """

    effects: Dict[str, Dict[int, float]] = {}
    for raw_module, by_class in validation_gradients.items():
        module_id = str(raw_module).upper()
        updates = virtual_updates.get(raw_module, virtual_updates.get(module_id, {}))
        effects[module_id] = {}
        for raw_class, gradients in by_class.items():
            value = 0.0
            for name, validation_gradient in gradients.items():
                update = updates.get(name)
                if update is None:
                    continue
                gradient_vector = validation_gradient.detach().float().reshape(-1)
                update_vector = update.detach().float().reshape(-1)
                if gradient_vector.numel() != update_vector.numel():
                    raise ValueError(f"Module C gradient/update shape mismatch for {module_id}:{name}.")
                value -= float(torch.dot(gradient_vector, update_vector).item())
            effects[module_id][int(raw_class)] = float(value)
    return effects


def _classwise_validation_loss(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
) -> Dict[int, float]:
    loss_sum: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    original_training = bool(model.training)
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                samples, targets, labels_raw = _prepare_batch(args, batch, device)
                labels = _classification_labels(labels_raw, args)
                if labels is None:
                    raise ValueError("Module C confirmation requires multi-class classification labels.")
                losses = _per_sample_loss(_forward_output(model, samples), targets, args)
                for raw_class in torch.unique(labels).detach().cpu().tolist():
                    class_id = int(raw_class)
                    mask = labels == class_id
                    class_count = int(mask.sum().item())
                    if class_count <= 0:
                        continue
                    loss_sum[class_id] = loss_sum.get(class_id, 0.0) + float(losses[mask].sum().detach().cpu().item())
                    counts[class_id] = counts.get(class_id, 0) + class_count
    finally:
        model.train(original_training)
    return {class_id: float(loss_sum[class_id] / float(counts[class_id])) for class_id in sorted(counts)}


@contextmanager
def _masked_module_adapter(model: nn.Module, adapter_param_to_module: Mapping[str, str], module_id: str) -> Iterator[None]:
    named_parameters = dict(model.named_parameters())
    saved = {
        name: named_parameters[name].detach().clone()
        for name, owner in adapter_param_to_module.items()
        if owner == module_id
    }
    try:
        with torch.no_grad():
            for name in saved:
                named_parameters[name].zero_()
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                named_parameters[name].copy_(value)


def _loss_metrics(per_class_loss: Mapping[int, float]) -> Tuple[float, float]:
    values = [float(value) for _, value in sorted(per_class_loss.items())]
    if not values:
        raise ValueError("Module C confirmation received no validation class losses.")
    return sum(values) / float(len(values)), max(values)


def prune_harmful_confirmation_additions(
    selected_modules: Sequence[str],
    primary_module: str,
    full_per_class_loss: Mapping[int, float],
    masked_per_class_loss: Mapping[str, Mapping[int, float]],
) -> Tuple[Tuple[str, ...], Dict[str, Dict[str, Any]]]:
    """Remove only added branches whose masking is Pareto-nonworse on validation.

    The first selected module is protected to preserve Module C's nonempty
    protocol. A later module is pruned only when masking it leaves both the
    class-balanced mean loss and the worst-class loss no higher, with one
    strictly lower. The numerical tolerance is floating-point housekeeping,
    not a selectable method threshold.
    """

    selected = tuple(str(module_id).upper() for module_id in selected_modules)
    if not selected:
        raise ValueError("Module C confirmation cannot prune an empty selection.")
    if selected[0] != str(primary_module).upper():
        raise ValueError("Module C confirmation primary module must be the first selected module.")
    full_classes = tuple(sorted(int(class_id) for class_id in full_per_class_loss))
    full_mean, full_worst = _loss_metrics(full_per_class_loss)
    diagnostics: Dict[str, Dict[str, Any]] = {
        selected[0]: {
            "checked": False,
            "pruned": False,
            "reason": "protected_primary_nonempty",
        }
    }
    retained = [selected[0]]
    for module_id in selected[1:]:
        masked = masked_per_class_loss.get(module_id)
        if masked is None or tuple(sorted(int(class_id) for class_id in masked)) != full_classes:
            diagnostics[module_id] = {
                "checked": False,
                "pruned": False,
                "reason": "missing_or_incomplete_masked_validation_loss",
            }
            retained.append(module_id)
            continue
        masked_mean, masked_worst = _loss_metrics(masked)
        nonworse = masked_mean <= full_mean + _NUMERICAL_TOLERANCE and masked_worst <= full_worst + _NUMERICAL_TOLERANCE
        strictly_better = masked_mean < full_mean - _NUMERICAL_TOLERANCE or masked_worst < full_worst - _NUMERICAL_TOLERANCE
        pruned = bool(nonworse and strictly_better)
        diagnostics[module_id] = {
            "checked": True,
            "pruned": pruned,
            "full_mean_loss": float(full_mean),
            "full_worst_class_loss": float(full_worst),
            "masked_mean_loss": float(masked_mean),
            "masked_worst_class_loss": float(masked_worst),
            "reason": "masked_branch_pareto_nonworse" if pruned else "masked_branch_needed_or_inconclusive",
        }
        if not pruned:
            retained.append(module_id)
    return tuple(retained), diagnostics


def _combined_effects(module_effects: Mapping[str, Mapping[int, float]], selected_modules: Sequence[str]) -> Dict[int, float]:
    class_ids = tuple(sorted(next(iter(module_effects.values()))))
    return {
        class_id: float(sum(float(module_effects[module_id][class_id]) for module_id in selected_modules))
        for class_id in class_ids
    }


def _vector_norm(vectors: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for vector in vectors.values():
        total += float(vector.detach().float().pow(2).sum().item())
    return float(total ** 0.5)


def _module_e_reference_diagnostic(model: nn.Module, model_name: str) -> Dict[str, Any]:
    branches = set()
    for name in structural_inventory_from_model(str(model_name), model):
        branch = module_e_branch_from_lora_param_name(str(model_name), name)
        if branch in STRUCTURAL_ROUTING_BLOCKS:
            branches.add(branch)
    covered = tuple(sorted(branches))
    return {
        "structural_branches_observed": list(covered),
        "structural_branch_count": len(covered),
        "structural_reference_status": "identified" if len(covered) >= 2 else "not_identifiable",
        "structural_reference_used_for_ranking": 0,
    }


def _build_diagnostics(
    args: Any,
    model: nn.Module,
    candidate_modules: Sequence[str],
    decision: ValidationRiskDecision,
    module_effects: Mapping[str, Mapping[int, float]],
    adapter_param_to_module: Mapping[str, str],
    parameter_counts: Mapping[str, int],
    support_gradients: Mapping[str, torch.Tensor],
    virtual_updates: Mapping[str, Mapping[str, torch.Tensor]],
    confirmation: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for module_id in candidate_modules:
        record = dict(decision.candidate_decisions[module_id])
        module_vectors = {
            name: value
            for name, value in support_gradients.items()
            if adapter_param_to_module.get(name) == module_id
        }
        diagnostics[module_id] = {
            "module_id": module_id,
            "functional_name": DEFAULT_CANDIDATE_MODULES[module_id]["name"],
            "functional_role": DEFAULT_CANDIDATE_MODULES[module_id]["role"],
            "functional_blocks": list(DEFAULT_CANDIDATE_MODULES[module_id]["blocks"]),
            "functional_diagnostics_role": "reference_only_not_used_for_ranking",
            "adapter_parameter_count": int(parameter_counts.get(module_id, 0)),
            "support_gradient_l2": _vector_norm(module_vectors),
            "virtual_update_l2": _vector_norm(virtual_updates.get(module_id, {})),
            "first_order_effect_by_class": {str(class_id): float(value) for class_id, value in module_effects[module_id].items()},
            "mean_first_order_effect": float(record["overall_effect"]),
            "worst_class_first_order_effect": float(record["worst_class_effect"]),
            "primary_gate": record.get("gate", ""),
            "dominated_by": list(record.get("dominated_by", ())),
            "selected_before_confirmation": int(module_id in confirmation["selected_before"]),
            "selected_after_confirmation": int(module_id in confirmation["selected_after"]),
            "confirmation": dict(confirmation["pruning"].get(module_id, {})),
        }
        if module_id == "E":
            diagnostics[module_id].update(_module_e_reference_diagnostic(model, str(getattr(args, "model_name", ""))))
    return diagnostics


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> str:
    if not path or not rows:
        return ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in row.items()
                }
            )
    return path


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    if not path:
        return ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _write_decision_json(
    path: str,
    args: Any,
    decision: ValidationRiskDecision,
    candidate_modules: Sequence[str],
    diagnostics: Mapping[str, Mapping[str, Any]],
    confirmation: Mapping[str, Any],
    validation_class_counts: Mapping[int, int],
    parameter_counts: Mapping[str, int],
    replaced_modules: Sequence[str],
) -> str:
    recipe = build_module_c_recipe(
        decision.selected_modules,
        registry=DEFAULT_CANDIDATE_MODULES,
        candidate_modules=candidate_modules,
        module_scores={module_id: decision.candidate_decisions[module_id]["overall_effect"] for module_id in candidate_modules},
    )
    train_cap = int(getattr(args, "module_c_preflight_train_batches", 0))
    val_cap = int(getattr(args, "module_c_preflight_val_batches", 0))
    payload = {
        "module_c_selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE,
        "test_used_for_selection": 0,
        "model_name": str(getattr(args, "model_name", "") or ""),
        "dataset": str(getattr(args, "dataset", "") or ""),
        "subject_mod": str(getattr(args, "subject_mod", "") or ""),
        "k_shot": getattr(args, "k_shot", ""),
        "seed": getattr(args, "seed", ""),
        "candidate_modules": list(candidate_modules),
        "selected_modules": list(decision.selected_modules),
        "forced_nonempty": bool(decision.forced_nonempty),
        "selection_reason": decision.reason,
        "first_order_effect_definition": "q[a,c] = -sum_p <grad_validation[a,c,p], virtual_update[a,p]>",
        "effect_sign": "positive predicts lower validation loss for that class",
        "primary_rule": "safe non-dominated action, then highest class-balanced mean effect, highest worst-class effect, and fewer adapter parameters",
        "addition_rule": "add only when class-balanced mean effect rises and worst-class effect does not fall",
        "nonempty_rule": "if no safe action exists, select the least-harmful B/D/E action by worst-class effect",
        "confirmation_rule": "after one support optimizer update, prune only added branches whose masking is Pareto-nonworse on class-balanced and worst-class validation loss",
        "numerical_tolerance": _NUMERICAL_TOLERANCE,
        "optimizer_for_virtual_and_confirmation_step": {
            "name": str(getattr(args, "opt", "")),
            "lr": float(getattr(args, "lr", 0.0)),
            "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            "layer_decay": float(getattr(args, "layer_decay", 1.0)),
            "clip_grad": getattr(args, "clip_grad", None),
            "lr_rule": "configured base learning rate times the final model layer scale, before epoch scheduling",
        },
        "probe_scope": {
            "support_batch_cap": train_cap,
            "validation_batch_cap": val_cap,
            "support_split": "full" if train_cap <= 0 else "debug_capped",
            "validation_split": "full" if val_cap <= 0 else "debug_capped",
            "model_mode": "eval",
            "probe_adapter_dropout": 0.0,
            "validation_class_counts": {str(class_id): int(count) for class_id, count in validation_class_counts.items()},
        },
        "parameter_counts": {module_id: int(parameter_counts.get(module_id, 0)) for module_id in candidate_modules},
        "candidate_decisions": decision.candidate_decisions,
        "search_steps": list(decision.search_steps),
        "predicted_selected_effect_by_class": {str(class_id): float(value) for class_id, value in decision.per_class_effect.items()},
        "predicted_class_balanced_effect": float(decision.overall_effect),
        "predicted_worst_class_effect": float(decision.worst_class_effect),
        "confirmation": confirmation,
        "diagnostics_by_module": diagnostics,
        "replaced_probe_modules": list(replaced_modules),
        "recipe": recipe,
    }
    return _write_json(path, payload)


def run_module_c_preflight_selection(
    args: Any,
    model: nn.Module,
    data_loader_train: Any,
    data_loader_val: Any,
    device: torch.device,
    criterion_builder: Optional[Callable[[Any, torch.device], nn.Module]] = None,
    is_main_process: bool = True,
) -> ModuleCPreflightResult:
    """Resolve a nonempty B/D/E subset before the real LoRA model is built."""

    if str(getattr(args, "task_mod", "")) != "Classification" or int(getattr(args, "nb_classes", 0)) < 3:
        raise ValueError("Module C validation-risk selection supports only classification tasks with at least three classes.")
    candidate_modules = tuple(parse_module_ids(getattr(args, "module_c_candidates", "B,D,E")))
    if not candidate_modules:
        raise RuntimeError("Module C requires at least one B/D/E candidate.")
    if data_loader_train is None or data_loader_val is None:
        raise RuntimeError("Module C preflight requires distinct support/train and validation dataloaders.")

    train_batches = _collect_probe_batches(data_loader_train, int(getattr(args, "module_c_preflight_train_batches", 0)))
    val_batches = _collect_probe_batches(data_loader_val, int(getattr(args, "module_c_preflight_val_batches", 0)))
    if not train_batches or not val_batches:
        raise RuntimeError("Module C preflight could not collect both support and validation batches.")

    original_training = bool(model.training)
    original_requires_grad = {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}
    try:
        model.to(device)
        replaced = apply_lora_to_eegfm(
            model=model,
            model_name=str(getattr(args, "model_name", "")),
            lora_target="module_c",
            module_c_selected=candidate_modules,
            module_b_sites=getattr(args, "module_b_sites", "both"),
            r=int(getattr(args, "lora_rank", 4)),
            alpha=float(getattr(args, "lora_alpha", 8.0)),
            dropout=0.0,
            verbose=False,
        )
        if not replaced:
            raise RuntimeError(
                f"Module C preflight injected no B/D/E probe adapters for model={getattr(args, 'model_name', '')}."
            )
        model.eval()
        adapter_param_to_module, parameter_counts = _build_adapter_param_to_module(
            model=model,
            model_name=str(getattr(args, "model_name", "")),
            candidate_modules=candidate_modules,
            replaced_modules=replaced,
        )
        if not adapter_param_to_module:
            raise RuntimeError("Module C preflight could not map injected LoRA adapter parameters to B/D/E.")
        _configure_adapter_trainability(model, adapter_param_to_module)
        named_parameters = _named_adapter_parameters(model, adapter_param_to_module)
        criterion = criterion_builder(args, device) if criterion_builder is not None else _default_criterion(args, device)

        support_gradients = _full_support_gradients(
            args, model, train_batches, device, criterion, named_parameters
        )
        virtual_updates: Dict[str, Dict[str, torch.Tensor]] = {}
        for module_id in candidate_modules:
            virtual_updates[module_id] = _optimizer_step_from_gradients(
                args=args,
                model=model,
                named_parameters=named_parameters,
                adapter_param_to_module=adapter_param_to_module,
                gradients=support_gradients,
                selected_modules=(module_id,),
                restore_parameters=True,
            ).get(module_id, {})

        validation_gradients, validation_class_counts = _class_grouped_validation_gradients(
            args=args,
            model=model,
            batches=val_batches,
            device=device,
            named_parameters=named_parameters,
            adapter_param_to_module=adapter_param_to_module,
            candidate_modules=candidate_modules,
        )
        module_effects = first_order_effects_from_snapshots(validation_gradients, virtual_updates)
        decision = select_validation_risk_subset(module_effects, parameter_counts)
        selected_before = tuple(decision.selected_modules)

        _optimizer_step_from_gradients(
            args=args,
            model=model,
            named_parameters=named_parameters,
            adapter_param_to_module=adapter_param_to_module,
            gradients=support_gradients,
            selected_modules=selected_before,
            restore_parameters=False,
        )
        full_validation_loss = _classwise_validation_loss(args, model, val_batches, device)
        masked_validation_loss: Dict[str, Dict[int, float]] = {}
        for module_id in selected_before[1:]:
            with _masked_module_adapter(model, adapter_param_to_module, module_id):
                masked_validation_loss[module_id] = _classwise_validation_loss(args, model, val_batches, device)
        selected_after, pruning = prune_harmful_confirmation_additions(
            selected_modules=selected_before,
            primary_module=selected_before[0],
            full_per_class_loss=full_validation_loss,
            masked_per_class_loss=masked_validation_loss,
        )
        for module_id, record in decision.candidate_decisions.items():
            record["selected_before_confirmation"] = int(module_id in selected_before)
            record["selected_after_confirmation"] = int(module_id in selected_after)
            record["selected"] = int(module_id in selected_after)
        if selected_after != selected_before:
            confirmed_effect = _combined_effects(module_effects, selected_after)
            confirmed_mean = sum(confirmed_effect.values()) / float(len(confirmed_effect))
            confirmed_worst = min(confirmed_effect.values())
            decision = replace(
                decision,
                selected_modules=selected_after,
                per_class_effect=confirmed_effect,
                overall_effect=float(confirmed_mean),
                worst_class_effect=float(confirmed_worst),
                reason=f"{decision.reason}_confirmation_pruned",
            )
        confirmation = {
            "selected_before": list(selected_before),
            "selected_after": list(selected_after),
            "full_per_class_loss": {str(class_id): float(value) for class_id, value in full_validation_loss.items()},
            "masked_per_class_loss": {
                module_id: {str(class_id): float(value) for class_id, value in per_class.items()}
                for module_id, per_class in masked_validation_loss.items()
            },
            "pruning": pruning,
        }
        diagnostics = _build_diagnostics(
            args=args,
            model=model,
            candidate_modules=candidate_modules,
            decision=decision,
            module_effects=module_effects,
            adapter_param_to_module=adapter_param_to_module,
            parameter_counts=parameter_counts,
            support_gradients=support_gradients,
            virtual_updates=virtual_updates,
            confirmation=confirmation,
        )

        setattr(args, "module_c_enable", True)
        setattr(args, "module_c_resolved_candidates", ",".join(candidate_modules))
        setattr(args, "module_c_resolved_selected", ",".join(decision.selected_modules))
        setattr(args, "module_c_selection_rule", MODULE_C_PREFLIGHT_SELECTION_RULE)

        output_dir = str(getattr(args, "output_dir", "") or "")
        score_path = ""
        decision_path = ""
        if is_main_process and output_dir:
            score_rows = [diagnostics[module_id] for module_id in candidate_modules]
            score_path = _write_csv(os.path.join(output_dir, MODULE_C_PREFLIGHT_SCORE_FILE), score_rows)
            decision_path = _write_decision_json(
                path=os.path.join(output_dir, MODULE_C_PREFLIGHT_DECISION_FILE),
                args=args,
                decision=decision,
                candidate_modules=candidate_modules,
                diagnostics=diagnostics,
                confirmation=confirmation,
                validation_class_counts=validation_class_counts,
                parameter_counts=parameter_counts,
                replaced_modules=replaced,
            )
        print(
            "[ModuleC] validation-risk selected "
            f"{','.join(decision.selected_modules)} ({decision.reason}); "
            f"predicted mean={decision.overall_effect:.6g}, worst={decision.worst_class_effect:.6g}."
        )
        return ModuleCPreflightResult(
            decision=decision,
            diagnostics_by_module=diagnostics,
            replaced_modules=tuple(replaced),
            score_csv_path=score_path,
            decision_json_path=decision_path,
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_training)
        current_parameters = dict(model.named_parameters())
        for name, requires_grad in original_requires_grad.items():
            if name in current_parameters:
                current_parameters[name].requires_grad_(requires_grad)
