"""Matched exhaustive low-fidelity selection for Module C.

The probe is disposable.  It anchors the downstream head for one support pass,
then gives every B/D/E subset the same one-pass training budget from the same
anchored state.  Selection uses validation macro log-loss only; no
probe parameters, optimizer state, or head weights enter formal training.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .lora import apply_lora_to_eegfm, freeze_all_parameters
from .module_c_exhaustive_policy import (
    ExhaustiveDecision,
    SubsetRisk,
    enumerate_action_subsets,
    select_exhaustive_subset,
)
from .module_c_lora_search import (
    DEFAULT_CANDIDATE_MODULES,
    MODULE_C_SELECTION_RULE,
    build_module_c_recipe,
    parse_module_ids,
)
from .module_e_structural_routing import (
    attach_module_e_dynamic_pressure_controller,
    module_e_branch_from_lora_param_name,
)
from .optim_factory import LayerDecayValueAssigner, create_optimizer


MODULE_C_PREFLIGHT_SELECTION_RULE = MODULE_C_SELECTION_RULE
MODULE_C_PREFLIGHT_SCORE_FILE = "module_c_preflight_scores.csv"
MODULE_C_PREFLIGHT_DECISION_FILE = "module_c_preflight_decision.json"


@dataclass(frozen=True)
class ModuleCDecision:
    selected_modules: Tuple[str, ...]
    reason: str
    selection_status: str
    runner_up_modules: Tuple[str, ...]
    selection_gap: float
    observed_gain: float
    ranked_subsets: Tuple[Tuple[str, ...], ...]
    conditional_contributions: Mapping[str, float]
    pair_interactions: Mapping[str, float]
    triple_interaction: Optional[float]


@dataclass(frozen=True)
class ActionOwnership:
    action_parameter_names: Mapping[str, Tuple[str, ...]]
    action_replacement_names: Mapping[str, Tuple[str, ...]]
    action_wrapped_base_parameter_names: Mapping[str, Tuple[str, ...]]
    adapter_parameter_owner: Mapping[str, str]
    parameter_default_trainable: Mapping[str, bool]
    parameter_counts: Mapping[str, int]
    replaced_modules: Tuple[str, ...]


@dataclass(frozen=True)
class ModuleCPreflightResult:
    decision: ModuleCDecision
    diagnostics_by_module: Mapping[str, Mapping[str, Any]]
    ownership: ActionOwnership
    head_anchor: Mapping[str, Any]
    branch_traces: Mapping[Tuple[str, ...], Mapping[str, Any]]
    replaced_modules: Tuple[str, ...]
    score_csv_path: str = ""
    decision_json_path: str = ""


@dataclass(frozen=True)
class _BranchEvaluation:
    subset: Tuple[str, ...]
    per_sample_loss: Tuple[float, ...]
    labels: Tuple[int, ...]
    per_class_loss: Dict[int, float]
    macro_loss: float
    micro_loss: float
    support_loss: Optional[float]
    support_examples: int
    optimizer_steps: int
    adapter_parameter_count: int
    trainable_parameter_count: int
    support_fingerprint: str
    optimizer_schedule_trace: Tuple[Dict[str, Any], ...]
    elapsed_seconds: float

    def summary(self) -> Dict[str, Any]:
        return {
            "subset": list(self.subset),
            "support_loss": None if self.support_loss is None else float(self.support_loss),
            "support_examples": int(self.support_examples),
            "optimizer_steps": int(self.optimizer_steps),
            "validation_per_class_loss": {int(k): float(v) for k, v in self.per_class_loss.items()},
            "validation_macro_log_loss": float(self.macro_loss),
            "validation_micro_log_loss": float(self.micro_loss),
            "validation_examples": len(self.per_sample_loss),
            "validation_loss_source": "direct_per_example_log_loss",
            "adapter_parameter_count": int(self.adapter_parameter_count),
            "trainable_parameter_count": int(self.trainable_parameter_count),
            "support_fingerprint": self.support_fingerprint,
            "optimizer_schedule_trace": [dict(row) for row in self.optimizer_schedule_trace],
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def _is_module_c_execution_target(target: Any) -> bool:
    return str(target or "").lower() in ("module_c", "module_c_auto", "c_auto")


def module_c_preflight_requested(args: Any) -> bool:
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


def capture_module_c_rng_state(data_loaders: Sequence[Any] = ()) -> Dict[str, Any]:
    """Capture process and explicit loader-generator state for exact restoration."""

    loader_states = []
    for loader in data_loaders:
        entries = []
        for owner_name, owner in (
            ("loader", loader),
            ("sampler", getattr(loader, "sampler", None)),
            ("batch_sampler", getattr(loader, "batch_sampler", None)),
        ):
            generator = getattr(owner, "generator", None) if owner is not None else None
            if isinstance(generator, torch.Generator):
                entries.append((owner_name, generator, generator.get_state().clone()))
        loader_states.append(entries)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "loaders": loader_states,
    }


def restore_module_c_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    for entries in state.get("loaders", ()):
        for _owner_name, generator, generator_state in entries:
            generator.set_state(generator_state)


def _prepare_batch(args: Any, batch: Sequence[Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = batch[0]
    labels_raw = batch[1]
    if str(getattr(args, "norm_method", "")) == "mv":
        samples = samples.float().to(device, non_blocking=True) * float(getattr(args, "mv_norm_value", 0.01))
    else:
        samples = samples.float().to(device, non_blocking=True)
    labels = labels_raw.to(device, non_blocking=True)
    if int(getattr(args, "nb_classes", 0)) == 1:
        targets = labels.float().view(-1, 1)
    else:
        targets = labels.long().view(-1)
    return samples, targets, labels


def _classification_labels(labels: torch.Tensor, args: Any) -> torch.Tensor:
    if int(getattr(args, "nb_classes", 0)) == 1:
        return labels.detach().view(-1).long().clamp(min=0, max=1)
    return labels.detach().view(-1).long()


def _default_criterion(args: Any, device: torch.device) -> nn.Module:
    del device
    if int(getattr(args, "nb_classes", 0)) == 1:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _per_sample_log_loss(output: torch.Tensor, targets: torch.Tensor, args: Any) -> torch.Tensor:
    if int(getattr(args, "nb_classes", 0)) == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            output.view(-1), targets.float().view(-1), reduction="none"
        )
    return torch.nn.functional.cross_entropy(output, targets.long(), reduction="none")


def _forward_output(model: nn.Module, samples: torch.Tensor) -> torch.Tensor:
    output = model(samples)
    return output[0] if isinstance(output, (list, tuple)) else output


def _collect_probe_batches(data_loader: Any, max_batches: int) -> List[Sequence[Any]]:
    batches: List[Sequence[Any]] = []
    limit = int(max_batches)
    if data_loader is None:
        return batches
    for batch_index, batch in enumerate(data_loader):
        if limit > 0 and batch_index >= limit:
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            batches.append(batch)
    return batches


def _resolve_module_c_support_batch_limit(
    loader_batch_count: int, debug_batch_cap: int
) -> int:
    loader_limit = int(loader_batch_count)
    if loader_limit <= 0:
        raise ValueError("Module C requires a nonempty support loader.")
    debug_cap = int(debug_batch_cap)
    return min(loader_limit, debug_cap) if debug_cap > 0 else loader_limit


def _support_fingerprint(batches: Sequence[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for index, batch in enumerate(batches):
        samples = batch[0]
        labels = batch[1]
        digest.update(str(index).encode("ascii"))
        digest.update(str(tuple(samples.shape)).encode("ascii"))
        digest.update(str(tuple(labels.shape)).encode("ascii"))
        digest.update(samples.detach().cpu().contiguous().numpy().tobytes())
        digest.update(labels.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _is_adapter_parameter(name: str) -> bool:
    lower = str(name or "").lower()
    return ".lora_a" in lower or ".lora_b" in lower or lower.endswith("lora_a") or lower.endswith("lora_b")


def install_module_c_action_registry(
    model: nn.Module,
    model_name: str,
    candidate_modules: Sequence[str],
    module_b_sites: str,
    r: int,
    alpha: float,
    dropout: float,
    seed: int = 0,
) -> ActionOwnership:
    """Install each action separately and audit exact, disjoint LoRA ownership."""

    candidates = tuple(parse_module_ids(candidate_modules))
    if not candidates:
        raise ValueError("Module C requires at least one B/D/E registry action.")
    action_parameters: Dict[str, Tuple[str, ...]] = {}
    action_replacements: Dict[str, Tuple[str, ...]] = {}
    action_wrapped_base_ids: Dict[str, Tuple[int, ...]] = {}
    owner: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    all_replacements: List[str] = []
    initial_trainability = {id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()}
    owned_wrapped_base_ids = set()
    for action in candidates:
        before_named = dict(model.named_parameters())
        before_names_by_id = {id(parameter): name for name, parameter in before_named.items()}
        before = {name for name, _ in model.named_parameters() if _is_adapter_parameter(name)}
        replaced = tuple(
            apply_lora_to_eegfm(
                model=model,
                model_name=str(model_name),
                lora_target="module_c",
                module_c_selected=(action,),
                module_b_sites=module_b_sites,
                r=int(r),
                alpha=float(alpha),
                dropout=float(dropout),
                module_c_seed=int(seed),
                verbose=False,
            )
        )
        after_named = dict(model.named_parameters())
        after_names_by_id = {id(parameter): name for name, parameter in after_named.items()}
        after = {name for name in after_named if _is_adapter_parameter(name)}
        created = tuple(sorted(after - before))
        if not replaced or not created:
            raise RuntimeError(f"Module C action {action} owns no injected LoRA surface for model={model_name}.")
        overlap = sorted(set(replaced).intersection(all_replacements))
        if overlap:
            raise RuntimeError(f"Module C action surfaces overlap for model={model_name}: {overlap[:5]}")
        wrapped_base_ids = tuple(
            sorted(
                parameter_id
                for parameter_id, before_name in before_names_by_id.items()
                if parameter_id in after_names_by_id
                and any(
                    before_name == module_name or before_name.startswith(f"{module_name}.")
                    for module_name in replaced
                )
            )
        )
        wrapped_overlap = sorted(set(wrapped_base_ids).intersection(owned_wrapped_base_ids))
        if wrapped_overlap:
            raise RuntimeError(
                f"Module C actions wrap the same base parameters for model={model_name}: {wrapped_overlap[:5]}"
            )
        action_parameters[action] = created
        action_replacements[action] = tuple(sorted(replaced))
        action_wrapped_base_ids[action] = wrapped_base_ids
        counts[action] = sum(int(after_named[name].numel()) for name in created)
        for name in created:
            if name in owner:
                raise RuntimeError(f"Module C adapter parameter has multiple owners: {name}")
            owner[name] = action
        all_replacements.extend(replaced)
        owned_wrapped_base_ids.update(wrapped_base_ids)
    final_adapter_names = {name for name, _ in model.named_parameters() if _is_adapter_parameter(name)}
    unowned = sorted(final_adapter_names - set(owner))
    if unowned:
        raise RuntimeError(f"Module C found unowned injected LoRA tensors: {unowned[:5]}")
    final_named = dict(model.named_parameters())
    final_names_by_id = {id(parameter): name for name, parameter in final_named.items()}
    action_wrapped_bases = {
        action: tuple(sorted(final_names_by_id[parameter_id] for parameter_id in parameter_ids))
        for action, parameter_ids in action_wrapped_base_ids.items()
    }
    parameter_default_trainable = {
        name: bool(initial_trainability.get(id(parameter), False))
        for name, parameter in final_named.items()
    }
    return ActionOwnership(
        action_parameter_names=action_parameters,
        action_replacement_names=action_replacements,
        action_wrapped_base_parameter_names=action_wrapped_bases,
        adapter_parameter_owner=owner,
        parameter_default_trainable=parameter_default_trainable,
        parameter_counts=counts,
        replaced_modules=tuple(all_replacements),
    )


def _head_parameter_names(model: nn.Module) -> Tuple[str, ...]:
    named = dict(model.named_parameters())
    head_ids = set()
    task_head = getattr(model, "task_head", None)
    if isinstance(task_head, nn.Module):
        head_ids.update(id(parameter) for parameter in task_head.parameters())
    names = [name for name, parameter in named.items() if id(parameter) in head_ids]
    if names:
        return tuple(sorted(names))
    fallback_suffixes = (
        "main_model.model.cls_head.weight",
        "main_model.model.cls_head.bias",
        "classification_head.weight",
        "classification_head.bias",
    )
    return tuple(sorted(name for name in named if name.endswith(fallback_suffixes)))


def _optimizer_layer_decay_callbacks(args: Any, model: nn.Module) -> Tuple[Optional[Callable[[str], int]], Optional[Callable[[int], float]]]:
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


def _create_probe_optimizer(args: Any, model: nn.Module, controller: Any = None) -> torch.optim.Optimizer:
    get_num_layer, get_layer_scale = _optimizer_layer_decay_callbacks(args, model)
    optimizer = create_optimizer(
        args,
        model,
        skip_list=[],
        get_num_layer=get_num_layer,
        get_layer_scale=get_layer_scale,
        get_param_group_tag=(controller.optimizer_group_tag if controller is not None else None),
    )
    base_lr = float(getattr(args, "lr", 0.0))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * float(group.get("lr_scale", 1.0))
    if controller is not None:
        controller.bind_optimizer(optimizer)
    return optimizer


def _configure_branch_trainability(
    model: nn.Module,
    args: Any,
    subset: Sequence[str],
    ownership: ActionOwnership,
) -> None:
    active = set(subset)
    base_update = str(getattr(args, "lora_base_update", "freeze") or "freeze").lower()
    if base_update not in ("freeze", "full"):
        raise ValueError(f"Unknown lora_base_update={base_update}")

    # Every branch starts from the same frozen union model. Full-update branches
    # then restore the pre-injection trainability of original parameters only.
    # Active and inactive actions therefore share exactly the same Full FT base;
    # a branch differs from another branch only by its enabled LoRA parameters.
    freeze_all_parameters(model)
    named = dict(model.named_parameters())
    if base_update == "full":
        for name, parameter in named.items():
            restore = bool(ownership.parameter_default_trainable.get(name, False))
            if restore and not (parameter.is_floating_point() or parameter.is_complex()):
                raise RuntimeError(f"Module C cannot restore gradients for non-differentiable parameter {name}.")
            parameter.requires_grad_(restore)

    with torch.no_grad():
        for name, owner in ownership.adapter_parameter_owner.items():
            parameter = named[name]
            is_active = owner in active
            parameter.requires_grad_(is_active)
            if not is_active:
                parameter.zero_()

    head_names = set(_head_parameter_names(model))
    train_head = bool(getattr(args, "lora_train_head", True))
    if base_update == "freeze" and train_head:
        for name in head_names:
            named[name].requires_grad_(True)
    if base_update == "full" and not train_head:
        for name in head_names:
            named[name].requires_grad_(False)

    if base_update == "freeze" and bool(getattr(args, "lora_train_chan_conv", False)):
        for name, parameter in named.items():
            if name == "chan_conv" or name.startswith("chan_conv."):
                parameter.requires_grad_(True)

    if str(getattr(args, "model_name", "")) == "CBraMod":
        if base_update == "freeze" and bool(getattr(args, "cbra_train_patch_embed_when_frozen", False)):
            for name, parameter in named.items():
                if "main_model.patch_embedding" in name:
                    parameter.requires_grad_(True)
        if base_update == "full" and bool(getattr(args, "cbra_freeze_patch_embed_in_full", False)):
            for name, parameter in named.items():
                if "main_model.patch_embedding" in name:
                    parameter.requires_grad_(False)
        if bool(getattr(args, "cbra_train_norm_bias", False)):
            train_bias = not bool(getattr(args, "cbra_train_norm_only", False))
            for name, parameter in named.items():
                lower = name.lower()
                if "norm" in lower or "layernorm" in lower or ".ln" in lower or (train_bias and lower.endswith(".bias")):
                    parameter.requires_grad_(True)


def _run_support_pass(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
    criterion: nn.Module,
    controller: Any = None,
    lr_schedule_values: Optional[Sequence[float]] = None,
    wd_schedule_values: Optional[Sequence[float]] = None,
    formal_epoch_zero_steps: Optional[int] = None,
    optimizer_schedule_trace: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[float], int, int]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        return None, 0, 0
    optimizer = _create_probe_optimizer(args, model, controller=controller)
    optimizer.zero_grad(set_to_none=True)
    update_freq = max(1, int(getattr(args, "update_freq", 1)))
    total_loss = 0.0
    total_examples = 0
    optimizer_steps = 0
    model.train(True)
    for batch_index, batch in enumerate(batches):
        samples, targets, _labels = _prepare_batch(args, batch, device)
        output = _forward_output(model, samples)
        loss = criterion(output, targets)
        if loss.ndim != 0:
            loss = loss.mean()
        if not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError("Module C support pass produced a non-finite loss.")
        batch_size = int(samples.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_examples += batch_size
        accumulation_group_start = (batch_index // update_freq) * update_freq
        accumulation_group_size = min(
            update_freq, len(batches) - accumulation_group_start
        )
        (loss / float(accumulation_group_size)).backward()
        should_step = (batch_index + 1) % update_freq == 0 or batch_index + 1 == len(batches)
        if should_step:
            schedule_index = optimizer_steps
            if formal_epoch_zero_steps is not None:
                epoch_zero_steps = int(formal_epoch_zero_steps)
                if epoch_zero_steps <= 0:
                    raise ValueError("Module C requires positive formal epoch-zero schedule geometry.")
                if optimizer_steps > epoch_zero_steps:
                    raise RuntimeError(
                        "Module C support coverage produced more than one optimizer step beyond formal epoch zero."
                    )
                schedule_index = min(optimizer_steps, epoch_zero_steps - 1)
            if lr_schedule_values is not None:
                if schedule_index >= len(lr_schedule_values):
                    raise RuntimeError("Module C LR schedule is shorter than the support optimizer-step budget.")
                lr_value = float(lr_schedule_values[schedule_index])
                for group in optimizer.param_groups:
                    group["lr"] = lr_value * float(group.get("lr_scale", 1.0))
            else:
                lr_value = None
            if wd_schedule_values is not None:
                if schedule_index >= len(wd_schedule_values):
                    raise RuntimeError("Module C WD schedule is shorter than the support optimizer-step budget.")
                wd_value = float(wd_schedule_values[schedule_index])
                for group in optimizer.param_groups:
                    if float(group.get("weight_decay", 0.0)) > 0.0:
                        group["weight_decay"] = wd_value
            else:
                wd_value = None
            if optimizer_schedule_trace is not None:
                optimizer_schedule_trace.append(
                    {
                        "optimizer_step": int(optimizer_steps),
                        "lr_schedule_value": lr_value,
                        "weight_decay_schedule_value": wd_value,
                    }
                )
            prepared = False
            step_applied = False
            try:
                if controller is not None:
                    controller.prepare_optimizer_step(
                        optimizer, global_step=optimizer_steps, epoch=1
                    )
                    prepared = True
                clip_grad = getattr(args, "clip_grad", None)
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(trainable, float(clip_grad))
                optimizer.step()
                step_applied = True
                optimizer_steps += 1
            finally:
                if controller is not None and prepared:
                    controller.finish_optimizer_step(optimizer, step_applied=step_applied)
            optimizer.zero_grad(set_to_none=True)
    return float(total_loss / total_examples), total_examples, optimizer_steps


def _summarize_validation_log_losses(
    per_sample_losses: Sequence[float],
    labels: Sequence[int],
) -> Tuple[Dict[int, float], float, float]:
    """Aggregate aligned CE/BCE losses into per-class, macro, and micro risks."""

    losses = tuple(float(value) for value in per_sample_losses)
    class_labels = tuple(int(value) for value in labels)
    if not losses or len(losses) != len(class_labels):
        raise ValueError("Module C validation losses and labels must be nonempty and aligned.")
    if any(not math.isfinite(value) for value in losses):
        raise ValueError("Module C validation produced a non-finite per-example log-loss.")
    per_class = {
        class_id: float(np.mean([loss for loss, label in zip(losses, class_labels) if label == class_id]))
        for class_id in sorted(set(class_labels))
    }
    if len(per_class) < 2:
        raise ValueError("Module C requires at least two observed validation labels.")
    return per_class, float(np.mean(list(per_class.values()))), float(np.mean(losses))


def _validation_losses(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
) -> Tuple[Tuple[float, ...], Tuple[int, ...], Dict[int, float], float, float]:
    losses_out: List[float] = []
    labels_out: List[int] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            samples, targets, labels_raw = _prepare_batch(args, batch, device)
            labels = _classification_labels(labels_raw, args)
            losses = _per_sample_log_loss(_forward_output(model, samples), targets, args)
            losses_out.extend(float(value) for value in losses.detach().cpu().tolist())
            labels_out.extend(int(value) for value in labels.detach().cpu().tolist())
    per_class, macro_loss, micro_loss = _summarize_validation_log_losses(
        losses_out, labels_out
    )
    return tuple(losses_out), tuple(labels_out), per_class, macro_loss, micro_loss


def _remove_dynamic_controller(model: nn.Module) -> None:
    controller = getattr(model, "_module_e_dynamic_pressure_controller", None)
    if controller is None:
        return
    for handle in getattr(controller, "_hook_handles", ()):
        handle.remove()
    delattr(model, "_module_e_dynamic_pressure_controller")


def _anchor_task_head(
    args: Any,
    model: nn.Module,
    support_batches: Sequence[Sequence[Any]],
    validation_batches: Sequence[Sequence[Any]],
    device: torch.device,
    criterion: nn.Module,
    effective_classes: int,
    lr_schedule_values: Optional[Sequence[float]] = None,
    wd_schedule_values: Optional[Sequence[float]] = None,
    formal_epoch_zero_steps: Optional[int] = None,
) -> Dict[str, Any]:
    initial_losses, _labels, initial_per_class, initial_macro, initial_micro = _validation_losses(
        args, model, validation_batches, device
    )
    del initial_losses
    head_names = _head_parameter_names(model)
    named = dict(model.named_parameters())
    before = {name: named[name].detach().cpu().clone() for name in head_names}
    freeze_all_parameters(model)
    for name in head_names:
        named[name].requires_grad_(True)
    anchor_rng = capture_module_c_rng_state()
    restore_module_c_rng_state(anchor_rng)
    optimizer_schedule_trace: List[Dict[str, Any]] = []
    support_loss, support_examples, optimizer_steps = _run_support_pass(
        args,
        model,
        support_batches,
        device,
        criterion,
        lr_schedule_values=lr_schedule_values,
        wd_schedule_values=wd_schedule_values,
        formal_epoch_zero_steps=formal_epoch_zero_steps,
        optimizer_schedule_trace=optimizer_schedule_trace,
    ) if head_names else (None, 0, 0)
    _final_losses, _labels, final_per_class, final_macro, final_micro = _validation_losses(
        args, model, validation_batches, device
    )
    parameter_delta_sq = 0.0
    for name in head_names:
        delta = named[name].detach().cpu().float() - before[name].float()
        parameter_delta_sq += float(torch.sum(delta * delta).item())
    uniform_reference = float(math.log(effective_classes))
    return {
        "status": "trained" if head_names else "no_exposed_head_parameters",
        "parameter_names": list(head_names),
        "parameter_count": sum(int(named[name].numel()) for name in head_names),
        "parameter_delta_l2": float(math.sqrt(parameter_delta_sq)),
        "support_passes": 1 if head_names else 0,
        "support_examples": int(support_examples),
        "optimizer_steps": int(optimizer_steps),
        "optimizer_schedule_trace": optimizer_schedule_trace,
        "support_loss": None if support_loss is None else float(support_loss),
        "validation_loss_before": float(initial_macro),
        "validation_loss_after": float(final_macro),
        "validation_loss_improvement": float(initial_macro - final_macro),
        "validation_micro_log_loss_before": float(initial_micro),
        "validation_micro_log_loss_after": float(final_micro),
        "validation_per_class_before": initial_per_class,
        "validation_per_class_after": final_per_class,
        "uniform_log_loss_reference": uniform_reference,
        "below_uniform_reference": bool(final_macro < uniform_reference),
        "validity_role": "diagnostic_only_not_a_selection_gate",
    }


def _canonical_subset(actions: Sequence[str], candidate_order: Sequence[str]) -> Tuple[str, ...]:
    requested = set(actions)
    return tuple(action for action in candidate_order if action in requested)


def _validate_probe_training_controls(args: Any) -> None:
    """Reject formal controls that the one-pass probe cannot mirror exactly."""

    unsupported = []
    if float(getattr(args, "lora_delta_lambda", 0.0)) != 0.0:
        unsupported.append("lora_delta_lambda")
    if float(getattr(args, "cbra_l2sp_lambda", 0.0)) != 0.0:
        unsupported.append("cbra_l2sp_lambda")
    if bool(getattr(args, "cbra_train_wrapped_base", False)):
        unsupported.append("cbra_train_wrapped_base")
    for name in (
        "cbra_grad_scale_wrapped_base",
        "cbra_grad_scale_patch",
        "cbra_grad_scale_norm_bias",
    ):
        if float(getattr(args, name, 1.0)) != 1.0:
            unsupported.append(name)
    if bool(getattr(args, "enable_deepspeed", False)):
        unsupported.append("enable_deepspeed")
    if unsupported:
        raise ValueError(
            "Module C matched probe does not silently approximate these formal training controls: "
            + ", ".join(unsupported)
        )


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


def _subset_label(subset: Sequence[str]) -> str:
    return "+".join(subset) if subset else "EMPTY"


def _decision_rows(
    branches: Mapping[Tuple[str, ...], _BranchEvaluation],
    decision: ExhaustiveDecision,
    branch_gains: Mapping[str, float],
) -> List[Dict[str, Any]]:
    rank_by_subset = {
        subset: rank for rank, subset in enumerate(decision.ranked_subsets, start=1)
    }
    rows: List[Dict[str, Any]] = []
    for subset in (subset for subset in branches if subset):
        evaluation = branches[subset]
        label = _subset_label(subset)
        rows.append(
            {
                "branch": label,
                "subset": list(subset),
                "rank": rank_by_subset[subset],
                "selected": int(subset == decision.selected_subset),
                "selection_status": decision.selection_status,
                "runner_up_subset": list(decision.runner_up_subset),
                "selection_gap": float(decision.selection_gap),
                "branch_gain": float(branch_gains[label]),
                "per_class_gain": dict(decision.per_class_gain[label]),
                "conditional_contributions": dict(decision.conditional_contributions),
                "pair_interactions": dict(decision.pair_interactions),
                "triple_interaction": decision.triple_interaction,
                **evaluation.summary(),
            }
        )
    return rows


def run_module_c_preflight_selection(
    args: Any,
    model: nn.Module,
    data_loader_train: Any,
    data_loader_val: Any,
    device: torch.device,
    criterion_builder: Optional[Callable[[Any, torch.device], nn.Module]] = None,
    is_main_process: bool = True,
    num_training_steps_per_epoch: Optional[int] = None,
    lr_schedule_values: Optional[Sequence[float]] = None,
    wd_schedule_values: Optional[Sequence[float]] = None,
) -> ModuleCPreflightResult:
    """Resolve a nonempty B/D/E subset with matched exhaustive branch trials."""

    started = time.perf_counter()
    raw_classes = int(getattr(args, "nb_classes", 0))
    effective_classes = 2 if raw_classes == 1 else raw_classes
    if str(getattr(args, "task_mod", "")) != "Classification" or effective_classes < 2:
        raise ValueError("Module C exhaustive selection supports classification with at least two labels.")
    _validate_probe_training_controls(args)
    candidate_modules = tuple(parse_module_ids(getattr(args, "module_c_candidates", "B,D,E")))
    canonical_candidates = tuple(DEFAULT_CANDIDATE_MODULES)
    if candidate_modules != canonical_candidates:
        raise ValueError(
            "Automatic Module C exhaustive selection requires exactly canonical B,D,E candidates; "
            f"got {','.join(candidate_modules) or 'EMPTY'}."
        )
    if data_loader_train is None or data_loader_val is None:
        raise RuntimeError("Module C requires distinct support/train and validation dataloaders.")
    if bool(getattr(data_loader_train, "drop_last", False)):
        raise ValueError("Module C exhaustive support requires drop_last=False for complete coverage.")

    support_loader_batch_count = len(data_loader_train)
    support_batch_limit = _resolve_module_c_support_batch_limit(
        support_loader_batch_count,
        int(getattr(args, "module_c_preflight_train_batches", 0)),
    )
    formal_epoch_zero_steps = (
        None if num_training_steps_per_epoch is None else int(num_training_steps_per_epoch)
    )
    if formal_epoch_zero_steps is not None and formal_epoch_zero_steps <= 0:
        raise ValueError("Module C requires positive formal epoch-zero schedule geometry.")
    support_batches = _collect_probe_batches(data_loader_train, support_batch_limit)
    validation_batches = _collect_probe_batches(
        data_loader_val, int(getattr(args, "module_c_preflight_val_batches", 0))
    )
    if not support_batches or not validation_batches:
        raise RuntimeError("Module C could not collect both support and validation batches.")
    support_fingerprint = _support_fingerprint(support_batches)
    original_trainability = {
        name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()
    }

    model.to(device)
    criterion = criterion_builder(args, device) if criterion_builder is not None else _default_criterion(args, device)
    head_anchor = _anchor_task_head(
        args,
        model,
        support_batches,
        validation_batches,
        device,
        criterion,
        effective_classes,
        lr_schedule_values=lr_schedule_values,
        wd_schedule_values=wd_schedule_values,
        formal_epoch_zero_steps=formal_epoch_zero_steps,
    )
    anchored_named = dict(model.named_parameters())
    if set(anchored_named) != set(original_trainability):
        raise RuntimeError("Module C head anchoring unexpectedly changed the model parameter registry.")
    for name, parameter in anchored_named.items():
        parameter.requires_grad_(original_trainability[name])
    model.to(torch.device("cpu"))
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ownership = install_module_c_action_registry(
        model=model,
        model_name=str(getattr(args, "model_name", "")),
        candidate_modules=candidate_modules,
        module_b_sites=str(getattr(args, "module_b_sites", "both")),
        r=int(getattr(args, "lora_rank", 4)),
        alpha=float(getattr(args, "lora_alpha", 8.0)),
        dropout=float(getattr(args, "lora_dropout", 0.0)),
        seed=int(getattr(args, "seed", 0)),
    )
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.to(device)
    injection_complete_rng = capture_module_c_rng_state()

    expected_classes = set(range(effective_classes))

    branch_cache: Dict[Tuple[str, ...], _BranchEvaluation] = {}

    def evaluate_subset(raw_subset: Sequence[str]) -> _BranchEvaluation:
        subset = _canonical_subset(raw_subset, candidate_modules)
        if subset in branch_cache:
            return branch_cache[subset]
        branch_started = time.perf_counter()
        model.load_state_dict(initial_state, strict=True)
        _remove_dynamic_controller(model)
        _configure_branch_trainability(model, args, subset, ownership)
        restore_module_c_rng_state(injection_complete_rng)
        branch_args = copy.copy(args)
        branch_args.output_dir = ""
        controller = None
        if "E" in subset and str(getattr(args, "module_e_mode", "")) == "dynamic_pressure_gate":
            branch_args.module_e_injected_names = ";".join(
                ownership.action_replacement_names["E"]
            )
            controller = attach_module_e_dynamic_pressure_controller(branch_args, model)
        branch_criterion = (
            criterion_builder(branch_args, device) if criterion_builder is not None else _default_criterion(branch_args, device)
        )
        optimizer_schedule_trace: List[Dict[str, Any]] = []
        support_loss, support_examples, optimizer_steps = _run_support_pass(
            branch_args,
            model,
            support_batches,
            device,
            branch_criterion,
            controller=controller,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            formal_epoch_zero_steps=formal_epoch_zero_steps,
            optimizer_schedule_trace=optimizer_schedule_trace,
        )
        losses, labels, per_class_loss, macro_loss, micro_loss = _validation_losses(
            branch_args, model, validation_batches, device
        )
        observed_classes = set(labels)
        if observed_classes != expected_classes:
            raise ValueError(
                f"Module C validation split must contain every expected label; expected={sorted(expected_classes)}, "
                f"observed={sorted(observed_classes)}."
            )
        empty_evaluation = branch_cache.get(())
        if empty_evaluation is not None and labels != empty_evaluation.labels:
            raise RuntimeError("Module C matched branches lost validation sample alignment.")
        named = dict(model.named_parameters())
        adapter_count = sum(
            int(named[name].numel())
            for name, owner in ownership.adapter_parameter_owner.items()
            if owner in set(subset)
        )
        evaluation = _BranchEvaluation(
            subset=subset,
            per_sample_loss=losses,
            labels=labels,
            per_class_loss=per_class_loss,
            macro_loss=macro_loss,
            micro_loss=micro_loss,
            support_loss=support_loss,
            support_examples=support_examples,
            optimizer_steps=optimizer_steps,
            adapter_parameter_count=adapter_count,
            trainable_parameter_count=sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
            support_fingerprint=support_fingerprint,
            optimizer_schedule_trace=tuple(optimizer_schedule_trace),
            elapsed_seconds=float(time.perf_counter() - branch_started),
        )
        _remove_dynamic_controller(model)
        branch_cache[subset] = evaluation
        return evaluation

    exhaustive_subsets = enumerate_action_subsets(candidate_modules)
    for subset in ((), *exhaustive_subsets):
        evaluate_subset(subset)

    branch_risks = {
        subset: SubsetRisk(
            subset=subset,
            macro_loss=evaluation.macro_loss,
            micro_loss=evaluation.micro_loss,
            per_class_loss=evaluation.per_class_loss,
            adapter_parameter_count=evaluation.adapter_parameter_count,
        )
        for subset, evaluation in branch_cache.items()
    }
    exhaustive_decision = select_exhaustive_subset(branch_risks, candidate_modules)
    selected = tuple(exhaustive_decision.selected_subset)
    final_reason = "global_minimum_validation_macro_log_loss"
    final_decision = ModuleCDecision(
        selected_modules=selected,
        reason=final_reason,
        selection_status=exhaustive_decision.selection_status,
        runner_up_modules=tuple(exhaustive_decision.runner_up_subset),
        selection_gap=float(exhaustive_decision.selection_gap),
        observed_gain=float(exhaustive_decision.observed_gain),
        ranked_subsets=tuple(exhaustive_decision.ranked_subsets),
        conditional_contributions=dict(exhaustive_decision.conditional_contributions),
        pair_interactions=dict(exhaustive_decision.pair_interactions),
        triple_interaction=exhaustive_decision.triple_interaction,
    )
    empty_macro_loss = branch_cache[()].macro_loss
    branch_gains = {
        _subset_label(subset): float(empty_macro_loss - evaluation.macro_loss)
        for subset, evaluation in branch_cache.items()
        if subset
    }
    diagnostics_by_module: Dict[str, Dict[str, Any]] = {}
    for action in candidate_modules:
        replacements = ownership.action_replacement_names[action]
        structural_branches = sorted(
            {
                branch
                for name in replacements
                for branch in [module_e_branch_from_lora_param_name(str(getattr(args, "model_name", "")), name)]
                if branch is not None
            }
        )
        diagnostics_by_module[action] = {
            "module_id": action,
            "functional_name": DEFAULT_CANDIDATE_MODULES[action]["name"],
            "functional_role": DEFAULT_CANDIDATE_MODULES[action]["role"],
            "functional_blocks": list(DEFAULT_CANDIDATE_MODULES[action]["blocks"]),
            "functional_diagnostics_used_for_ranking": 0,
            "common_ranking_measure": "validation_macro_log_loss",
            "adapter_parameter_count": int(ownership.parameter_counts[action]),
            "adapter_parameter_names": list(ownership.action_parameter_names[action]),
            "replacement_names": list(replacements),
            "structural_branches_reference_only": structural_branches,
            "selected": int(action in selected),
            "singleton_branch": branch_cache[(action,)].summary(),
            "singleton_gain": branch_gains[action],
            "per_class_gain": dict(exhaustive_decision.per_class_gain[action]),
            "conditional_contribution": exhaustive_decision.conditional_contributions.get(action),
        }

    setattr(args, "module_c_enable", True)
    setattr(args, "module_c_resolved_candidates", ",".join(candidate_modules))
    setattr(args, "module_c_resolved_selected", ",".join(selected))
    setattr(args, "module_c_selection_rule", MODULE_C_PREFLIGHT_SELECTION_RULE)

    branch_summaries = {subset: evaluation.summary() for subset, evaluation in branch_cache.items()}
    train_cap = int(getattr(args, "module_c_preflight_train_batches", 0))
    val_cap = int(getattr(args, "module_c_preflight_val_batches", 0))
    total_seconds = float(time.perf_counter() - started)
    support_visible_examples = sum(int(batch[0].shape[0]) for batch in support_batches)
    validation_visible_examples = sum(int(batch[0].shape[0]) for batch in validation_batches)
    support_raw_examples = len(data_loader_train.dataset)
    validation_raw_examples = len(data_loader_val.dataset)
    payload = {
        "module_c_selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE,
        "test_used_for_selection": 0,
        "candidate_modules": list(candidate_modules),
        "selected_modules": list(selected),
        "selection_reason": final_reason,
        "selection_status": exhaustive_decision.selection_status,
        "runner_up_modules": list(exhaustive_decision.runner_up_subset),
        "selection_gap": float(exhaustive_decision.selection_gap),
        "selected_observed_gain": float(exhaustive_decision.observed_gain),
        "ranked_subsets": [list(subset) for subset in exhaustive_decision.ranked_subsets],
        "score_definition": "validation_macro_log_loss",
        "gain_definition": "EMPTY_validation_macro_log_loss_minus_branch_validation_macro_log_loss",
        "aggregation_order": "per_example_log_loss_then_per_class_mean_then_equal_class_macro_mean",
        "exhaustive_selection": {
            "strategy": "evaluate_EMPTY_and_every_BDE_subset_once_then_rank_globally",
            "selectable_branch_count": len(exhaustive_subsets),
            "empty_is_selectable": 0,
            "tie_break_order": [
                "validation_macro_log_loss",
                "fewer_actions",
                "fewer_adapter_parameters",
                "canonical_BDE_order",
            ],
        },
        "branch_gains": branch_gains,
        "per_class_gains": {
            label: dict(gains)
            for label, gains in exhaustive_decision.per_class_gain.items()
        },
        "conditional_contributions": dict(exhaustive_decision.conditional_contributions),
        "pair_interactions": dict(exhaustive_decision.pair_interactions),
        "triple_interaction": exhaustive_decision.triple_interaction,
        "head_anchor": head_anchor,
        "probe_training": {
            "support_passes_per_branch": 1,
            "budget_unit_reason": "one complete sequential no-drop support pass",
            "optimizer": str(getattr(args, "opt", "")),
            "base_learning_rate": float(getattr(args, "lr", 0.0)),
            "learning_rate_rule": "formal epoch-zero schedule geometry",
            "partial_accumulation_tail_schedule_rule": "reuse_final_epoch_zero_lr_and_weight_decay_without_consuming_next_epoch",
            "partial_accumulation_gradient_rule": "average_over_the_actual_number_of_microbatches_in_each_step",
            "formal_epoch_zero_optimizer_steps": formal_epoch_zero_steps,
            "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            "lora_base_update": str(getattr(args, "lora_base_update", "")),
            "full_update_base_control": "same_pre_injection_base_trainability_for_every_branch",
            "lora_rank": int(getattr(args, "lora_rank", 0)),
            "lora_alpha": float(getattr(args, "lora_alpha", 0.0)),
            "lora_dropout": float(getattr(args, "lora_dropout", 0.0)),
            "support_batch_cap": train_cap,
            "support_loader_batch_count": support_loader_batch_count,
            "support_effective_batch_limit": support_batch_limit,
            "validation_batch_cap": val_cap,
            "support_scope": "complete" if train_cap <= 0 else "debug_capped",
            "validation_scope": "complete" if val_cap <= 0 else "debug_capped",
            "support_loader": {
                "sampler_type": type(getattr(data_loader_train, "sampler", None)).__name__,
                "drop_last": bool(getattr(data_loader_train, "drop_last", False)),
                "raw_dataset_size": int(support_raw_examples),
                "visible_example_count": int(support_visible_examples),
                "coverage_fraction": float(support_visible_examples / support_raw_examples) if support_raw_examples else 0.0,
            },
            "validation_loader": {
                "sampler_type": type(getattr(data_loader_val, "sampler", None)).__name__,
                "drop_last": bool(getattr(data_loader_val, "drop_last", False)),
                "raw_dataset_size": int(validation_raw_examples),
                "visible_example_count": int(validation_visible_examples),
                "coverage_fraction": float(validation_visible_examples / validation_raw_examples) if validation_raw_examples else 0.0,
            },
            "formal_state_transfer": 0,
        },
        "action_ownership": {
            action: {
                "parameter_count": int(ownership.parameter_counts[action]),
                "parameter_names": list(ownership.action_parameter_names[action]),
                "replacement_names": list(ownership.action_replacement_names[action]),
                "wrapped_base_parameter_names": list(ownership.action_wrapped_base_parameter_names[action]),
            }
            for action in candidate_modules
        },
        "branches": {"+".join(subset) if subset else "EMPTY": trace for subset, trace in branch_summaries.items()},
        "diagnostics_by_module": diagnostics_by_module,
        "runtime": {
            "total_seconds": total_seconds,
            "branch_count": len(branch_cache),
            "support_pass_count": len(branch_cache) + int(head_anchor["support_passes"]),
            "validation_pass_count": len(branch_cache) + 2,
        },
        "claim_boundary": "exhaustive_low_fidelity_BDE_selection_not_a_guarantee_of_the_final_test_winner",
        "recipe": build_module_c_recipe(
            selected,
            registry=DEFAULT_CANDIDATE_MODULES,
            candidate_modules=candidate_modules,
        ),
    }

    output_dir = str(getattr(args, "output_dir", "") or "")
    score_path = ""
    decision_path = ""
    if is_main_process and output_dir:
        score_path = _write_csv(
            os.path.join(output_dir, MODULE_C_PREFLIGHT_SCORE_FILE),
            _decision_rows(branch_cache, exhaustive_decision, branch_gains),
        )
        decision_path = _write_json(
            os.path.join(output_dir, MODULE_C_PREFLIGHT_DECISION_FILE),
            payload,
        )

    print(
        "[ModuleC] exhaustive low-fidelity selection chose "
        f"{','.join(selected)}; status={exhaustive_decision.selection_status}; "
        f"runner_up={','.join(exhaustive_decision.runner_up_subset)}; "
        f"gap={exhaustive_decision.selection_gap:.6g}; "
        f"branches={len(branch_cache)}, elapsed={total_seconds:.1f}s."
    )
    return ModuleCPreflightResult(
        decision=final_decision,
        diagnostics_by_module=diagnostics_by_module,
        ownership=ownership,
        head_anchor=head_anchor,
        branch_traces=branch_summaries,
        replaced_modules=ownership.replaced_modules,
        score_csv_path=score_path,
        decision_json_path=decision_path,
    )
