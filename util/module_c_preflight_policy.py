# --------------------------------------------------------
# Module C: zero-update preflight LoRA module selector.
#
# This file turns Module C from an offline candidate-training replay into a
# cheap pre-training policy. It calibrates the disposable task head, inserts
# temporary probe adapters, measures signed class-wise relief, writes
# interpretable diagnostics, and lets RGFS pick the final B/D/E subset.
# --------------------------------------------------------

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lora import apply_lora_to_eegfm, freeze_all_parameters
from .module_c_lora_search import (
    DEFAULT_CANDIDATE_MODULES,
    build_module_c_recipe,
    is_module_c_baseline_candidate,
    parse_module_ids,
)
from .module_c_rgfs_policy import (
    RGFSConfig,
    RGFSDecision,
    rgfs_config_dict,
    select_rgfs_subset,
)
from .module_e_structural_routing import (
    module_e_module_prefix_from_name,
    structural_inventory_from_model,
)


MODULE_C_PREFLIGHT_SELECTION_RULE = "rgfs_zero_update_preflight_no_test"
MODULE_C_PREFLIGHT_SCORE_FILE = "module_c_preflight_scores.csv"
MODULE_C_PREFLIGHT_INTERACTION_FILE = "module_c_preflight_interactions.csv"
MODULE_C_PREFLIGHT_DECISION_FILE = "module_c_preflight_decision.json"
MODULE_C_RGFS_RELIEF_FILE = "module_c_rgfs_relief.csv"
MODULE_C_ACTION_OVERLAP_FILE = "module_c_action_overlap.json"


@dataclass(frozen=True)
class ModuleCPreflightResult:
    decision: RGFSDecision
    diagnostics_by_module: Mapping[str, Mapping[str, Any]]
    interaction_scores: Mapping[Tuple[str, str], float]
    replaced_modules: Tuple[str, ...]
    score_csv_path: str = ""
    interaction_csv_path: str = ""
    decision_json_path: str = ""
    relief_csv_path: str = ""
    action_overlap_json_path: str = ""


def _is_module_c_execution_target(target: Any) -> bool:
    return str(target or "").lower() in ("module_c", "module_c_auto", "c_auto")


def module_c_preflight_requested(args: Any) -> bool:
    """Return whether this run should auto-resolve Module C before training."""
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def _clip01(value: Any, default: float = 0.0) -> float:
    out = _safe_float(value, default=default)
    return float(max(0.0, min(1.0, out)))


def _as_list(value: Iterable[Any]) -> List[Any]:
    return list(value) if value is not None else []


def _prepare_batch(args: Any, batch: Sequence[Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = batch[0]
    targets_raw = batch[1]
    if str(getattr(args, "norm_method", "")) == "mv":
        samples = samples.float().to(device, non_blocking=True) * float(getattr(args, "mv_norm_value", 0.01))
    else:
        samples = samples.float().to(device, non_blocking=True)

    labels = targets_raw.to(device, non_blocking=True)
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
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output.view(-1),
            targets.float().view(-1),
            reduction="none",
        )
        return loss.view(-1)
    return torch.nn.functional.cross_entropy(output, targets.long(), reduction="none")


def _collect_probe_batches(data_loader: Any, max_batches: int) -> List[Sequence[Any]]:
    batches: List[Sequence[Any]] = []
    if data_loader is None or max_batches <= 0:
        return batches
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            batches.append(batch)
    return batches


def _strip_lora_param_suffix(param_name: str) -> str:
    name = str(param_name or "")
    lower = name.lower()
    markers = (
        ".lora_a.",
        ".lora_b.",
        ".lora_a",
        ".lora_b",
        ".base.weight",
        ".base.bias",
        ".base.in_proj_weight",
        ".base.in_proj_bias",
        ".base.out_proj.weight",
        ".base.out_proj.bias",
    )
    for marker in markers:
        idx = lower.find(marker)
        if idx >= 0:
            return name[:idx]
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
    if "E" in candidates:
        for prefix in structural_prefixes:
            if prefix and _prefix_match(name, prefix):
                return "E"
    if "D" in candidates:
        return "D"
    return ""


def _build_param_to_module(
    model: nn.Module,
    model_name: str,
    candidate_modules: Sequence[str],
    replaced_modules: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, int]]:
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
    replacement_owner: Dict[str, str] = {}
    for replacement in replaced_modules:
        owner = _replacement_owner(replacement, candidate_modules, structural_prefixes)
        if owner:
            replacement_owner[str(replacement)] = owner

    param_to_module: Dict[str, str] = {}
    adapter_params: Dict[str, int] = {m: 0 for m in candidate_modules}
    probe_params: Dict[str, int] = {m: 0 for m in candidate_modules}
    for name, param in model.named_parameters():
        module_prefix = _strip_lora_param_suffix(name)
        owner = ""
        best_len = -1
        for replacement, candidate in replacement_owner.items():
            if _prefix_match(module_prefix, replacement) and len(replacement) > best_len:
                owner = candidate
                best_len = len(replacement)
        if not owner:
            continue
        param_to_module[name] = owner
        probe_params[owner] = probe_params.get(owner, 0) + int(param.numel())
        if ".lora_" in name.lower() or "input_side_lora" in name:
            adapter_params[owner] = adapter_params.get(owner, 0) + int(param.numel())
    return param_to_module, adapter_params, probe_params


def _configure_probe_trainability(model: nn.Module, param_to_module: Mapping[str, str]) -> None:
    freeze_all_parameters(model)
    for name, param in model.named_parameters():
        if name in param_to_module:
            param.requires_grad_(True)


def _is_base_gradient_name(name: str) -> bool:
    lower = str(name or "").lower()
    return ".lora_" not in lower and (
        ".base.weight" in lower
        or ".base.in_proj_weight" in lower
        or ".base.out_proj.weight" in lower
        or lower.endswith(".weight")
    )


def _low_rank_fit_from_grad(grad: torch.Tensor, rank: int, max_numel: int) -> Optional[float]:
    if grad is None or grad.numel() == 0 or grad.dim() < 2:
        return None
    if max_numel > 0 and grad.numel() > max_numel:
        return None
    matrix = grad.detach().float().reshape(grad.shape[0], -1).cpu()
    if matrix.numel() == 0 or min(matrix.shape) <= 0:
        return None
    try:
        singular_values = torch.linalg.svdvals(matrix)
    except Exception:
        return None
    energy = singular_values.pow(2)
    denom = float(energy.sum().item())
    if denom <= 0.0:
        return None
    k = max(1, min(int(rank), int(energy.numel())))
    return float(energy[:k].sum().item() / denom)


def _empty_snapshot(candidate_modules: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {
        m: {
            "energy": 0.0,
            "param_count": 0,
            "vectors": {},
            "lrf_weighted": 0.0,
            "lrf_energy": 0.0,
        }
        for m in candidate_modules
    }


def _snapshot_gradients(
    model: nn.Module,
    param_to_module: Mapping[str, str],
    candidate_modules: Sequence[str],
    rank: int,
    svd_max_numel: int,
) -> Dict[str, Dict[str, Any]]:
    snap = _empty_snapshot(candidate_modules)
    for name, param in model.named_parameters():
        module_id = param_to_module.get(name)
        if not module_id or param.grad is None:
            continue
        grad = param.grad.detach().float()
        energy = float(grad.pow(2).sum().cpu().item())
        item = snap[module_id]
        item["energy"] += energy
        item["param_count"] += int(grad.numel())
        item["vectors"][name] = grad.reshape(-1).cpu()
        if _is_base_gradient_name(name):
            fit = _low_rank_fit_from_grad(
                grad,
                rank=rank,
                max_numel=svd_max_numel,
            )
            if fit is not None and energy > 0.0:
                item["lrf_weighted"] += float(fit) * energy
                item["lrf_energy"] += energy
    return snap


def _merge_snapshots(
    target: Dict[str, Dict[str, Any]],
    source: Mapping[str, Mapping[str, Any]],
) -> None:
    for module_id, src in source.items():
        dst = target[module_id]
        dst["energy"] += float(src.get("energy", 0.0))
        dst["param_count"] = max(int(dst.get("param_count", 0)), int(src.get("param_count", 0)))
        dst["lrf_weighted"] += float(src.get("lrf_weighted", 0.0))
        dst["lrf_energy"] += float(src.get("lrf_energy", 0.0))
        dst_vectors = dst["vectors"]
        for name, vector in src.get("vectors", {}).items():
            if name in dst_vectors:
                dst_vectors[name] = dst_vectors[name] + vector
            else:
                dst_vectors[name] = vector.clone()


def _cosine_named_vectors(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    keys = sorted(set(left.keys()) & set(right.keys()))
    if not keys:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for key in keys:
        a = left[key].float().view(-1)
        b = right[key].float().view(-1)
        n = min(a.numel(), b.numel())
        if n <= 0:
            continue
        a = a[:n]
        b = b[:n]
        dot += float(torch.dot(a, b).item())
        left_norm += float(torch.dot(a, a).item())
        right_norm += float(torch.dot(b, b).item())
    denom = math.sqrt(max(left_norm, 0.0)) * math.sqrt(max(right_norm, 0.0))
    if denom <= 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, dot / denom)))


def _forward_loss(model: nn.Module, samples: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    output = model(samples)
    if isinstance(output, (list, tuple)):
        output = output[0]
    return criterion(output, targets)


def _calibrate_probe_head(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
    criterion: nn.Module,
) -> int:
    """Briefly fit only the disposable task head before measuring RGFS evidence."""
    steps = max(0, int(getattr(args, "module_c_probe_head_steps", 3)))
    if steps <= 0 or not batches or not hasattr(model, "task_head"):
        return 0

    freeze_all_parameters(model)
    params = []
    for param in model.task_head.parameters():
        param.requires_grad_(True)
        params.append(param)
    if not params:
        return 0

    lr = float(getattr(args, "module_c_probe_head_lr", 1e-3))
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    original_training = bool(model.training)
    completed = 0
    try:
        model.train(True)
        for step in range(steps):
            batch = batches[step % len(batches)]
            samples, targets, _labels = _prepare_batch(args, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _forward_loss(model, samples, targets, criterion)
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Module C probe-head calibration produced non-finite loss: {float(loss.detach().cpu())}")
            loss.backward()
            optimizer.step()
            completed += 1
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        model.train(original_training)
    return completed


def _backward_batches(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
    criterion: nn.Module,
    param_to_module: Mapping[str, str],
    candidate_modules: Sequence[str],
    rank: int,
    svd_max_numel: int,
    class_filter: Optional[Iterable[int]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    aggregate = _empty_snapshot(candidate_modules)
    filter_set = None if class_filter is None else set(int(x) for x in class_filter)
    seen = 0
    for batch in batches:
        samples, targets, labels_raw = _prepare_batch(args, batch, device)
        if filter_set is not None:
            labels = _classification_labels(labels_raw, args)
            if labels is None:
                continue
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls in filter_set:
                mask |= labels == int(cls)
            if not bool(mask.any().item()):
                continue
            samples = samples[mask]
            if int(getattr(args, "nb_classes", 0)) == 1:
                targets = targets.view(-1, 1)[mask]
            else:
                targets = targets.view(-1)[mask]

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = _forward_loss(model, samples, targets, criterion)
            if not torch.isfinite(loss.detach()):
                raise RuntimeError(f"Module C preflight produced non-finite loss: {float(loss.detach().cpu())}")
            loss.backward()
        snap = _snapshot_gradients(
            model=model,
            param_to_module=param_to_module,
            candidate_modules=candidate_modules,
            rank=rank,
            svd_max_numel=svd_max_numel,
        )
        _merge_snapshots(aggregate, snap)
        seen += 1
    model.zero_grad(set_to_none=True)
    return aggregate, seen


def _validation_class_stats(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
) -> Tuple[Tuple[int, ...], Dict[int, int], Dict[int, float]]:
    if str(getattr(args, "task_mod", "")) != "Classification":
        return tuple(), {}, {}
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
                    continue
                output = model(samples)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                losses = _per_sample_loss(output, targets, args)
                for cls in torch.unique(labels).detach().cpu().tolist():
                    cls = int(cls)
                    mask = labels == cls
                    if not bool(mask.any().item()):
                        continue
                    cls_losses = losses[mask]
                    counts[cls] = counts.get(cls, 0) + int(mask.sum().item())
                    loss_sum[cls] = loss_sum.get(cls, 0.0) + float(cls_losses.detach().sum().cpu().item())
    finally:
        model.train(original_training)

    loss_mean = {cls: loss_sum[cls] / max(1, counts.get(cls, 0)) for cls in loss_sum}
    ranked = sorted(loss_mean.items(), key=lambda item: (-item[1], item[0]))
    hard_k = max(1, int(getattr(args, "module_c_preflight_hard_k", 2)))
    hard = tuple(cls for cls, _loss in ranked[: min(hard_k, len(ranked))])
    return hard, counts, loss_mean


def _class_pressure_profiles(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
    criterion: nn.Module,
    param_to_module: Mapping[str, str],
    candidate_modules: Sequence[str],
    rank: int,
    svd_max_numel: int,
    classes: Sequence[int],
) -> Dict[str, Dict[int, float]]:
    profiles: Dict[str, Dict[int, float]] = {m: {} for m in candidate_modules}
    max_classes = max(0, int(getattr(args, "module_c_preflight_max_profile_classes", 8)))
    for cls in _as_list(classes)[:max_classes]:
        snap, seen = _backward_batches(
            args=args,
            model=model,
            batches=batches,
            device=device,
            criterion=criterion,
            param_to_module=param_to_module,
            candidate_modules=candidate_modules,
            rank=rank,
            svd_max_numel=svd_max_numel,
            class_filter=(int(cls),),
        )
        if seen <= 0:
            continue
        for module_id in candidate_modules:
            profiles[module_id][int(cls)] = float(snap[module_id]["energy"]) / float(seen)
    return profiles


def _profile_cosine(left: Mapping[int, float], right: Mapping[int, float], classes: Sequence[int]) -> float:
    if not classes:
        return 0.0
    a = torch.tensor([float(left.get(int(cls), 0.0)) for cls in classes], dtype=torch.float32)
    b = torch.tensor([float(right.get(int(cls), 0.0)) for cls in classes], dtype=torch.float32)
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, float(torch.dot(a, b).item() / denom))))


def _complexity_values(adapter_param_counts: Mapping[str, int], candidate_modules: Sequence[str]) -> Dict[str, float]:
    counts = [max(1, int(adapter_param_counts.get(m, 0))) for m in candidate_modules]
    sorted_counts = sorted(counts)
    if not sorted_counts:
        return {}
    median = float(sorted_counts[len(sorted_counts) // 2])
    return {
        m: float(max(0.5, min(1.5, max(1, int(adapter_param_counts.get(m, 0))) / max(1.0, median))))
        for m in candidate_modules
    }


def _filter_named_vectors(vectors: Mapping[str, torch.Tensor], kind: str) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    want_lora = str(kind).lower() == "lora"
    for name, vector in vectors.items():
        lower = str(name or "").lower()
        is_lora = "lora_" in lower or lower.endswith("lora_a") or lower.endswith("lora_b")
        if want_lora == is_lora:
            out[name] = vector
    return out


def _confidence_margin(args: Any, seen: int) -> float:
    scale = max(0.0, float(getattr(args, "module_c_rgfs_confidence_scale", 0.20)))
    if seen <= 0:
        return 1.0
    return float(min(0.50, scale / math.sqrt(float(seen))))


def _low_rank_fit_from_snapshot(snapshot: Mapping[str, Any]) -> float:
    lrf_energy = float(snapshot.get("lrf_energy", 0.0))
    if lrf_energy <= 0.0:
        return 1.0
    return _clip01(float(snapshot.get("lrf_weighted", 0.0)) / lrf_energy, default=1.0)


def _class_burden_from_loss(class_loss_mean: Mapping[int, float]) -> Dict[int, float]:
    losses = {int(cls): max(0.0, float(loss)) for cls, loss in class_loss_mean.items()}
    total = sum(losses.values())
    if total <= 0.0 and losses:
        uniform = 1.0 / float(len(losses))
        return {cls: uniform for cls in losses}
    if total <= 0.0:
        return {}
    return {cls: float(loss / total) for cls, loss in losses.items()}


def _class_gradient_snapshots(
    args: Any,
    model: nn.Module,
    batches: Sequence[Sequence[Any]],
    device: torch.device,
    criterion: nn.Module,
    param_to_module: Mapping[str, str],
    candidate_modules: Sequence[str],
    rank: int,
    svd_max_numel: int,
    classes: Sequence[int],
) -> Tuple[Dict[int, Dict[str, Dict[str, Any]]], Dict[int, int]]:
    snapshots: Dict[int, Dict[str, Dict[str, Any]]] = {}
    seen_by_class: Dict[int, int] = {}
    max_classes = max(0, int(getattr(args, "module_c_preflight_max_profile_classes", 8)))
    for cls in _as_list(classes)[:max_classes]:
        snap, seen = _backward_batches(
            args=args,
            model=model,
            batches=batches,
            device=device,
            criterion=criterion,
            param_to_module=param_to_module,
            candidate_modules=candidate_modules,
            rank=rank,
            svd_max_numel=svd_max_numel,
            class_filter=(int(cls),),
        )
        if seen <= 0:
            continue
        snapshots[int(cls)] = snap
        seen_by_class[int(cls)] = int(seen)
    return snapshots, seen_by_class


def _cosine_float_vectors(left: Mapping[int, float], right: Mapping[int, float], classes: Sequence[int]) -> float:
    return _profile_cosine(left, right, classes)


def _build_rgfs_evidence(
    args: Any,
    candidate_modules: Sequence[str],
    train_snap: Mapping[str, Mapping[str, Any]],
    train_seen: int,
    val_snap: Mapping[str, Mapping[str, Any]],
    val_seen: int,
    class_snaps: Mapping[int, Mapping[str, Mapping[str, Any]]],
    class_seen: Mapping[int, int],
    hard_classes: Sequence[int],
    class_counts: Mapping[int, int],
    class_loss_mean: Mapping[int, float],
    adapter_param_counts: Mapping[str, int],
    probe_param_counts: Mapping[str, int],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, str], float], List[Dict[str, Any]], Dict[int, float], Dict[str, Dict[int, float]], Dict[str, Dict[int, float]], Dict[str, float], Dict[str, Any]]:
    class_ids = tuple(sorted(int(cls) for cls in class_loss_mean.keys()))
    burden = _class_burden_from_loss(class_loss_mean)
    train_energy = {m: float(train_snap[m]["energy"]) / max(1, train_seen) for m in candidate_modules}
    val_energy = {m: float(val_snap[m]["energy"]) / max(1, val_seen) for m in candidate_modules}
    per_param = {
        m: (train_energy[m] + val_energy[m]) / max(1, int(probe_param_counts.get(m, 0)))
        for m in candidate_modules
    }
    max_pressure = max([v for v in per_param.values() if v > 0.0], default=0.0)
    complexity = _complexity_values(adapter_param_counts, candidate_modules)

    relief_lcb: Dict[str, Dict[int, float]] = {m: {} for m in candidate_modules}
    harm_lcb: Dict[str, Dict[int, float]] = {m: {} for m in candidate_modules}
    relief_rows: List[Dict[str, Any]] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}

    for module_id in candidate_modules:
        lrf = _low_rank_fit_from_snapshot(train_snap[module_id])
        train_vectors = train_snap[module_id].get("vectors", {})
        val_vectors = val_snap[module_id].get("vectors", {})
        train_val_cos = _cosine_named_vectors(train_vectors, val_vectors)
        module_relief_sum = 0.0
        module_harm_sum = 0.0
        signed_lora_cos_by_class: Dict[int, float] = {}
        signed_base_cos_by_class: Dict[int, float] = {}

        for cls in class_ids:
            cls_snap = class_snaps.get(int(cls), {})
            cls_module_snap = cls_snap.get(module_id, {}) if cls_snap else {}
            cls_vectors = cls_module_snap.get("vectors", {}) if cls_module_snap else {}
            seen = int(class_seen.get(int(cls), 0))
            margin = _confidence_margin(args, seen)

            base_cos = _cosine_named_vectors(
                _filter_named_vectors(train_vectors, "base"),
                _filter_named_vectors(cls_vectors, "base"),
            )
            lora_cos = _cosine_named_vectors(
                _filter_named_vectors(train_vectors, "lora"),
                _filter_named_vectors(cls_vectors, "lora"),
            )
            base_lcb = base_cos - margin
            base_ucb = base_cos + margin
            lora_lcb = lora_cos - margin
            lora_ucb = lora_cos + margin
            relief = max(0.0, lrf * max(0.0, base_lcb), max(0.0, lora_lcb))
            harm = max(0.0, -base_ucb, -lora_ucb)
            relief_lcb[module_id][int(cls)] = float(relief)
            harm_lcb[module_id][int(cls)] = float(harm)
            module_relief_sum += float(burden.get(int(cls), 0.0)) * float(relief)
            module_harm_sum += float(burden.get(int(cls), 0.0)) * float(harm)
            signed_lora_cos_by_class[int(cls)] = float(lora_cos)
            signed_base_cos_by_class[int(cls)] = float(base_cos)
            relief_rows.append(
                {
                    "module_id": module_id,
                    "class_id": int(cls),
                    "class_count": int(class_counts.get(int(cls), 0)),
                    "class_probe_batches": int(seen),
                    "class_loss_mean": float(class_loss_mean.get(int(cls), 0.0)),
                    "class_burden": float(burden.get(int(cls), 0.0)),
                    "confidence_margin": float(margin),
                    "low_rank_fit": float(lrf),
                    "base_signed_cosine": float(base_cos),
                    "base_relief_lcb": float(base_lcb),
                    "lora_tangent_signed_cosine": float(lora_cos),
                    "lora_tangent_relief_lcb": float(lora_lcb),
                    "relief_lcb": float(relief),
                    "harm_lcb": float(harm),
                    "class_grad_energy": float(cls_module_snap.get("energy", 0.0)) / max(1, seen) if cls_module_snap else 0.0,
                    "selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE,
                    "test_used_for_selection": 0,
                }
            )

        pressure_norm = 0.0 if max_pressure <= 0.0 else per_param[module_id] / max_pressure
        pressure_gap = abs(train_energy[module_id] - val_energy[module_id]) / max(train_energy[module_id] + val_energy[module_id], 1e-12)
        diagnostics[module_id] = {
            "module_id": module_id,
            "metric_source": MODULE_C_PREFLIGHT_SELECTION_RULE,
            "test_used_for_selection": 0,
            "pressure": _clip01(pressure_norm),
            "module_pressure_raw": per_param[module_id],
            "low_rank_fit": float(lrf),
            "train_val_signed_cosine": float(train_val_cos),
            "train_val_agreement": max(0.0, float(train_val_cos)),
            "train_val_mismatch_risk": _clip01(max(0.0, pressure_gap - 0.50) + max(0.0, -train_val_cos)),
            "rgfs_weighted_relief": float(module_relief_sum),
            "rgfs_weighted_harm": float(module_harm_sum),
            "complexity": complexity.get(module_id, 1.0),
            "train_grad_energy": train_energy[module_id],
            "val_grad_energy": val_energy[module_id],
            "adapter_param_count": int(adapter_param_counts.get(module_id, 0)),
            "probe_param_count": int(probe_param_counts.get(module_id, 0)),
            "hard_classes_by_loss": ",".join(str(x) for x in hard_classes),
            "class_loss_mean": json.dumps({str(k): v for k, v in sorted(class_loss_mean.items())}, ensure_ascii=False),
            "class_burden": json.dumps({str(k): v for k, v in sorted(burden.items())}, ensure_ascii=False),
            "relief_lcb_by_class": json.dumps({str(k): v for k, v in sorted(relief_lcb[module_id].items())}, ensure_ascii=False),
            "harm_lcb_by_class": json.dumps({str(k): v for k, v in sorted(harm_lcb[module_id].items())}, ensure_ascii=False),
            "signed_lora_cos_by_class": json.dumps({str(k): v for k, v in sorted(signed_lora_cos_by_class.items())}, ensure_ascii=False),
            "signed_base_cos_by_class": json.dumps({str(k): v for k, v in sorted(signed_base_cos_by_class.items())}, ensure_ascii=False),
        }

    overlap_scores: Dict[Tuple[str, str], float] = {}
    overlap_payload: Dict[str, Any] = {"selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE, "pairs": []}
    for left, right in combinations(candidate_modules, 2):
        cos = _cosine_float_vectors(relief_lcb.get(left, {}), relief_lcb.get(right, {}), class_ids)
        overlap_scores[(left, right)] = float(cos)
        overlap_payload["pairs"].append(
            {
                "left_module": left,
                "right_module": right,
                "relief_profile_cosine": float(cos),
                "redundancy_level": "high" if cos >= 0.85 else "low",
                "complementarity_level": "high" if cos <= 0.35 else "medium" if cos <= 0.70 else "low",
            }
        )
    return diagnostics, overlap_scores, relief_rows, burden, relief_lcb, harm_lcb, complexity, overlap_payload


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    if not path or not rows:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def _write_json(path: str, payload: Mapping[str, Any]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def _write_decision_json(
    path: str,
    args: Any,
    decision: RGFSDecision,
    diagnostics: Mapping[str, Mapping[str, Any]],
    overlap_scores: Mapping[Tuple[str, str], float],
    action_overlap: Mapping[str, Any],
    replaced_modules: Sequence[str],
    relief_csv_path: str = "",
    action_overlap_json_path: str = "",
) -> Optional[str]:
    if not path:
        return None
    candidate_modules = parse_module_ids(getattr(args, "module_c_candidates", "B,D,E"))
    recipe = build_module_c_recipe(
        decision.selected_modules,
        registry=DEFAULT_CANDIDATE_MODULES,
        candidate_modules=candidate_modules,
        module_scores={m: decision.candidate_decisions.get(m, {}).get("focus_marginal_gain", 0.0) for m in candidate_modules},
    )
    policy_config = {
        **rgfs_config_dict(
            RGFSConfig(
                min_marginal_gain=float(getattr(args, "module_c_preflight_min_score", 0.01)),
                tie_tolerance=float(getattr(args, "module_c_preflight_margin", 0.03)),
                harm_veto_threshold=float(getattr(args, "module_c_rgfs_harm_threshold", 0.05)),
                focus_burden_ratio=float(getattr(args, "module_c_rgfs_focus_ratio", 0.80)),
                allow_empty=True,
            )
        ),
        "train_batches": max(1, int(getattr(args, "module_c_preflight_train_batches", 1))),
        "val_batches": max(1, int(getattr(args, "module_c_preflight_val_batches", 1))),
        "probe_head_steps": max(0, int(getattr(args, "module_c_probe_head_steps", 3))),
        "probe_head_lr": float(getattr(args, "module_c_probe_head_lr", 1e-3)),
        "relief_confidence_scale": float(getattr(args, "module_c_rgfs_confidence_scale", 0.20)),
        "svd_max_numel": int(getattr(args, "module_c_preflight_svd_max_numel", 1000000)),
        "test_used_for_selection": 0,
    }
    payload = {
        "module_c_selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE,
        "test_used_for_selection": 0,
        "model_name": str(getattr(args, "model_name", "") or ""),
        "dataset": str(getattr(args, "dataset", "") or ""),
        "subject_mod": str(getattr(args, "subject_mod", "") or ""),
        "k_shot": getattr(args, "k_shot", ""),
        "seed": getattr(args, "seed", ""),
        "candidate_modules": candidate_modules,
        "selected_modules": list(decision.selected_modules),
        "selected_score": float(decision.selected_score),
        "reason": decision.reason,
        "policy_config": policy_config,
        "class_burden": {str(k): float(v) for k, v in decision.class_burden.items()},
        "class_coverage": {str(k): float(v) for k, v in decision.class_coverage.items()},
        "focus_classes": list(decision.focus_classes),
        "candidate_decisions": decision.candidate_decisions,
        "search_steps": list(decision.search_steps),
        "module_scores": {m: float(decision.candidate_decisions.get(m, {}).get("focus_marginal_gain", 0.0)) for m in candidate_modules},
        "subset_scores": {"+".join(decision.selected_modules) if decision.selected_modules else "none": float(decision.selected_score)},
        "interaction_scores": {"+".join(key): float(value) for key, value in overlap_scores.items()},
        "action_overlap": action_overlap,
        "diagnostics_by_module": diagnostics,
        "relief_csv_path": relief_csv_path,
        "action_overlap_json_path": action_overlap_json_path,
        "replaced_probe_modules": list(replaced_modules),
        "recipe": recipe,
        "qv_qvffn_role": "baseline_control_only_not_module_c_candidate",
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
    """Resolve Module C selected modules with a zero-update RGFS probe.

    The caller passes a disposable model. This function may briefly calibrate
    the disposable task head and insert temporary probe adapters, but it never
    updates or reuses the final training model.
    """
    candidate_modules = tuple(parse_module_ids(getattr(args, "module_c_candidates", "B,D,E")))
    candidate_modules = tuple(m for m in candidate_modules if not is_module_c_baseline_candidate(m))
    if not candidate_modules:
        raise RuntimeError("Module C preflight has no valid candidate modules after excluding qv-style baselines.")
    if data_loader_train is None:
        raise RuntimeError("Module C preflight requires a training dataloader.")

    train_batches = _collect_probe_batches(
        data_loader_train,
        max(1, int(getattr(args, "module_c_preflight_train_batches", 1))),
    )
    val_batches = _collect_probe_batches(
        data_loader_val if data_loader_val is not None else data_loader_train,
        max(1, int(getattr(args, "module_c_preflight_val_batches", 1))),
    )
    if not train_batches or not val_batches:
        raise RuntimeError("Module C preflight could not collect train/validation probe batches.")

    original_training = bool(model.training)
    model.to(device)
    model.eval()
    criterion = criterion_builder(args, device) if criterion_builder is not None else _default_criterion(args, device)
    head_steps_completed = _calibrate_probe_head(args, model, train_batches, device, criterion)
    model.eval()

    replaced = apply_lora_to_eegfm(
        model=model,
        model_name=str(getattr(args, "model_name", "")),
        lora_target="module_c",
        module_c_selected=candidate_modules,
        r=int(getattr(args, "lora_rank", 4)),
        alpha=float(getattr(args, "lora_alpha", 8.0)),
        dropout=float(getattr(args, "module_c_preflight_dropout", 0.0)),
        verbose=False,
    )
    if not replaced:
        raise RuntimeError(
            f"Module C preflight injected no probe adapters for model={getattr(args, 'model_name', '')}, "
            f"candidates={','.join(candidate_modules)}."
        )
    model.to(device)

    param_to_module, adapter_param_counts, probe_param_counts = _build_param_to_module(
        model=model,
        model_name=str(getattr(args, "model_name", "")),
        candidate_modules=candidate_modules,
        replaced_modules=replaced,
    )
    if not param_to_module:
        raise RuntimeError("Module C preflight could not map probe parameters back to candidate modules.")
    _configure_probe_trainability(model, param_to_module)

    rank = int(getattr(args, "lora_rank", 4))
    svd_max_numel = int(getattr(args, "module_c_preflight_svd_max_numel", 1000000))

    try:
        train_snap, train_seen = _backward_batches(
            args, model, train_batches, device, criterion, param_to_module, candidate_modules, rank, svd_max_numel
        )
        val_snap, val_seen = _backward_batches(
            args, model, val_batches, device, criterion, param_to_module, candidate_modules, rank, svd_max_numel
        )
        hard_classes, class_counts, class_loss_mean = _validation_class_stats(args, model, val_batches, device)
        profile_classes = tuple(sorted(class_counts.keys()))
        class_snaps, class_seen = _class_gradient_snapshots(
            args, model, val_batches, device, criterion, param_to_module, candidate_modules, rank, svd_max_numel,
            classes=profile_classes,
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_training)

    diagnostics, overlap_scores, relief_rows, burden, relief_lcb, harm_lcb, complexity, action_overlap = _build_rgfs_evidence(
        args=args,
        candidate_modules=candidate_modules,
        train_snap=train_snap,
        train_seen=train_seen,
        val_snap=val_snap,
        val_seen=val_seen,
        class_snaps=class_snaps,
        class_seen=class_seen,
        hard_classes=hard_classes,
        class_counts=class_counts,
        class_loss_mean=class_loss_mean,
        adapter_param_counts=adapter_param_counts,
        probe_param_counts=probe_param_counts,
    )

    config = RGFSConfig(
        min_marginal_gain=float(getattr(args, "module_c_preflight_min_score", 0.01)),
        tie_tolerance=float(getattr(args, "module_c_preflight_margin", 0.03)),
        harm_veto_threshold=float(getattr(args, "module_c_rgfs_harm_threshold", 0.05)),
        focus_burden_ratio=float(getattr(args, "module_c_rgfs_focus_ratio", 0.80)),
        allow_empty=True,
    )
    decision = select_rgfs_subset(
        module_ids=candidate_modules,
        class_ids=sorted(burden.keys()),
        burden=burden,
        relief_lcb=relief_lcb,
        harm_lcb=harm_lcb,
        complexity=complexity,
        config=config,
    )

    for module_id, record in decision.candidate_decisions.items():
        if module_id in diagnostics:
            diagnostics[module_id].update(
                {
                    "rgfs_gate": record.get("gate", ""),
                    "rgfs_focus_marginal_gain": record.get("focus_marginal_gain", 0.0),
                    "rgfs_marginal_gain": record.get("marginal_gain", 0.0),
                    "rgfs_blocked_harm_classes": ",".join(str(x) for x in record.get("blocked_harm_classes", ())),
                    "rgfs_selected": int(module_id in decision.selected_modules),
                    "probe_head_steps_completed": int(head_steps_completed),
                }
            )

    selected = tuple(decision.selected_modules)
    setattr(args, "module_c_enable", True)
    setattr(args, "module_c_resolved_candidates", ",".join(candidate_modules))
    setattr(args, "module_c_resolved_selected", ",".join(selected))
    setattr(args, "module_c_selection_rule", MODULE_C_PREFLIGHT_SELECTION_RULE)
    setattr(args, "module_c_preflight_selected_modules", ",".join(selected))

    output_dir = str(getattr(args, "output_dir", "") or "").strip()
    diag_dir = os.path.join(output_dir, "diagnostics") if output_dir else ""
    score_path = os.path.join(diag_dir, MODULE_C_PREFLIGHT_SCORE_FILE) if diag_dir else ""
    interaction_path = os.path.join(diag_dir, MODULE_C_PREFLIGHT_INTERACTION_FILE) if diag_dir else ""
    decision_path = os.path.join(diag_dir, MODULE_C_PREFLIGHT_DECISION_FILE) if diag_dir else ""
    relief_path = os.path.join(diag_dir, MODULE_C_RGFS_RELIEF_FILE) if diag_dir else ""
    overlap_path = os.path.join(diag_dir, MODULE_C_ACTION_OVERLAP_FILE) if diag_dir else ""

    if is_main_process:
        score_rows = [diagnostics[m] for m in candidate_modules if m in diagnostics]
        interaction_rows = [
            {
                "left_module": left,
                "right_module": right,
                "relief_profile_cosine": float(value),
                "selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE,
                "test_used_for_selection": 0,
            }
            for (left, right), value in sorted(overlap_scores.items())
        ]
        _write_csv(score_path, score_rows)
        _write_csv(interaction_path, interaction_rows)
        _write_csv(relief_path, relief_rows)
        _write_json(overlap_path, action_overlap)
        _write_decision_json(decision_path, args, decision, diagnostics, overlap_scores, action_overlap, replaced, relief_path, overlap_path)
        print(f"[ModuleC] RGFS relief saved to: {relief_path}")
        print(f"[ModuleC] RGFS decision saved to: {decision_path}")

    if selected:
        print(f"[ModuleC] RGFS selected modules: {','.join(selected)} ({decision.reason})")
    else:
        print("[ModuleC] RGFS selected no B/D/E modules; continuing without qv/qv_ffn fallback.")

    return ModuleCPreflightResult(
        decision=decision,
        diagnostics_by_module=diagnostics,
        interaction_scores=overlap_scores,
        replaced_modules=tuple(replaced),
        score_csv_path=score_path,
        interaction_csv_path=interaction_path,
        decision_json_path=decision_path,
        relief_csv_path=relief_path,
        action_overlap_json_path=overlap_path,
    )
