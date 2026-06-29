# --------------------------------------------------------
# Module C: zero-update preflight LoRA module selector.
#
# This file turns Module C from an offline candidate-training replay into a
# cheap pre-training policy. It inserts temporary probe adapters, measures
# gradient pressure on train/validation batches, writes interpretable
# diagnostics, and lets module_c_lora_search pick the final B/D/E subset.
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
    ModuleCDecision,
    ModuleCPolicyConfig,
    ModuleCScore,
    is_module_c_baseline_candidate,
    parse_module_ids,
    select_module_subset,
)
from .module_e_structural_routing import (
    module_e_module_prefix_from_name,
    structural_inventory_from_model,
)


MODULE_C_PREFLIGHT_SELECTION_RULE = "zero_update_preflight_no_test"
MODULE_C_PREFLIGHT_SCORE_FILE = "module_c_preflight_scores.csv"
MODULE_C_PREFLIGHT_INTERACTION_FILE = "module_c_preflight_interactions.csv"
MODULE_C_PREFLIGHT_DECISION_FILE = "module_c_preflight_decision.json"


@dataclass(frozen=True)
class ModuleCPreflightResult:
    decision: ModuleCDecision
    diagnostics_by_module: Mapping[str, Mapping[str, Any]]
    interaction_scores: Mapping[Tuple[str, str], float]
    replaced_modules: Tuple[str, ...]
    score_csv_path: str = ""
    interaction_csv_path: str = ""
    decision_json_path: str = ""


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


def _build_scores_and_interactions(
    args: Any,
    candidate_modules: Sequence[str],
    train_snap: Mapping[str, Mapping[str, Any]],
    train_seen: int,
    val_snap: Mapping[str, Mapping[str, Any]],
    val_seen: int,
    hard_snap: Mapping[str, Mapping[str, Any]],
    hard_seen: int,
    stable_snap: Mapping[str, Mapping[str, Any]],
    stable_seen: int,
    hard_classes: Sequence[int],
    class_counts: Mapping[int, int],
    class_loss_mean: Mapping[int, float],
    class_profiles: Mapping[str, Mapping[int, float]],
    adapter_param_counts: Mapping[str, int],
    probe_param_counts: Mapping[str, int],
) -> Tuple[List[ModuleCScore], Dict[str, Dict[str, Any]], Dict[Tuple[str, str], float]]:
    del args
    train_energy = {
        m: float(train_snap[m]["energy"]) / max(1, train_seen)
        for m in candidate_modules
    }
    val_energy = {
        m: float(val_snap[m]["energy"]) / max(1, val_seen)
        for m in candidate_modules
    }
    hard_energy = {
        m: float(hard_snap[m]["energy"]) / max(1, hard_seen)
        for m in candidate_modules
    }
    stable_energy = {
        m: float(stable_snap[m]["energy"]) / max(1, stable_seen)
        for m in candidate_modules
    }
    per_param = {
        m: (train_energy[m] + val_energy[m]) / max(1, int(probe_param_counts.get(m, 0)))
        for m in candidate_modules
    }
    max_pressure = max([v for v in per_param.values() if v > 0.0], default=0.0)
    complexity = _complexity_values(adapter_param_counts, candidate_modules)
    hard_samples = sum(int(class_counts.get(int(cls), 0)) for cls in hard_classes)
    reliability = min(1.0, float(hard_samples) / max(1.0, float(sum(class_counts.values()) or 1)))

    scores: List[ModuleCScore] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for module_id in candidate_modules:
        mps = 0.0 if max_pressure <= 0.0 else per_param[module_id] / max_pressure
        lrf_energy = float(train_snap[module_id].get("lrf_energy", 0.0))
        lrf = (
            float(train_snap[module_id].get("lrf_weighted", 0.0)) / lrf_energy
            if lrf_energy > 0.0
            else 1.0
        )
        tva_raw = _cosine_named_vectors(
            train_snap[module_id].get("vectors", {}),
            val_snap[module_id].get("vectors", {}),
        )
        stability = max(0.0, tva_raw)
        hard_total = hard_energy[module_id]
        val_total = max(val_energy[module_id], 1e-12)
        hcl = _clip01((hard_total / val_total) * reliability)
        hard_stable_cos = _cosine_named_vectors(
            hard_snap[module_id].get("vectors", {}),
            stable_snap[module_id].get("vectors", {}),
        )
        conflict = max(0.0, -hard_stable_cos)
        pressure_gap = abs(train_energy[module_id] - val_energy[module_id]) / max(
            train_energy[module_id] + val_energy[module_id],
            1e-12,
        )
        overfit_risk = max(0.0, pressure_gap - 0.50) + max(0.0, -tva_raw)
        overfit_risk = _clip01(overfit_risk)

        diag = {
            "module_id": module_id,
            "metric_source": MODULE_C_PREFLIGHT_SELECTION_RULE,
            "test_used_for_selection": 0,
            "pressure": _clip01(mps),
            "module_pressure_raw": per_param[module_id],
            "low_rank_fit": _clip01(lrf, default=1.0),
            "hard_class_leverage": hcl,
            "train_val_agreement": stability,
            "stability": stability,
            "class_conflict": _clip01(conflict),
            "overfit_risk": overfit_risk,
            "val_test_risk": overfit_risk,
            "complexity": complexity.get(module_id, 1.0),
            "train_grad_energy": train_energy[module_id],
            "val_grad_energy": val_energy[module_id],
            "hard_grad_energy": hard_energy[module_id],
            "stable_grad_energy": stable_energy[module_id],
            "adapter_param_count": int(adapter_param_counts.get(module_id, 0)),
            "probe_param_count": int(probe_param_counts.get(module_id, 0)),
            "hard_classes": ",".join(str(x) for x in hard_classes),
            "hard_class_sample_count": int(hard_samples),
            "hard_class_reliability": reliability,
            "class_loss_mean": json.dumps({str(k): v for k, v in sorted(class_loss_mean.items())}, ensure_ascii=False),
        }
        diagnostics[module_id] = diag
        scores.append(
            ModuleCScore(
                module_id=module_id,
                pressure=diag["pressure"],
                low_rank_fit=diag["low_rank_fit"],
                hard_class_leverage=diag["hard_class_leverage"],
                class_conflict=diag["class_conflict"],
                stability=diag["stability"],
                val_test_risk=diag["val_test_risk"],
                complexity=diag["complexity"],
                metadata=diag,
            )
        )

    interaction_scores: Dict[Tuple[str, str], float] = {}
    profile_classes = sorted({int(cls) for profile in class_profiles.values() for cls in profile.keys()})
    for left, right in combinations(candidate_modules, 2):
        cos = _profile_cosine(class_profiles.get(left, {}), class_profiles.get(right, {}), profile_classes)
        complement = max(0.0, 1.0 - cos)
        redundancy = max(0.0, cos - 0.85)
        pressure_floor = min(
            diagnostics[left]["pressure"],
            diagnostics[right]["pressure"],
        )
        interaction = 0.04 * complement * pressure_floor - 0.04 * redundancy
        interaction_scores[(left, right)] = float(max(-0.06, min(0.06, interaction)))
    return scores, diagnostics, interaction_scores


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


def _write_decision_json(
    path: str,
    args: Any,
    decision: ModuleCDecision,
    diagnostics: Mapping[str, Mapping[str, Any]],
    interaction_scores: Mapping[Tuple[str, str], float],
    replaced_modules: Sequence[str],
) -> Optional[str]:
    if not path:
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "module_c_selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE,
        "test_used_for_selection": 0,
        "model_name": str(getattr(args, "model_name", "") or ""),
        "dataset": str(getattr(args, "dataset", "") or ""),
        "subject_mod": str(getattr(args, "subject_mod", "") or ""),
        "k_shot": getattr(args, "k_shot", ""),
        "seed": getattr(args, "seed", ""),
        "candidate_modules": parse_module_ids(getattr(args, "module_c_candidates", "B,D,E")),
        "selected_modules": list(decision.selected_modules),
        "selected_score": float(decision.selected_score),
        "reason": decision.reason,
        "module_scores": {k: float(v) for k, v in decision.module_scores.items()},
        "subset_scores": {
            "+".join(key) if key else "none": float(value)
            for key, value in decision.subset_scores.items()
        },
        "interaction_scores": {
            "+".join(key): float(value)
            for key, value in interaction_scores.items()
        },
        "diagnostics_by_module": diagnostics,
        "replaced_probe_modules": list(replaced_modules),
        "recipe": decision.recipe,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def run_module_c_preflight_selection(
    args: Any,
    model: nn.Module,
    data_loader_train: Any,
    data_loader_val: Any,
    device: torch.device,
    criterion_builder: Optional[Callable[[Any, torch.device], nn.Module]] = None,
    is_main_process: bool = True,
) -> ModuleCPreflightResult:
    """Resolve Module C selected modules with a zero-update probe.

    The caller should pass a disposable base model. This function mutates that
    model by inserting temporary probe adapters, but it does not update weights
    and never touches the final training model.
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

    param_to_module, adapter_param_counts, probe_param_counts = _build_param_to_module(
        model=model,
        model_name=str(getattr(args, "model_name", "")),
        candidate_modules=candidate_modules,
        replaced_modules=replaced,
    )
    if not param_to_module:
        raise RuntimeError("Module C preflight could not map probe parameters back to candidate modules.")
    _configure_probe_trainability(model, param_to_module)

    criterion = criterion_builder(args, device) if criterion_builder is not None else _default_criterion(args, device)
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
        stable_classes = tuple(sorted(set(class_counts.keys()) - set(hard_classes)))
        if hard_classes:
            hard_snap, hard_seen = _backward_batches(
                args, model, val_batches, device, criterion, param_to_module, candidate_modules, rank, svd_max_numel,
                class_filter=hard_classes,
            )
        else:
            hard_snap, hard_seen = _empty_snapshot(candidate_modules), 0
        if stable_classes:
            stable_snap, stable_seen = _backward_batches(
                args, model, val_batches, device, criterion, param_to_module, candidate_modules, rank, svd_max_numel,
                class_filter=stable_classes,
            )
        else:
            stable_snap, stable_seen = _empty_snapshot(candidate_modules), 0
        profile_classes = tuple(sorted(class_counts.keys()))
        class_profiles = _class_pressure_profiles(
            args, model, val_batches, device, criterion, param_to_module, candidate_modules, rank, svd_max_numel,
            classes=profile_classes,
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_training)

    scores, diagnostics, interaction_scores = _build_scores_and_interactions(
        args=args,
        candidate_modules=candidate_modules,
        train_snap=train_snap,
        train_seen=train_seen,
        val_snap=val_snap,
        val_seen=val_seen,
        hard_snap=hard_snap,
        hard_seen=hard_seen,
        stable_snap=stable_snap,
        stable_seen=stable_seen,
        hard_classes=hard_classes,
        class_counts=class_counts,
        class_loss_mean=class_loss_mean,
        class_profiles=class_profiles,
        adapter_param_counts=adapter_param_counts,
        probe_param_counts=probe_param_counts,
    )

    config = ModuleCPolicyConfig(
        marginal_margin=float(getattr(args, "module_c_preflight_margin", 0.03)),
        min_module_score=float(getattr(args, "module_c_preflight_min_score", 0.01)),
        allow_empty=False,
    )
    decision = select_module_subset(
        scores,
        interaction_scores=interaction_scores,
        config=config,
        registry=DEFAULT_CANDIDATE_MODULES,
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

    if is_main_process:
        score_rows = [diagnostics[m] for m in candidate_modules if m in diagnostics]
        interaction_rows = [
            {
                "left_module": left,
                "right_module": right,
                "interaction_score": float(value),
                "selection_rule": MODULE_C_PREFLIGHT_SELECTION_RULE,
                "test_used_for_selection": 0,
            }
            for (left, right), value in sorted(interaction_scores.items())
        ]
        _write_csv(score_path, score_rows)
        _write_csv(interaction_path, interaction_rows)
        _write_decision_json(decision_path, args, decision, diagnostics, interaction_scores, replaced)
        print(f"[ModuleC] preflight scores saved to: {score_path}")
        print(f"[ModuleC] preflight decision saved to: {decision_path}")

    if not selected:
        raise RuntimeError(
            "Module C preflight selected no modules. This run will not fall back to qv/qv_ffn; "
            f"inspect {decision_path or MODULE_C_PREFLIGHT_DECISION_FILE} and adjust candidates or thresholds."
        )

    print(f"[ModuleC] preflight selected modules: {','.join(selected)} ({decision.reason})")
    return ModuleCPreflightResult(
        decision=decision,
        diagnostics_by_module=diagnostics,
        interaction_scores=interaction_scores,
        replaced_modules=tuple(replaced),
        score_csv_path=score_path,
        interaction_csv_path=interaction_path,
        decision_json_path=decision_path,
    )
