# --------------------------------------------------------
# Module D: semantic / FFN refinement utilities.
#
# Module D is the semantic-boundary refinement action in the current EEG
# framework. It controls LoRA targets that adapt FFN/MLP semantic blocks and
# exposes the SBR metric used to judge whether the adaptation refined weak class
# boundaries instead of simply moving errors to stable classes.
# --------------------------------------------------------

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch.nn as nn


MODULE_D_CURRENT = "semantic_boundary_refinement"
MODULE_D_ROLE = "semantic_ffn_boundary_calibration"
MODULE_D_METRIC_NAME = "semantic_boundary_refinement_score"
MODULE_D_METRIC_SHORT = "SBR"
MODULE_D_SBR_EVAL_FILE = "module_d_sbr_eval.csv"
MODULE_D_TARGET = "semantic"


# 这边是 D 模块的目标接口：当前正式 D 只认 semantic。
def normalize_lora_target(lora_target: Any) -> str:
    return str(lora_target or "").lower()


def is_module_d_target(lora_target: Any) -> bool:
    return normalize_lora_target(lora_target) == MODULE_D_TARGET


def should_lora_semantic_ffn(lora_target: Any) -> bool:
    return is_module_d_target(lora_target)


def module_d_variant(lora_target: Any) -> str:
    return "pure_semantic" if is_module_d_target(lora_target) else ""


# 这块给日志、fb_collect 和 resolved_recipe 用，主要说清楚这次 run 有没有真的打开 D。
def module_d_metadata(
    args: Optional[Any] = None,
    lora_target: Optional[Any] = None,
    lora_base_update: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return shared Module D metadata for logs, probes, and collected outputs."""
    target = lora_target
    base_update = lora_base_update
    if args is not None:
        if target is None:
            target = getattr(args, "lora_target", "")
        if base_update is None:
            base_update = getattr(args, "lora_base_update", "")

    target = str(target or "")
    base_update = str(base_update or "")
    active = int(is_module_d_target(target))
    touches_ffn = int(should_lora_semantic_ffn(target))
    pure = int(bool(active) and base_update.lower() == "freeze")
    composite = 0

    if pure:
        note = "pure_frozen_d_isolation"
    elif active and base_update.lower() == "full":
        note = "full_ft_plus_lora_confounded"
    elif active:
        note = "pure_semantic_refinement"
    else:
        note = ""

    return {
        "module_d_current": MODULE_D_CURRENT if active else "",
        "module_d_role": MODULE_D_ROLE if active else "",
        "module_d_is_active": active,
        "module_d_touches_semantic_ffn": touches_ffn,
        "module_d_is_pure_isolation": pure,
        "module_d_is_composite": composite,
        "module_d_variant": module_d_variant(target),
        "module_d_reference_metric": MODULE_D_METRIC_NAME if active else "",
        "module_d_attribution_note": note,
        "adapter_target": target,
        "lora_base_update": base_update,
    }


# 下面几个函数是留给 lora.py 的接口。
# semantic 打开就表示语义 FFN 路径整体可插。
def semantic_layer_mode(lora_target: Any) -> str:
    return "all"


def extract_semantic_layer_index(module_name: str) -> Optional[int]:
    """Best-effort parser for transformer layer indices in common EEGFM names."""
    parts = module_name.split(".")
    for key in ("blocks", "layers", "encoder", "transformer"):
        for i, part in enumerate(parts[:-1]):
            if part == key and i + 1 < len(parts):
                nxt = parts[i + 1]
                if nxt == "layers" and i + 2 < len(parts) and parts[i + 2].isdigit():
                    return int(parts[i + 2])
                if nxt.isdigit():
                    return int(nxt)
    for part in parts:
        if part.isdigit():
            return int(part)
    return None


def max_semantic_layer_index(model: nn.Module) -> Optional[int]:
    idxs: List[int] = []
    for name, _ in model.named_modules():
        idx = extract_semantic_layer_index(name)
        if idx is not None:
            idxs.append(idx)
    return max(idxs) if idxs else None


def layer_selected_for_semantic(module_name: str, lora_target: Any, max_idx: Optional[int]) -> bool:
    return should_lora_semantic_ffn(lora_target)


def should_include_csbrain_spectral_proj(lora_target: Any) -> bool:
    """Keep CSBrain spectral projection out of semantic-only targets."""
    return not is_module_d_target(lora_target)


# 这块开始是 SBR 的核心计算。前面这些小函数只是CSV用的
def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def select_reference_hard_classes(
    reference_recall: Sequence[Any],
    hard_k: int = 2,
    valid_indices: Optional[Iterable[int]] = None,
) -> Tuple[int, ...]:
    """Select hard classes by bottom-k reference recall without using test labels."""
    allowed = set(valid_indices) if valid_indices is not None else set(range(len(reference_recall)))
    candidates: List[Tuple[float, int]] = []
    for idx, value in enumerate(reference_recall):
        if idx not in allowed:
            continue
        recall = _safe_float(value)
        if recall is not None:
            candidates.append((recall, idx))
    if not candidates:
        raise ValueError("Cannot select hard classes without finite reference recalls.")
    k = max(1, min(int(hard_k), len(candidates)))
    return tuple(idx for _, idx in sorted(candidates, key=lambda item: (item[0], item[1]))[:k])


# SBR 的核心结果对象，后面写 CSV 时也直接从这里展开字段。
@dataclass(frozen=True)
class SemanticBoundaryRefinementScore:
    """SBR = mean hard-class recall gain - mean stable-class recall loss."""

    hard_classes: Tuple[int, ...]
    stable_classes: Tuple[int, ...]
    hard_gain: float
    stable_loss: float
    score: float
    deltas: Tuple[float, ...]
    reference_bacc: Optional[float] = None
    adapted_bacc: Optional[float] = None
    bacc_delta: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module_d_metric": MODULE_D_METRIC_NAME,
            "module_d_metric_short": MODULE_D_METRIC_SHORT,
            "hard_classes": ",".join(str(x) for x in self.hard_classes),
            "stable_classes": ",".join(str(x) for x in self.stable_classes),
            "hard_gain": self.hard_gain,
            "stable_loss": self.stable_loss,
            "sbr": self.score,
            "reference_bacc": "" if self.reference_bacc is None else self.reference_bacc,
            "adapted_bacc": "" if self.adapted_bacc is None else self.adapted_bacc,
            "bacc_delta": "" if self.bacc_delta is None else self.bacc_delta,
        }


def semantic_boundary_refinement_score(
    reference_recall: Sequence[Any],
    adapted_recall: Sequence[Any],
    hard_classes: Optional[Iterable[int]] = None,
    hard_k: int = 2,
    reference_bacc: Optional[Any] = None,
    adapted_bacc: Optional[Any] = None,
) -> SemanticBoundaryRefinementScore:
    """Compute SBR from a reference run and a Module D run.

    Hard classes default to the bottom-k reference recalls, so the metric can be
    computed from validation reference behavior without peeking at test labels.
    """
    if len(reference_recall) != len(adapted_recall):
        raise ValueError("reference_recall and adapted_recall must have the same length.")
    if len(reference_recall) == 0:
        raise ValueError("SBR needs at least one class recall.")

    valid: List[int] = []
    deltas: List[float] = []
    for idx, (ref, adapted) in enumerate(zip(reference_recall, adapted_recall)):
        ref_f = _safe_float(ref)
        adapted_f = _safe_float(adapted)
        if ref_f is None or adapted_f is None:
            deltas.append(float("nan"))
            continue
        valid.append(idx)
        deltas.append(float(adapted_f - ref_f))

    if not valid:
        raise ValueError("SBR needs at least one finite reference/adapted recall pair.")

    if hard_classes is None:
        hard = select_reference_hard_classes(reference_recall, hard_k=hard_k, valid_indices=valid)
    else:
        seen = set()
        hard_list: List[int] = []
        for raw_idx in hard_classes:
            idx = int(raw_idx)
            if idx in seen:
                continue
            if idx < 0 or idx >= len(reference_recall):
                raise ValueError(f"hard class index {idx} is out of range.")
            if idx not in valid:
                continue
            seen.add(idx)
            hard_list.append(idx)
        if not hard_list:
            raise ValueError("No finite hard classes remain for SBR.")
        hard = tuple(hard_list)

    stable = tuple(idx for idx in valid if idx not in set(hard))
    # hard 类希望涨；stable 类不要求一定涨，但如果被牺牲了就要扣分。
    hard_gain = _mean([deltas[idx] for idx in hard])
    stable_loss = _mean([max(0.0, -deltas[idx]) for idx in stable])

    ref_bacc = _safe_float(reference_bacc)
    d_bacc = _safe_float(adapted_bacc)
    bacc_delta = None if ref_bacc is None or d_bacc is None else float(d_bacc - ref_bacc)

    return SemanticBoundaryRefinementScore(
        hard_classes=tuple(hard),
        stable_classes=stable,
        hard_gain=float(hard_gain),
        stable_loss=float(stable_loss),
        score=float(hard_gain - stable_loss),
        deltas=tuple(deltas),
        reference_bacc=ref_bacc,
        adapted_bacc=d_bacc,
        bacc_delta=bacc_delta,
    )


# 下面这些是读 reference/adapted 结果表的小工具，主要兼容不同 CSV 字段命名。
def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _row_matches_optional(row: Mapping[str, Any], keys: Sequence[str], expected: Any) -> bool:
    expected_norm = _norm_text(expected)
    if not expected_norm:
        return True
    seen = False
    for key in keys:
        value = row.get(key, "")
        value_norm = _norm_text(value)
        if not value_norm:
            continue
        seen = True
        if value_norm == expected_norm:
            return True
    return not seen


def _row_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    key_norm = key.lower()
    for raw_key, value in row.items():
        if str(raw_key).lower() == key_norm:
            return value
    return None


def load_module_d_sbr_reference(
    csv_path: str,
    model_name: Optional[Any] = None,
    reference_name: Optional[Any] = None,
) -> Dict[str, str]:
    """Load one reference row for SBR from adaptive_swa_eval-style CSV files."""
    rows = _read_csv_rows(str(csv_path or ""))
    if not rows:
        raise FileNotFoundError(f"Module D SBR reference CSV has no rows: {csv_path}")

    model_keys = ("model_name", "model", "foundation_model", "fm")
    name_keys = ("reference_name", "name", "method", "run_tag", "tag", "adapter_target", "fb_recipe", "mode")
    candidates = [
        row
        for row in rows
        if _row_matches_optional(row, model_keys, model_name)
        and _row_matches_optional(row, name_keys, reference_name)
    ]
    if not candidates:
        raise ValueError(
            "No Module D SBR reference row matched "
            f"model={model_name!r}, reference_name={reference_name!r} in {csv_path}"
        )
    return dict(candidates[-1])


def _extract_split_recall(
    row: Mapping[str, Any],
    split: str,
    nb_classes: Optional[int] = None,
) -> List[Any]:
    split_norm = str(split or "").strip().lower()
    values: Dict[int, Any] = {}

    prefixes: List[str] = []
    if split_norm:
        prefixes.extend((f"{split_norm}_class_", f"{split_norm}_class"))
    if not split_norm or _norm_text(row.get("split", "")) == split_norm:
        prefixes.extend(("class_", "class"))

    for raw_key, value in row.items():
        key = str(raw_key).strip().lower()
        for prefix in prefixes:
            if not key.startswith(prefix):
                continue
            tail = key[len(prefix):]
            if tail.startswith("_"):
                tail = tail[1:]
            if tail.isdigit():
                values[int(tail)] = value
                break

    if not values:
        return []
    if nb_classes is None or int(nb_classes) <= 0:
        count = max(values) + 1
    else:
        count = int(nb_classes)
    return [values.get(i, float("nan")) for i in range(count)]


def _extract_split_bacc(row: Mapping[str, Any], split: str) -> Optional[Any]:
    split_norm = str(split or "").strip().lower()
    keys = []
    if split_norm:
        keys.extend(
            (
                f"{split_norm}_balanced_accuracy",
                f"{split_norm}_balanced_acc",
                f"{split_norm}_bacc",
            )
        )
    if not split_norm or _norm_text(row.get("split", "")) == split_norm:
        keys.extend(("balanced_accuracy", "balanced_acc", "bacc"))
    for key in keys:
        value = _row_value(row, key)
        if _safe_float(value) is not None:
            return value
    return None


def _clean_metric_value(value: Any) -> Any:
    out = _safe_float(value)
    if out is None:
        return ""
    return float(out)


def module_d_eval_row_from_details(
    val_stats: Optional[Mapping[str, Any]] = None,
    val_details: Optional[Mapping[str, Any]] = None,
    test_stats: Optional[Mapping[str, Any]] = None,
    test_details: Optional[Mapping[str, Any]] = None,
    source: str = "final_epoch",
) -> Dict[str, Any]:
    """Flatten current run validation/test metrics into adaptive_swa_eval-like keys."""
    # 训练刚结束时还不是 CSV 行，这里先压成 SBR 能读取的 val_class_* / test_class_* 格式。
    row: Dict[str, Any] = {"adapted_source": source}
    for prefix, stats in (("val", val_stats), ("test", test_stats)):
        if not stats:
            continue
        for key, value in stats.items():
            clean = _clean_metric_value(value)
            if clean != "":
                row[f"{prefix}_{key}"] = clean
    for prefix, details in (("val", val_details), ("test", test_details)):
        if not details:
            continue
        per_class = details.get("per_class_recall", None)
        if per_class is None:
            continue
        for idx, value in enumerate(per_class):
            row[f"{prefix}_class_{idx}"] = _clean_metric_value(value)
    return row


def _sbr_result_row(
    split: str,
    result: SemanticBoundaryRefinementScore,
    reference_recall: Sequence[Any],
    adapted_recall: Sequence[Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(metadata)
    row.update(result.as_dict())
    row["split"] = split
    row["hard_class_source_split"] = "val"
    for idx, value in enumerate(reference_recall):
        row[f"reference_recall_{idx}"] = _clean_metric_value(value)
    for idx, value in enumerate(adapted_recall):
        row[f"adapted_recall_{idx}"] = _clean_metric_value(value)
    for idx, value in enumerate(result.deltas):
        row[f"class_delta_{idx}"] = _clean_metric_value(value)
    return row


def module_d_sbr_rows(
    reference_row: Mapping[str, Any],
    adapted_row: Mapping[str, Any],
    hard_k: int = 2,
    nb_classes: Optional[int] = None,
    model_name: str = "",
    run_tag: str = "",
    reference_source: str = "",
    reference_name: str = "",
    adapted_source: str = "",
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], ...]:
    """Compute validation SBR and test SBR using validation-selected hard classes."""
    # 先在 validation 上选 hard class；test 只复用这个选择，避免拿 test 来挑类别。
    ref_val = _extract_split_recall(reference_row, "val", nb_classes=nb_classes)
    adapted_val = _extract_split_recall(adapted_row, "val", nb_classes=nb_classes)
    if not ref_val or not adapted_val:
        raise ValueError("Module D SBR needs val_class_* recalls in both reference and adapted rows.")

    metadata: Dict[str, Any] = {
        "module_d_current": MODULE_D_CURRENT,
        "module_d_role": MODULE_D_ROLE,
        "reference_source": str(reference_source or ""),
        "reference_name": str(reference_name or _row_text(reference_row, "reference_name", "method", "run_tag", "tag", "mode")),
        "adapted_source": str(adapted_source or _row_text(adapted_row, "adapted_source", "mode", "run_tag", "tag")),
        "model_name": str(model_name or _row_text(adapted_row, "model_name", "model") or _row_text(reference_row, "model_name", "model")),
        "run_tag": str(run_tag or _row_text(adapted_row, "run_tag", "tag")),
        "score_is_validation_only": 1,
        "test_used_for_hard_class_selection": 0,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))

    val_result = semantic_boundary_refinement_score(
        reference_recall=ref_val,
        adapted_recall=adapted_val,
        hard_k=hard_k,
        reference_bacc=_extract_split_bacc(reference_row, "val"),
        adapted_bacc=_extract_split_bacc(adapted_row, "val"),
    )
    rows = [
        _sbr_result_row(
            split="val",
            result=val_result,
            reference_recall=ref_val,
            adapted_recall=adapted_val,
            metadata=metadata,
        )
    ]

    ref_test = _extract_split_recall(reference_row, "test", nb_classes=nb_classes)
    adapted_test = _extract_split_recall(adapted_row, "test", nb_classes=nb_classes)
    if ref_test and adapted_test:
        test_result = semantic_boundary_refinement_score(
            reference_recall=ref_test,
            adapted_recall=adapted_test,
            hard_classes=val_result.hard_classes,
            hard_k=hard_k,
            reference_bacc=_extract_split_bacc(reference_row, "test"),
            adapted_bacc=_extract_split_bacc(adapted_row, "test"),
        )
        rows.append(
            _sbr_result_row(
                split="test",
                result=test_result,
                reference_recall=ref_test,
                adapted_recall=adapted_test,
                metadata=metadata,
            )
        )
    return tuple(rows)


def write_module_d_sbr_eval(csv_path: str, rows: Iterable[Mapping[str, Any]]) -> Optional[str]:
    rows = [dict(row) for row in rows]
    if not rows:
        return None
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _load_default_adapted_row(output_dir: str) -> Optional[Dict[str, str]]:
    diag_dir = os.path.join(str(output_dir or ""), "diagnostics")
    for name in ("adaptive_swa_eval.csv",):
        rows = _read_csv_rows(os.path.join(diag_dir, name))
        if rows:
            row = dict(rows[-1])
            row.setdefault("adapted_source", name)
            return row
    return None


def save_module_d_sbr_eval(
    args: Any,
    adapted_row: Optional[Mapping[str, Any]] = None,
    reference_csv_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
) -> Optional[str]:
    """Write diagnostics/module_d_sbr_eval.csv when a Module D reference is supplied."""
    # 训练后的统一入口：有 reference CSV 才能算 SBR，没有就跳过，不硬造指标。
    if args is None:
        return None
    if not bool(getattr(args, "module_d_sbr_eval", False)) and not str(reference_csv_path or getattr(args, "module_d_reference_csv", "") or "").strip():
        return None

    meta = module_d_metadata(args=args)
    if not int(meta.get("module_d_is_active", 0)):
        return None

    output_dir = str(getattr(args, "output_dir", "") or "")
    if not output_dir:
        return None

    ref_path = str(reference_csv_path or getattr(args, "module_d_reference_csv", "") or "").strip()
    if not ref_path:
        print("[ModuleD] SBR skipped: --module_d_reference_csv is required.")
        return None

    try:
        reference_row = load_module_d_sbr_reference(
            ref_path,
            model_name=getattr(args, "model_name", ""),
            reference_name=getattr(args, "module_d_reference_name", ""),
        )
        adapted = dict(adapted_row) if adapted_row is not None else _load_default_adapted_row(output_dir)
        if adapted is None:
            print("[ModuleD] SBR skipped: no adapted eval row is available.")
            return None
        extra = {
            "adapter_target": str(getattr(args, "lora_target", "") or ""),
            "lora_base_update": str(getattr(args, "lora_base_update", "") or ""),
            "module_d_variant": meta.get("module_d_variant", ""),
            "module_d_attribution_note": meta.get("module_d_attribution_note", ""),
        }
        rows = module_d_sbr_rows(
            reference_row=reference_row,
            adapted_row=adapted,
            hard_k=int(getattr(args, "module_d_hard_k", 2)),
            nb_classes=int(getattr(args, "nb_classes", 0) or 0),
            model_name=str(getattr(args, "model_name", "") or ""),
            run_tag=str(getattr(args, "run_tag", "") or ""),
            reference_source=ref_path,
            reference_name=str(getattr(args, "module_d_reference_name", "") or ""),
            adapted_source=str(adapted.get("adapted_source", adapted.get("mode", "")) or ""),
            extra_metadata=extra,
        )
    except Exception as exc:
        print(f"[ModuleD] SBR skipped: {exc}")
        return None

    path_out = output_csv_path or os.path.join(output_dir, "diagnostics", MODULE_D_SBR_EVAL_FILE)
    path = write_module_d_sbr_eval(path_out, rows)
    if path:
        print(f"[ModuleD] SBR eval saved to: {path}")
    return path
