from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from .module_d_semantic_refinement import (
        load_module_d_sbr_reference,
        module_d_sbr_rows,
    )
except ImportError:  # pragma: no cover - allows direct script execution.
    try:
        from module_d_semantic_refinement import (  # type: ignore
            load_module_d_sbr_reference,
            module_d_sbr_rows,
        )
    except ImportError:  # pragma: no cover - supports pure CSV use without torch.
        load_module_d_sbr_reference = None
        module_d_sbr_rows = None


SUMMARY_FILE_NAME = "module_d_semantic_boundary_summary.csv"


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _ratio(numerator: Optional[float], denominator: Optional[float]) -> Any:
    if numerator is None or denominator is None or denominator <= 0:
        return ""
    return float(numerator / denominator)


def _clean(value: Any) -> Any:
    out = _safe_float(value)
    if out is None:
        return "" if value is None else value
    return float(out)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _row_matches_optional(row: Mapping[str, Any], keys: Tuple[str, ...], expected: Any) -> bool:
    expected_norm = _norm_text(expected)
    if not expected_norm:
        return True
    seen = False
    for key in keys:
        value_norm = _norm_text(row.get(key, ""))
        if not value_norm:
            continue
        seen = True
        if value_norm == expected_norm:
            return True
    return not seen


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    key_norm = key.lower()
    for raw_key, value in row.items():
        if str(raw_key).lower() == key_norm:
            return value
    return None


def _fallback_load_module_d_sbr_reference(csv_path: str, model_name: Optional[Any] = None, reference_name: Optional[Any] = None) -> Dict[str, str]:
    rows = _read_csv_rows(str(csv_path or ""))
    if not rows:
        raise FileNotFoundError(f"Module D SBR reference CSV has no rows: {csv_path}")
    model_keys = ("model_name", "model", "foundation_model", "fm")
    name_keys = ("reference_name", "name", "method", "run_tag", "tag", "adapter_target", "fb_recipe", "mode")
    candidates = [
        row for row in rows
        if _row_matches_optional(row, model_keys, model_name)
        and _row_matches_optional(row, name_keys, reference_name)
    ]
    if not candidates:
        raise ValueError(f"No Module D SBR reference row matched model={model_name!r}, reference_name={reference_name!r} in {csv_path}")
    return dict(candidates[-1])


def _extract_split_recall(row: Mapping[str, Any], split: str, nb_classes: Optional[int] = None) -> List[Any]:
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
    count = max(values) + 1 if nb_classes is None or int(nb_classes) <= 0 else int(nb_classes)
    return [values.get(i, float("nan")) for i in range(count)]


def _extract_split_bacc(row: Mapping[str, Any], split: str) -> Optional[Any]:
    split_norm = str(split or "").strip().lower()
    keys: List[str] = []
    if split_norm:
        keys.extend((f"{split_norm}_balanced_accuracy", f"{split_norm}_balanced_acc", f"{split_norm}_bacc"))
    if not split_norm or _norm_text(row.get("split", "")) == split_norm:
        keys.extend(("balanced_accuracy", "balanced_acc", "bacc"))
    for key in keys:
        value = _row_value(row, key)
        if _safe_float(value) is not None:
            return value
    return None


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _fallback_sbr_result(reference_recall: List[Any], adapted_recall: List[Any], hard_k: int = 2, hard_classes: Optional[Tuple[int, ...]] = None, reference_bacc: Optional[Any] = None, adapted_bacc: Optional[Any] = None) -> Dict[str, Any]:
    if len(reference_recall) != len(adapted_recall) or not reference_recall:
        raise ValueError("SBR needs matching non-empty recall vectors.")
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
        raise ValueError("SBR needs at least one finite recall pair.")
    if hard_classes is None:
        hard = tuple(sorted(valid, key=lambda idx: (float(reference_recall[idx]), idx))[:max(1, int(hard_k))])
    else:
        hard = tuple(idx for idx in hard_classes if idx in valid)
    stable = tuple(idx for idx in valid if idx not in set(hard))
    hard_gain = _mean(deltas[idx] for idx in hard)
    stable_loss = _mean(max(0.0, -deltas[idx]) for idx in stable)
    ref_bacc = _safe_float(reference_bacc)
    d_bacc = _safe_float(adapted_bacc)
    return {
        "hard_classes": ",".join(str(x) for x in hard),
        "stable_classes": ",".join(str(x) for x in stable),
        "hard_gain": float(hard_gain),
        "stable_loss": float(stable_loss),
        "sbr": float(hard_gain - stable_loss),
        "reference_bacc": "" if ref_bacc is None else ref_bacc,
        "adapted_bacc": "" if d_bacc is None else d_bacc,
        "bacc_delta": "" if ref_bacc is None or d_bacc is None else float(d_bacc - ref_bacc),
    }


def _fallback_module_d_sbr_rows(reference_row: Mapping[str, Any], adapted_row: Mapping[str, Any], hard_k: int = 2, nb_classes: Optional[int] = None, model_name: str = "", run_tag: str = "", reference_source: str = "", reference_name: str = "", adapted_source: str = "", extra_metadata: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], ...]:
    ref_val = _extract_split_recall(reference_row, "val", nb_classes=nb_classes)
    adapted_val = _extract_split_recall(adapted_row, "val", nb_classes=nb_classes)
    val_result = _fallback_sbr_result(ref_val, adapted_val, hard_k=hard_k, reference_bacc=_extract_split_bacc(reference_row, "val"), adapted_bacc=_extract_split_bacc(adapted_row, "val"))
    metadata = {
        "model_name": str(model_name or adapted_row.get("model_name", reference_row.get("model_name", ""))),
        "run_tag": str(run_tag or adapted_row.get("run_tag", adapted_row.get("tag", ""))),
        "reference_source": str(reference_source or ""),
        "reference_name": str(reference_name or reference_row.get("reference_name", "")),
        "adapted_source": str(adapted_source or ""),
        "score_is_validation_only": 1,
        "test_used_for_hard_class_selection": 0,
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    rows = [dict(metadata, split="val", **val_result)]
    ref_test = _extract_split_recall(reference_row, "test", nb_classes=nb_classes)
    adapted_test = _extract_split_recall(adapted_row, "test", nb_classes=nb_classes)
    if ref_test and adapted_test:
        hard_classes = tuple(int(x) for x in str(val_result["hard_classes"]).split(",") if str(x).strip())
        test_result = _fallback_sbr_result(ref_test, adapted_test, hard_k=hard_k, hard_classes=hard_classes, reference_bacc=_extract_split_bacc(reference_row, "test"), adapted_bacc=_extract_split_bacc(adapted_row, "test"))
        rows.append(dict(metadata, split="test", **test_result))
    return tuple(rows)


if load_module_d_sbr_reference is None:
    load_module_d_sbr_reference = _fallback_load_module_d_sbr_reference
if module_d_sbr_rows is None:
    module_d_sbr_rows = _fallback_module_d_sbr_rows


def _looks_like_sbr_eval(rows: Iterable[Mapping[str, Any]]) -> bool:
    for row in rows:
        if "split" in row and ("sbr" in row or "semantic_boundary_refinement_score" in row):
            return True
    return False


def _split_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        split = str(row.get("split", "") or "").strip().lower()
        if split:
            out[split] = dict(row)
    return out


def _load_named_row(csv_path: str, model_name: str = "", row_name: str = "") -> Dict[str, str]:
    return load_module_d_sbr_reference(
        csv_path,
        model_name=model_name,
        reference_name=row_name,
    )


def _computed_sbr_rows(
    reference_row: Mapping[str, Any],
    adapted_row: Mapping[str, Any],
    hard_k: int,
    nb_classes: Optional[int],
    model_name: str,
    reference_source: str,
    reference_name: str,
    adapted_source: str,
    adapted_name: str,
) -> Dict[str, Dict[str, Any]]:
    rows = module_d_sbr_rows(
        reference_row=reference_row,
        adapted_row=adapted_row,
        hard_k=hard_k,
        nb_classes=nb_classes,
        model_name=model_name,
        reference_source=reference_source,
        reference_name=reference_name,
        adapted_source=adapted_source,
        extra_metadata={
            "boundary_pair": adapted_name,
        },
    )
    return _split_rows(rows)


def _load_or_compute_effect_rows(
    reference_row: Mapping[str, Any],
    adapted_csv: str,
    model_name: str,
    adapted_name: str,
    reference_source: str,
    reference_name: str,
    hard_k: int,
    nb_classes: Optional[int],
) -> Dict[str, Dict[str, Any]]:
    rows = _read_csv_rows(adapted_csv)
    if _looks_like_sbr_eval(rows):
        return _split_rows(rows)
    adapted_row = _load_named_row(adapted_csv, model_name=model_name, row_name=adapted_name)
    return _computed_sbr_rows(
        reference_row=reference_row,
        adapted_row=adapted_row,
        hard_k=hard_k,
        nb_classes=nb_classes,
        model_name=model_name,
        reference_source=reference_source,
        reference_name=reference_name,
        adapted_source=adapted_csv,
        adapted_name=adapted_name or "module_d_adapted",
    )


def _interpret(need_sbr: Optional[float], d_sbr: Optional[float], d_hard_gain: Optional[float], d_stable_loss: Optional[float]) -> str:
    if need_sbr is None or need_sbr <= 0:
        return "no_positive_semantic_need"
    if d_sbr is None:
        return "missing_d_effect"
    if d_sbr <= 0:
        if d_hard_gain is not None and d_hard_gain > 0 and d_stable_loss is not None and d_stable_loss >= d_hard_gain:
            return "hard_gain_erased_by_stable_loss"
        return "no_semantic_boundary_recovery"
    if d_stable_loss is not None and d_stable_loss > 0:
        return "partial_recovery_with_stable_tradeoff"
    return "clean_hard_class_recovery"


def _summary_row(
    split: str,
    need_row: Mapping[str, Any],
    effect_row: Mapping[str, Any],
    reference_csv: str,
    capability_csv: str,
    adapted_csv: str,
    reference_name: str,
    capability_name: str,
    adapted_name: str,
) -> Dict[str, Any]:
    need_sbr = _safe_float(need_row.get("sbr"))
    d_sbr = _safe_float(effect_row.get("sbr"))
    d_hard_gain = _safe_float(effect_row.get("hard_gain"))
    d_stable_loss = _safe_float(effect_row.get("stable_loss"))
    return {
        "model_name": need_row.get("model_name", effect_row.get("model_name", "")),
        "split": split,
        "hard_class_source_split": "val",
        "hard_classes": need_row.get("hard_classes", effect_row.get("hard_classes", "")),
        "stable_classes": need_row.get("stable_classes", effect_row.get("stable_classes", "")),
        "need_sbr": _clean(need_sbr),
        "need_hard_gain": _clean(need_row.get("hard_gain")),
        "need_stable_loss": _clean(need_row.get("stable_loss")),
        "d_effect_sbr": _clean(d_sbr),
        "d_hard_gain": _clean(d_hard_gain),
        "d_stable_loss": _clean(d_stable_loss),
        "recovery_ratio": _clean(_ratio(d_sbr, need_sbr)),
        "reference_bacc": _clean(need_row.get("reference_bacc")),
        "capability_bacc": _clean(need_row.get("adapted_bacc")),
        "adapted_bacc": _clean(effect_row.get("adapted_bacc")),
        "capability_bacc_delta": _clean(need_row.get("bacc_delta")),
        "adapted_bacc_delta": _clean(effect_row.get("bacc_delta")),
        "reference_name": reference_name,
        "capability_name": capability_name,
        "adapted_name": adapted_name,
        "reference_source": reference_csv,
        "capability_source": capability_csv,
        "adapted_source": adapted_csv,
        "test_used_for_hard_class_selection": 0,
        "interpretation": _interpret(need_sbr, d_sbr, d_hard_gain, d_stable_loss),
    }


def build_semantic_boundary_summary(
    reference_csv: str,
    capability_csv: str,
    adapted_csv: str,
    output_csv: str,
    model_name: str = "",
    reference_name: str = "",
    capability_name: str = "",
    adapted_name: str = "",
    hard_k: int = 2,
    nb_classes: Optional[int] = None,
) -> Tuple[Dict[str, Any], ...]:
    reference_row = _load_named_row(reference_csv, model_name=model_name, row_name=reference_name)
    capability_row = _load_named_row(capability_csv, model_name=model_name, row_name=capability_name)
    need_rows = _computed_sbr_rows(
        reference_row=reference_row,
        adapted_row=capability_row,
        hard_k=hard_k,
        nb_classes=nb_classes,
        model_name=model_name,
        reference_source=reference_csv,
        reference_name=reference_name,
        adapted_source=capability_csv,
        adapted_name=capability_name or "capability_anchor",
    )
    effect_rows = _load_or_compute_effect_rows(
        reference_row=reference_row,
        adapted_csv=adapted_csv,
        model_name=model_name,
        adapted_name=adapted_name,
        reference_source=reference_csv,
        reference_name=reference_name,
        hard_k=hard_k,
        nb_classes=nb_classes,
    )

    rows: List[Dict[str, Any]] = []
    for split in ("val", "test"):
        if split not in need_rows or split not in effect_rows:
            continue
        rows.append(
            _summary_row(
                split=split,
                need_row=need_rows[split],
                effect_row=effect_rows[split],
                reference_csv=reference_csv,
                capability_csv=capability_csv,
                adapted_csv=adapted_csv,
                reference_name=reference_name,
                capability_name=capability_name,
                adapted_name=adapted_name,
            )
        )
    if not rows:
        raise ValueError("No shared val/test SBR rows were available for semantic-boundary summary.")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Module D semantic-boundary diagnostic card from "
            "R=few-shot reference, U=capability anchor, and D=semantic-LoRA outputs."
        )
    )
    parser.add_argument("--reference-csv", required=True, help="R: few-shot baseline/reference CSV.")
    parser.add_argument("--capability-csv", required=True, help="U: full-data or non-few-shot capability anchor CSV.")
    parser.add_argument("--adapted-csv", required=True, help="D: semantic-LoRA eval CSV or module_d_sbr_eval.csv.")
    parser.add_argument("--output-csv", default=SUMMARY_FILE_NAME)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--reference-name", default="")
    parser.add_argument("--capability-name", default="")
    parser.add_argument("--adapted-name", default="")
    parser.add_argument("--hard-k", type=int, default=2)
    parser.add_argument("--nb-classes", type=int, default=0)
    args = parser.parse_args()

    rows = build_semantic_boundary_summary(
        reference_csv=args.reference_csv,
        capability_csv=args.capability_csv,
        adapted_csv=args.adapted_csv,
        output_csv=args.output_csv,
        model_name=args.model_name,
        reference_name=args.reference_name,
        capability_name=args.capability_name,
        adapted_name=args.adapted_name,
        hard_k=args.hard_k,
        nb_classes=args.nb_classes or None,
    )
    print(f"[ModuleD] semantic-boundary summary saved to: {args.output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
