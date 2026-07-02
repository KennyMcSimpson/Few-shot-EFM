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
    from module_d_semantic_refinement import (  # type: ignore
        load_module_d_sbr_reference,
        module_d_sbr_rows,
    )


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
