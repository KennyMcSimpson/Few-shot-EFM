"""Compute EEGFM compatibility and few-shot retention from result CSVs.

Expected input is one row per result with at least:
  dataset, model, ratio/k_shot, role, balanced_accuracy, num_classes

Recommended role values:
  fm        - pretrained EEGFM baseline at this label ratio
  random    - same architecture without pretraining at this label ratio
  module_b  - Module B result at this label ratio

The script writes chance-normalized scores, full-data pretraining gain,
few-shot compatibility retention, absolute accessibility retention, and
optional Module-B recovery.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def _first(row, names, default=""):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return row[name]
    return default


def _parse_float(value, default=math.nan):
    try:
        text = str(value).strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        out = float(text)
        return out / 100.0 if out > 1.0 else out
    except Exception:
        return default


def _parse_ratio(value):
    text = str(value).strip().lower()
    if text in {"full", "all", "100", "100%", "1", "1.0"}:
        return 1.0
    return _parse_float(text)


def _chance_normalized(bacc, num_classes):
    if not math.isfinite(bacc) or not math.isfinite(num_classes) or num_classes <= 1:
        return math.nan
    chance = 1.0 / float(num_classes)
    return (float(bacc) - chance) / max(1e-12, 1.0 - chance)


def _mean(values):
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else math.nan


def _safe_div(num, den):
    return float(num) / float(den) if math.isfinite(num) and math.isfinite(den) and abs(den) > 1e-12 else math.nan


def main():
    parser = argparse.ArgumentParser(description="Compute EEGFM compatibility retention.")
    parser.add_argument("--input", required=True, help="Input CSV with baseline/module results.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--dataset-col", default="dataset")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--role-col", default="role")
    parser.add_argument("--ratio-col", default="", help="Defaults to label_ratio/k_shot/shot_ratio if omitted.")
    parser.add_argument("--bacc-col", default="", help="Defaults to bacc/balanced_accuracy/test_balanced_accuracy.")
    parser.add_argument("--classes-col", default="", help="Defaults to num_classes/n_classes/classes.")
    parser.add_argument("--fm-role", default="fm")
    parser.add_argument("--random-role", default="random")
    parser.add_argument("--module-role", default="module_b")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    score_map = defaultdict(list)
    class_map = defaultdict(list)
    for row in rows:
        dataset = str(_first(row, [args.dataset_col, "dataset", "data"], "")).strip()
        model = str(_first(row, [args.model_col, "model", "backbone"], "")).strip()
        role = str(_first(row, [args.role_col, "role", "method"], "")).strip().lower()
        ratio_value = _first(row, [args.ratio_col] if args.ratio_col else ["label_ratio", "k_shot", "shot_ratio", "ratio"], "")
        bacc_value = _first(row, [args.bacc_col] if args.bacc_col else ["bacc", "balanced_accuracy", "test_balanced_accuracy"], "")
        class_value = _first(row, [args.classes_col] if args.classes_col else ["num_classes", "n_classes", "classes"], "")
        ratio = _parse_ratio(ratio_value)
        bacc = _parse_float(bacc_value)
        num_classes = _parse_float(class_value)
        if num_classes <= 1.0:
            try:
                num_classes = float(str(class_value).strip())
            except Exception:
                num_classes = math.nan
        norm = _chance_normalized(bacc, num_classes)
        if not dataset or not model or not role or not math.isfinite(ratio) or not math.isfinite(norm):
            continue
        key = (dataset, model, ratio, role)
        score_map[key].append(norm)
        class_map[(dataset, model)].append(num_classes)

    grouped_scores = {k: _mean(v) for k, v in score_map.items()}
    grouped_classes = {k: _mean(v) for k, v in class_map.items()}
    datasets = sorted(set((d, m) for d, m, _, _ in grouped_scores.keys()))
    output_rows = []
    for dataset, model in datasets:
        full_fm = grouped_scores.get((dataset, model, 1.0, args.fm_role.lower()), math.nan)
        full_random = grouped_scores.get((dataset, model, 1.0, args.random_role.lower()), math.nan)
        full_gain = full_fm - full_random if math.isfinite(full_fm) and math.isfinite(full_random) else math.nan
        ratios = sorted(set(r for d, m, r, _ in grouped_scores.keys() if d == dataset and m == model and r < 1.0))
        for ratio in ratios:
            fs_fm = grouped_scores.get((dataset, model, ratio, args.fm_role.lower()), math.nan)
            fs_random = grouped_scores.get((dataset, model, ratio, args.random_role.lower()), math.nan)
            fs_module = grouped_scores.get((dataset, model, ratio, args.module_role.lower()), math.nan)
            fs_gain = fs_fm - fs_random if math.isfinite(fs_fm) and math.isfinite(fs_random) else math.nan
            module_gain = fs_module - fs_random if math.isfinite(fs_module) and math.isfinite(fs_random) else math.nan
            fcr = _safe_div(fs_gain, full_gain)
            module_fcr = _safe_div(module_gain, full_gain)
            recovery = _safe_div(module_fcr - fcr, 1.0 - fcr)
            fs_absolute_retention = _safe_div(fs_fm, full_fm)
            module_absolute_retention = _safe_div(fs_module, full_fm)
            module_absolute_recovery = _safe_div(module_absolute_retention - fs_absolute_retention, 1.0 - fs_absolute_retention)
            output_rows.append({
                "dataset": dataset,
                "model": model,
                "num_classes": grouped_classes.get((dataset, model), ""),
                "label_ratio": ratio,
                "full_pretraining_gain": full_gain,
                "fewshot_pretraining_gain": fs_gain,
                "fewshot_compatibility_retention": fcr,
                "fewshot_absolute_accessibility_retention": fs_absolute_retention,
                "module_b_gain": module_gain,
                "module_b_compatibility_retention": module_fcr,
                "module_b_recovery_of_lost_retention": recovery,
                "module_b_absolute_accessibility_retention": module_absolute_retention,
                "module_b_absolute_recovery_of_lost_accessibility": module_absolute_recovery,
                "fm_norm_score_full": full_fm,
                "random_norm_score_full": full_random,
                "fm_norm_score_fewshot": fs_fm,
                "random_norm_score_fewshot": fs_random,
                "module_b_norm_score_fewshot": fs_module,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(output_rows[0].keys()) if output_rows else [
            "dataset", "model", "num_classes", "label_ratio", "full_pretraining_gain",
            "fewshot_pretraining_gain", "fewshot_compatibility_retention",
            "fewshot_absolute_accessibility_retention",
            "module_b_gain", "module_b_compatibility_retention",
            "module_b_recovery_of_lost_retention",
            "module_b_absolute_accessibility_retention",
            "module_b_absolute_recovery_of_lost_accessibility",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
