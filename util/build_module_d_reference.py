import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def _read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _class_columns(row):
    cols = []
    for key in row.keys():
        if not key.startswith("class_"):
            continue
        suffix = key[len("class_") :]
        if suffix.isdigit():
            cols.append((int(suffix), key))
    return [key for _, key in sorted(cols)]


def _row_for_epoch_and_split(rows, epoch, split):
    for row in reversed(rows):
        try:
            row_epoch = int(float(row.get("epoch", "")))
        except (TypeError, ValueError):
            continue
        if row_epoch == int(epoch) and str(row.get("split", "")).strip().lower() == split:
            return row
    return None


def _metric_row_for_epoch(rows, epoch):
    for row in reversed(rows):
        try:
            row_epoch = int(float(row.get("epoch", "")))
        except (TypeError, ValueError):
            continue
        if row_epoch == int(epoch):
            return row
    return None


def build_reference(
    run_dir,
    output_csv,
    model_name,
    reference_name,
    run_tag,
    epoch,
):
    run_dir = Path(run_dir)
    output_csv = Path(output_csv)
    per_class_path = run_dir / "diagnostics" / "per_class_recall.csv"
    metrics_path = run_dir / "diagnostics" / "epoch_metrics.csv"

    if not per_class_path.exists():
        raise FileNotFoundError(f"missing per_class_recall.csv: {per_class_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing epoch_metrics.csv: {metrics_path}")

    per_class_rows = _read_csv(per_class_path)
    metrics_rows = _read_csv(metrics_path)
    val_row = _row_for_epoch_and_split(per_class_rows, epoch, "val")
    test_row = _row_for_epoch_and_split(per_class_rows, epoch, "test")
    metrics_row = _metric_row_for_epoch(metrics_rows, epoch)

    if val_row is None or test_row is None:
        raise ValueError(f"missing final val/test per-class rows for epoch {epoch}")
    if metrics_row is None:
        raise ValueError(f"missing epoch_metrics row for epoch {epoch}")

    classes = _class_columns(val_row)
    if not classes:
        raise ValueError(f"no class_* columns found in {per_class_path}")

    out = OrderedDict()
    out["model_name"] = model_name
    out["reference_name"] = reference_name
    out["run_tag"] = run_tag
    out["source_output_dir"] = str(run_dir)
    out["val_balanced_accuracy"] = metrics_row.get("val_balanced_accuracy", "")
    out["test_balanced_accuracy"] = metrics_row.get("test_balanced_accuracy", "")

    for col in classes:
        out[f"val_{col}"] = val_row.get(col, "")
    for col in classes:
        out[f"test_{col}"] = test_row.get(col, "")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out.keys()))
        writer.writeheader()
        writer.writerow(out)
    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Build a Module D SBR reference row from a completed fewshot Full FT run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--reference-name", default="fewshot_full_final")
    args = parser.parse_args()

    path = build_reference(
        run_dir=args.run_dir,
        output_csv=args.output_csv,
        model_name=args.model_name,
        reference_name=args.reference_name,
        run_tag=args.run_tag,
        epoch=args.epoch,
    )
    print(f"[ABD] reference saved to: {path}")


if __name__ == "__main__":
    main()
