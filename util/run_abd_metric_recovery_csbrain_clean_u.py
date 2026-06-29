import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util import run_abd_metric_recovery as abd


MODEL = "CSBrain"
PREFIX = "cs"
CLEAN_U_RECIPE = "clean_ada_full_ft"


def _copy_file(src, dst):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    return True


def collect_clean_u_output(tag, col_dir):
    out_dir = abd.run_output_dir(MODEL, "full", tag)
    dst_root = abd.collected_output_dir(MODEL, tag, col_dir)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = []
    for name in ("log.txt", "args.json", "config.json"):
        if _copy_file(out_dir / name, dst_root / name):
            copied.append(name)

    diag_dir = out_dir / "diagnostics"
    if diag_dir.exists():
        for src in diag_dir.glob("*.csv"):
            rel = Path("diagnostics") / src.name
            if _copy_file(src, dst_root / rel):
                copied.append(str(rel).replace("/", "\\"))

    rows = [
        ("model", MODEL),
        ("tag", tag),
        ("seed", abd.SEED),
        ("output_dir", abd.rel(out_dir)),
        ("dataset", abd.DATASET),
        ("subject_mod", "multi"),
        ("k_shot", abd.KSHOT),
        ("epochs", abd.EPOCHS),
        ("finetune_mod", "full"),
        ("lora_target", "none"),
        ("lora_base_update", ""),
        ("fb_enable", 0),
        ("fb_probe", 0),
        ("fb_recipe", CLEAN_U_RECIPE),
        ("loss_type", "ce"),
        ("adaptive_swa_eval", 0),
        ("score_is_validation_only", 1),
        ("test_used_for_selection", 0),
    ]
    with open(dst_root / "run_info.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("key", "value"))
        writer.writerows(rows)

    print(f"[ABD-CS] collected clean U files to: {abd.rel(dst_root)} ({len(copied)} files)")
    return dst_root


def run_clean_u(tag, col_name, status_csv, col_dir, dry_run):
    print()
    print(f"[ABD-CS] Run {MODEL} phase=U subject=multi finetune=full tag={tag}")
    print("[ABD-CS] U protocol: clean Ada-style Full FT, no fb/probe, no LoRA, no Module A, CE loss.")
    row = {
        "model": MODEL,
        "phase": "U",
        "subject_mod": "multi",
        "finetune_mod": "full",
        "lora_target": "none",
        "fb_recipe": CLEAN_U_RECIPE,
        "run_tag": tag,
        "reference_csv": "",
    }

    if dry_run:
        command = clean_u_command(tag)
        print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))
        return 0

    abd.prepare_retry_artifacts(MODEL, "full", tag, col_dir)
    rc = subprocess.run(clean_u_command(tag), cwd=str(ROOT)).returncode
    if int(rc) != 0:
        return abd.fail(status_csv, row, col_dir, f"[ABD-CS][ERROR] {MODEL} phase=U failed.")

    collect_clean_u_output(tag, col_dir)
    abd.append_status(status_csv, {**row, "status": "done"})
    return 0


def clean_u_command(tag):
    return [
        str(abd.PYTHON),
        "run_finetuning.py",
        "--dataset",
        abd.DATASET,
        "--model_name",
        MODEL,
        "--task_mod",
        "Classification",
        "--subject_mod",
        "multi",
        "--finetune_mod",
        "full",
        "--k_shot",
        abd.KSHOT,
        "--epochs",
        abd.EPOCHS,
        "--batch_size",
        abd.BATCH_SIZE,
        "--lr",
        abd.LR,
        "--weight_decay",
        abd.WEIGHT_DECAY,
        "--num_workers",
        abd.num_workers_for_model(MODEL),
        "--loader_prefetch_factor",
        abd.LOADER_PREFETCH_FACTOR,
        "--seed",
        abd.SEED,
        "--loss_type",
        "ce",
        "--best_metric",
        "balanced_accuracy",
        "--short_output_tag_only",
        "--run_tag",
        tag,
        "--no_auto_resume",
    ]


def run_csbrain(col_dir, ref_dir, status_csv, dry_run):
    stamp = col_dir.name.replace("col_abd_metric_", "", 1)
    col_name = col_dir.name
    tag_u = f"{PREFIX}_u_full_{stamp}"
    tag_r = f"{PREFIX}_r_fsfull_{stamp}"
    tag_a = f"{PREFIX}_a_fsfull_{stamp}"
    tag_b = f"{PREFIX}_b_a_{stamp}"
    tag_d = f"{PREFIX}_d_a_{stamp}"
    ref_path = ref_dir / f"{PREFIX}_fewshot_full_ref.csv"

    steps = (
        ("U", tag_u, lambda: run_clean_u(tag_u, col_name, status_csv, col_dir, dry_run)),
        ("R", tag_r, lambda: abd.run_full(MODEL, "R", "fewshot", tag_r, "probe_only", False, col_name, status_csv, col_dir, dry_run)),
        ("REF", tag_r, lambda: abd.make_reference(MODEL, tag_r, ref_path, status_csv, col_dir, dry_run)),
        ("A", tag_a, lambda: abd.run_full(MODEL, "A", "fewshot", tag_a, "probe_only", True, col_name, status_csv, col_dir, dry_run)),
        ("B", tag_b, lambda: abd.run_lora(MODEL, "B", "fewshot", tag_b, "signal_align", "sig_align", None, col_name, status_csv, col_dir, dry_run)),
        ("D", tag_d, lambda: abd.run_lora(MODEL, "D", "fewshot", tag_d, "semantic", "sem_lif", ref_path, col_name, status_csv, col_dir, dry_run)),
    )

    for phase, tag, fn in steps:
        rc = abd.run_phase_if_needed(status_csv, MODEL, phase, tag, fn)
        if rc != 0:
            return rc
    return 0


def main():
    parser = argparse.ArgumentParser(description="Resume ABD metric recovery for CSBrain only, with a clean Ada-style U phase.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-stamp", "--resume", dest="resume_stamp", default="20260625_215700")
    parser.add_argument("--num-workers", default="2")
    parser.add_argument("--loader-prefetch-factor", default="1")
    args = parser.parse_args()

    abd.NUM_WORKERS = str(args.num_workers)
    abd.LOADER_PREFETCH_FACTOR = str(args.loader_prefetch_factor)

    stamp = args.resume_stamp.strip()
    col_dir = ROOT / f"col_abd_metric_{stamp}"
    ref_dir = col_dir / "references"
    status_csv = col_dir / "run_status.csv"

    print(f"[ABD-CS] Root: {ROOT}")
    print(f"[ABD-CS] Collection: {col_dir.name}")
    print("[ABD-CS] Protocol: CSBrain only. U is clean Ada-style Full FT; R/REF/A/B/D keep ABD metric protocol.")
    print(f"[ABD-CS] Dataset={abd.DATASET} k_shot={abd.KSHOT} epochs={abd.EPOCHS} seed={abd.SEED}")
    print(f"[ABD-CS] DataLoader: num_workers={abd.NUM_WORKERS} loader_prefetch_factor={abd.LOADER_PREFETCH_FACTOR}")
    if args.dry_run:
        print("[ABD-CS] DRY RUN: commands will not be launched.")
    print()

    if not col_dir.exists():
        print(f"[ABD-CS][ERROR] Cannot resume; collection folder not found: {col_dir.name}")
        return 1

    if not args.dry_run:
        ref_dir.mkdir(parents=True, exist_ok=True)
        abd.ensure_status_file(status_csv)

    rc = run_csbrain(col_dir, ref_dir, status_csv, args.dry_run)
    if rc != 0:
        return rc

    if args.dry_run:
        print()
        print("[ABD-CS] Dry run done. No training, status file, folders, or zip package were written.")
        return 0

    zip_path = abd.zip_collection(col_dir)
    print()
    print("[ABD-CS] Done.")
    print(f"[ABD-CS] Collected folder: {col_dir.name}")
    print(f"[ABD-CS] Zip package: {zip_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
