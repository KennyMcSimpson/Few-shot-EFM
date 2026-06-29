import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.build_module_d_reference import build_reference


PYTHON = Path(sys.executable)

DATASET = "TUEV"
KSHOT = "0.05"
EPOCHS = "30"
BATCH_SIZE = "16"
LR = "1e-4"
WEIGHT_DECAY = "0.05"
NUM_WORKERS = "4"
LOADER_PREFETCH_FACTOR = "1"
SEED = "0"
REFERENCE_NAME = "fewshot_full_final"
MODEL_NUM_WORKERS = {}

MODELS = (
    ("EEGPT", "eeg"),
    ("BIOT", "bi"),
    ("LaBraM", "la"),
    ("CBraMod", "cb"),
    ("Gram", "gr"),
    ("CSBrain", "cs"),
)

STATUS_FIELDS = (
    "model",
    "phase",
    "subject_mod",
    "finetune_mod",
    "lora_target",
    "fb_recipe",
    "run_tag",
    "reference_csv",
    "status",
)


def rel(path):
    return str(path).replace("/", "\\")


def run_output_dir(model, finetune_mod, tag):
    return ROOT / "finetuning_results" / "Classification" / f"{model}_results" / f"finetune_{finetune_mod}" / tag


def collected_output_dir(model, tag, col_dir):
    return col_dir / f"{model}_{tag}_s0"


def backup_existing_path(path):
    path = Path(path)
    if not path.exists():
        return None
    backup_path = path.with_name(f"{path.name}_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    while backup_path.exists():
        backup_path = backup_path.with_name(f"{backup_path.name}_again")
    shutil.move(str(path), str(backup_path))
    return backup_path


def prepare_retry_artifacts(model, finetune_mod, tag, col_dir):
    moved = []
    for path in (run_output_dir(model, finetune_mod, tag), collected_output_dir(model, tag, col_dir)):
        backup_path = backup_existing_path(path)
        if backup_path is not None:
            moved.append((path, backup_path))
    for old, new in moved:
        print(f"[ABD] Existing incomplete artifact moved aside: {rel(old)} -> {rel(new)}")


def model_extra_args(model):
    if model.lower() == "eegpt":
        return ["--sampling_rate", "256"]
    if model.lower() == "gram":
        return [
            "--gram_ckpt",
            r"checkpoints\base.pth",
            "--gram_vqgan_ckpt",
            r"checkpoints\base_class_quantization.pth",
            "--gram_root",
            r"external\Gram",
        ]
    return []


def adaptive_swa_args():
    return [
        "--adaptive_swa_eval",
        "--adaptive_swa_epoch_min",
        "1",
        "--adaptive_swa_epoch_max",
        EPOCHS,
        "--adaptive_swa_min_len",
        "3",
        "--adaptive_swa_max_len",
        "8",
        "--adaptive_swa_stride",
        "1",
        "--adaptive_swa_select_metric",
        "selection_bacc_min02_std",
        "--adaptive_swa_profile",
        "generic",
        "--adaptive_swa_balance_lambda",
        "0.10",
        "--adaptive_swa_hard_classes",
        "0,2",
        "--adaptive_swa_hard_floor",
        "0.05",
        "--adaptive_swa_hard_floor_lambda",
        "0.20",
        "--adaptive_swa_std_lambda",
        "0.04",
        "--adaptive_swa_tie_mode",
        "hard_stable",
        "--adaptive_swa_tie_eps",
        "0.002",
    ]


def num_workers_for_model(model):
    return str(MODEL_NUM_WORKERS.get(model, NUM_WORKERS))


def common_train_args(model, subject_mod, run_tag, fb_recipe, col_name):
    return [
        "--dataset",
        DATASET,
        "--model_name",
        model,
        "--task_mod",
        "Classification",
        "--subject_mod",
        subject_mod,
        "--k_shot",
        KSHOT,
        "--epochs",
        EPOCHS,
        "--batch_size",
        BATCH_SIZE,
        "--lr",
        LR,
        "--weight_decay",
        WEIGHT_DECAY,
        "--num_workers",
        num_workers_for_model(model),
        "--loader_prefetch_factor",
        LOADER_PREFETCH_FACTOR,
        "--seed",
        SEED,
        "--loss_type",
        "sqrt_balanced_ce",
        "--best_metric",
        "balanced_accuracy",
        "--selection_worst_alpha",
        "0.35",
        "--selection_min02_alpha",
        "0.40",
        "--selection_std_gamma",
        "0.16",
        "--monitor_dynamics",
        "--eval_train_set",
        "--diag_freq",
        "5",
        "--save_epoch_ckpt_freq",
        "999",
        "--short_output_tag_only",
        "--run_tag",
        run_tag,
        "--no_auto_resume",
        "--fb_enable",
        "--fb_probe",
        "--fb_recipe",
        fb_recipe,
        "--fb_split_check",
        "--fb_collect",
        "--fb_collect_name",
        col_name,
    ]


def append_status(status_csv, row):
    exists = status_csv.exists()
    with open(status_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def ensure_status_file(status_csv):
    if status_csv.exists():
        return
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(status_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=STATUS_FIELDS).writeheader()


def phase_done(status_csv, model, phase, run_tag):
    if not status_csv.exists():
        return False
    with open(status_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (
                row.get("model") == model
                and row.get("phase") == phase
                and row.get("run_tag") == run_tag
                and str(row.get("status", "")).lower() == "done"
            ):
                return True
    return False


def zip_collection(col_dir, suffix=""):
    zip_base = ROOT / f"{col_dir.name}{suffix}"
    zip_path = zip_base.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_base), "zip", root_dir=ROOT, base_dir=col_dir.name)
    return zip_path


def fail(status_csv, row, col_dir, message):
    append_status(status_csv, {**row, "status": "failed"})
    partial = zip_collection(col_dir, "_partial")
    print(message)
    print(f"[ABD][ERROR] Partial package: {partial.name}")
    return 1


def run_command(command, dry_run):
    if dry_run:
        print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))
        return 0
    completed = subprocess.run(command, cwd=str(ROOT))
    return int(completed.returncode)


def run_full(model, phase, subject_mod, tag, fb_recipe, use_a, col_name, status_csv, col_dir, dry_run):
    print()
    print(f"[ABD] Run {model} phase={phase} subject={subject_mod} finetune=full tag={tag}")
    row = {
        "model": model,
        "phase": phase,
        "subject_mod": subject_mod,
        "finetune_mod": "full",
        "lora_target": "none",
        "fb_recipe": fb_recipe,
        "run_tag": tag,
        "reference_csv": "",
    }
    if dry_run:
        return 0

    prepare_retry_artifacts(model, "full", tag, col_dir)

    command = [
        str(PYTHON),
        "run_finetuning.py",
        "--finetune_mod",
        "full",
    ]
    command.extend(common_train_args(model, subject_mod, tag, fb_recipe, col_name))
    if use_a:
        command.extend(adaptive_swa_args())
    command.extend(model_extra_args(model))

    rc = run_command(command, dry_run=False)
    if rc != 0:
        return fail(status_csv, row, col_dir, f"[ABD][ERROR] {model} phase={phase} failed.")
    append_status(status_csv, {**row, "status": "done"})
    return 0


def run_lora(
    model,
    phase,
    subject_mod,
    tag,
    lora_target,
    fb_recipe,
    ref_path,
    col_name,
    status_csv,
    col_dir,
    dry_run,
):
    print()
    print(f"[ABD] Run {model} phase={phase} subject={subject_mod} lora_target={lora_target} tag={tag}")
    if ref_path:
        print(f"[ABD] D reference: {rel(ref_path)}")
    row = {
        "model": model,
        "phase": phase,
        "subject_mod": subject_mod,
        "finetune_mod": "lora",
        "lora_target": lora_target,
        "fb_recipe": fb_recipe,
        "run_tag": tag,
        "reference_csv": rel(ref_path) if ref_path else "",
    }
    if dry_run:
        return 0

    prepare_retry_artifacts(model, "lora", tag, col_dir)

    command = [
        str(PYTHON),
        "run_finetuning.py",
        "--finetune_mod",
        "lora",
        "--lora_target",
        lora_target,
        "--lora_base_update",
        "full",
        "--lora_rank",
        "4",
        "--lora_alpha",
        "8",
        "--lora_dropout",
        "0.1",
    ]
    command.extend(common_train_args(model, subject_mod, tag, fb_recipe, col_name))
    command.extend(adaptive_swa_args())
    if phase == "D":
        command.extend(
            [
                "--module_d_sbr_eval",
                "--module_d_reference_csv",
                rel(ref_path),
                "--module_d_reference_name",
                REFERENCE_NAME,
                "--module_d_hard_k",
                "2",
            ]
        )
    command.extend(model_extra_args(model))

    rc = run_command(command, dry_run=False)
    if rc != 0:
        return fail(status_csv, row, col_dir, f"[ABD][ERROR] {model} phase={phase} failed.")
    append_status(status_csv, {**row, "status": "done"})
    return 0


def make_reference(model, tag, ref_path, status_csv, col_dir, dry_run):
    run_dir = Path("finetuning_results") / "Classification" / f"{model}_results" / "finetune_full" / tag
    print()
    print(f"[ABD] Build D reference for {model} from {rel(run_dir)}")
    row = {
        "model": model,
        "phase": "REF",
        "subject_mod": "fewshot",
        "finetune_mod": "full",
        "lora_target": "none",
        "fb_recipe": "probe_only",
        "run_tag": tag,
        "reference_csv": rel(ref_path),
    }
    if dry_run:
        return 0
    try:
        out = build_reference(
            run_dir=run_dir,
            output_csv=ref_path,
            model_name=model,
            reference_name=REFERENCE_NAME,
            run_tag=tag,
            epoch=int(EPOCHS),
        )
        print(f"[ABD] reference saved to: {rel(out)}")
    except Exception as exc:
        return fail(status_csv, row, col_dir, f"[ABD][ERROR] reference generation failed for {model}: {exc}")
    append_status(status_csv, {**row, "status": "done"})
    return 0


def run_phase_if_needed(status_csv, model, phase, tag, fn):
    if phase_done(status_csv, model, phase, tag):
        print(f"[ABD] Skip {model} phase={phase} tag={tag} already done.")
        return 0
    return fn()


def run_model(model, prefix, col_dir, ref_dir, status_csv, dry_run):
    stamp = col_dir.name.replace("col_abd_metric_", "", 1)
    col_name = col_dir.name
    tag_u = f"{prefix}_u_full_{stamp}"
    tag_r = f"{prefix}_r_fsfull_{stamp}"
    tag_a = f"{prefix}_a_fsfull_{stamp}"
    tag_b = f"{prefix}_b_a_{stamp}"
    tag_d = f"{prefix}_d_a_{stamp}"
    ref_path = ref_dir / f"{prefix}_fewshot_full_ref.csv"

    steps = (
        ("U", tag_u, lambda: run_full(model, "U", "multi", tag_u, "probe_only", False, col_name, status_csv, col_dir, dry_run)),
        ("R", tag_r, lambda: run_full(model, "R", "fewshot", tag_r, "probe_only", False, col_name, status_csv, col_dir, dry_run)),
        ("REF", tag_r, lambda: make_reference(model, tag_r, ref_path, status_csv, col_dir, dry_run)),
        ("A", tag_a, lambda: run_full(model, "A", "fewshot", tag_a, "probe_only", True, col_name, status_csv, col_dir, dry_run)),
        ("B", tag_b, lambda: run_lora(model, "B", "fewshot", tag_b, "signal_align", "sig_align", None, col_name, status_csv, col_dir, dry_run)),
        ("D", tag_d, lambda: run_lora(model, "D", "fewshot", tag_d, "semantic", "sem_lif", ref_path, col_name, status_csv, col_dir, dry_run)),
    )
    for phase, tag, fn in steps:
        rc = run_phase_if_needed(status_csv, model, phase, tag, fn)
        if rc != 0:
            return rc
    return 0


def main():
    global NUM_WORKERS, LOADER_PREFETCH_FACTOR

    parser = argparse.ArgumentParser(description="Run ABD metric recovery experiments.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-stamp", "--resume", dest="resume_stamp", default="")
    parser.add_argument("--num-workers", default=NUM_WORKERS)
    parser.add_argument("--loader-prefetch-factor", default=LOADER_PREFETCH_FACTOR)
    args = parser.parse_args()

    NUM_WORKERS = str(args.num_workers)
    LOADER_PREFETCH_FACTOR = str(args.loader_prefetch_factor)

    stamp = args.resume_stamp.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    col_dir = ROOT / f"col_abd_metric_{stamp}"
    ref_dir = col_dir / "references"
    status_csv = col_dir / "run_status.csv"

    print(f"[ABD] Root: {ROOT}")
    print(f"[ABD] Collection: {col_dir.name}")
    print("[ABD] Protocol: U=multi Full FT, R=fewshot Full FT, A=fewshot Full FT+A, B=fewshot Full FT+ModuleB+A, D=fewshot Full FT+ModuleD+A.")
    print(f"[ABD] Dataset={DATASET} k_shot={KSHOT} epochs={EPOCHS} seed={SEED}")
    overrides = ", ".join(f"{model}={workers}" for model, workers in MODEL_NUM_WORKERS.items())
    print(f"[ABD] DataLoader: default num_workers={NUM_WORKERS} loader_prefetch_factor={LOADER_PREFETCH_FACTOR}")
    if overrides:
        print(f"[ABD] DataLoader model overrides: {overrides}")
    if args.dry_run:
        print("[ABD] DRY RUN: commands will not be launched.")
    if args.resume_stamp:
        print("[ABD] RESUME: done phases in run_status.csv will be skipped.")
    print()

    if args.resume_stamp:
        if not col_dir.exists():
            print(f"[ABD][ERROR] Cannot resume; collection folder not found: {col_dir.name}")
            return 1
        if not args.dry_run:
            ref_dir.mkdir(parents=True, exist_ok=True)
    else:
        if col_dir.exists():
            print(f"[ABD][ERROR] Existing collection folder found: {col_dir.name}")
            return 1
        if col_dir.with_suffix(".zip").exists():
            print(f"[ABD][ERROR] Existing zip found: {col_dir.name}.zip")
            return 1
        if not args.dry_run:
            ref_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        ensure_status_file(status_csv)

    for model, prefix in MODELS:
        rc = run_model(model, prefix, col_dir, ref_dir, status_csv, args.dry_run)
        if rc != 0:
            return rc

    if args.dry_run:
        print()
        print("[ABD] Dry run done. No training, status file, folders, or zip package were written.")
        return 0

    zip_path = zip_collection(col_dir)
    print()
    print("[ABD] Done.")
    print(f"[ABD] Collected folder: {col_dir.name}")
    print(f"[ABD] Zip package: {zip_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
