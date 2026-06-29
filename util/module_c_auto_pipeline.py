from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .module_c_lora_search import (
    DEFAULT_CANDIDATE_MODULES,
    ModuleCDecision,
    ModuleCPolicyConfig,
    is_module_c_baseline_candidate,
    parse_module_ids,
    select_from_module_diagnostics,
)


DEFAULT_MODELS: Tuple[str, ...] = ("EEGPT", "BIOT", "LaBraM", "CBraMod", "Gram", "CSBrain")
MODEL_TAG_PREFIX = {
    "EEGPT": "eeg",
    "BIOT": "bi",
    "LaBraM": "la",
    "CBraMod": "cb",
    "Gram": "gr",
    "CSBrain": "cs",
}
MODEL_EXTRA_ARGS = {
    "EEGPT": ("--sampling_rate", "256"),
    "Gram": (
        "--gram_ckpt",
        "checkpoints\\base.pth",
        "--gram_vqgan_ckpt",
        "checkpoints\\base_class_quantization.pth",
        "--gram_root",
        "external\\Gram",
    ),
}
MODULE_TAG_SUFFIX = {"B": "b", "D": "d", "E": "e"}


@dataclass(frozen=True)
class AutoRunRecord:
    model_name: str
    module_id: str
    run_tag: str
    command: Tuple[str, ...]
    collected_dir: Path


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _clip01(value: Any, default: float = 0.0) -> float:
    out = _safe_float(value)
    if out is None:
        return float(default)
    return float(max(0.0, min(1.0, out)))


def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _last_row(path: Path) -> Dict[str, str]:
    rows = _read_rows(path)
    return dict(rows[-1]) if rows else {}


def _last_row_matching(path: Path, key: str, value: Any) -> Dict[str, str]:
    rows = _read_rows(path)
    if not rows:
        return {}
    expected = str(value or "").strip().lower()
    for row in reversed(rows):
        if str(row.get(key, "") or "").strip().lower() == expected:
            return dict(row)
    return dict(rows[-1])


def _first_number(row: Mapping[str, Any], *keys: str) -> Optional[float]:
    lower = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        out = _safe_float(lower.get(key.lower()))
        if out is not None:
            return out
    return None


def _best_epoch_row(path: Path) -> Dict[str, str]:
    rows = _read_rows(path)
    if not rows:
        return {}

    def score(row: Mapping[str, Any]) -> float:
        return _first_number(
            row,
            "val_selection_bacc_min02_std",
            "val_balanced_accuracy",
            "balanced_accuracy",
        ) or -1.0

    return dict(sorted(rows, key=score)[-1])


def _selection_min02(row: Mapping[str, Any]) -> Optional[float]:
    direct = _first_number(row, "val_selection_min02", "selection_min02")
    if direct is not None:
        return direct
    c0 = _first_number(row, "val_class_0", "val_class0", "val_selection_class0")
    c2 = _first_number(row, "val_class_2", "val_class2", "val_selection_class2")
    vals = [v for v in (c0, c2) if v is not None]
    return min(vals) if vals else None


def _merge_csv_numbers(diagnostics: Dict[str, Any], path: Path, keys: Sequence[str]) -> None:
    row = _last_row(path)
    for key in keys:
        value = _first_number(row, key)
        if value is not None:
            diagnostics[key] = value


def _structural_pressure_from_block_delta(path: Path) -> Optional[float]:
    rows = _read_rows(path)
    if not rows:
        return None
    epochs = [_safe_float(row.get("epoch")) for row in rows]
    epochs = [e for e in epochs if e is not None]
    if not epochs:
        return None
    latest = max(epochs)
    structural = 0.0
    total = 0.0
    for row in rows:
        epoch = _safe_float(row.get("epoch"))
        if epoch != latest:
            continue
        delta = _safe_float(row.get("delta_norm_l2")) or 0.0
        energy = delta * delta
        total += energy
        if str(row.get("block", "")).lower() in {"spatial", "temporal", "mixing"}:
            structural += energy
    return float(structural / total) if total > 0.0 else None


def extract_module_c_diagnostics(run_dir: os.PathLike[str] | str, module_id: Any) -> Dict[str, Any]:
    """Extract validation-side diagnostics for the original Module C selector.

    This function does not select modules. It only converts existing diagnostic
    CSVs into the generic fields already accepted by ``module_c_lora_search``.
    """
    module = parse_module_ids([module_id])
    normalized = module[0] if module else str(module_id or "").strip().upper()
    run_path = Path(run_dir)
    diag_dir = run_path / "diagnostics"
    epoch_row = _best_epoch_row(diag_dir / "epoch_metrics.csv")
    final_row = _last_row(diag_dir / "adaptive_swa_eval.csv") or epoch_row

    val_bacc = _first_number(final_row, "val_balanced_accuracy", "balanced_accuracy")
    worst = _first_number(final_row, "val_worst_class_recall", "worst_class_recall")
    min02 = _selection_min02(final_row)
    recall_std = _first_number(final_row, "val_recall_std", "recall_std")
    train_bacc = _first_number(epoch_row, "train_eval_balanced_accuracy")

    diagnostics: Dict[str, Any] = {
        "module_id": normalized,
        "source_run_dir": str(run_path),
        "pressure": _clip01(val_bacc, default=0.0),
        "hard_class_leverage": _clip01(max([v for v in (worst, min02) if v is not None], default=0.0)),
        "stability": max(0.0, 1.0 - _clip01(recall_std, default=0.0)),
    }
    if train_bacc is not None and val_bacc is not None:
        diagnostics["overfit_risk"] = max(0.0, float(train_bacc) - float(val_bacc))

    if normalized == "B":
        probe_row = _last_row(diag_dir / "signal_alignment_probe.csv")
        ratio = _first_number(probe_row, "delta_input_ratio_mean")
        ratio_max = _first_number(probe_row, "delta_input_ratio_max")
        if ratio is not None:
            diagnostics["input_front_gain"] = ratio
            diagnostics["low_rank_fit"] = _clip01(ratio / 0.05)
            diagnostics["pressure"] = max(diagnostics["pressure"], _clip01(ratio / 0.05))
        if ratio_max is not None:
            diagnostics["module_b_probe_delta_ratio_max"] = ratio_max
    elif normalized == "D":
        sbr_row = _last_row_matching(diag_dir / "module_d_sbr_eval.csv", "split", "val")
        for key in ("sbr", "hard_gain", "stable_loss", "bacc_delta"):
            value = _first_number(sbr_row, key)
            if value is not None:
                diagnostics[key] = value
        if min02 is not None and "hard_gain" not in diagnostics:
            diagnostics["hard_gain"] = min02
        if recall_std is not None:
            diagnostics["class_conflict"] = _clip01(recall_std)
    elif normalized == "E":
        _merge_csv_numbers(
            diagnostics,
            diag_dir / "module_e_coverage_audit.csv",
            ("esc", "pressure_weighted_esc"),
        )
        srp = _structural_pressure_from_block_delta(diag_dir / "fb_block_delta_summary.csv")
        if srp is not None:
            diagnostics["srp"] = srp
        if min02 is not None:
            diagnostics["delta_worst_recall"] = min02

    return diagnostics


def select_module_c_from_run_dirs(
    run_dirs_by_module: Mapping[Any, os.PathLike[str] | str],
    interaction_scores: Optional[Mapping[Tuple[Any, Any], float]] = None,
    config: Optional[ModuleCPolicyConfig] = None,
) -> ModuleCDecision:
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for raw_module, run_dir in run_dirs_by_module.items():
        module_id = str(raw_module or "").strip().upper()
        if not module_id or is_module_c_baseline_candidate(module_id):
            continue
        diagnostics[module_id] = extract_module_c_diagnostics(run_dir, module_id)
    return select_from_module_diagnostics(
        diagnostics,
        interaction_scores=interaction_scores,
        config=config,
        registry=DEFAULT_CANDIDATE_MODULES,
    )


def selected_blocks(selected_modules: Sequence[Any]) -> Tuple[str, ...]:
    blocks: List[str] = []
    for module_id in parse_module_ids(selected_modules):
        for block in DEFAULT_CANDIDATE_MODULES.get(module_id, {}).get("blocks", ()):
            if block not in blocks:
                blocks.append(str(block))
    return tuple(blocks)


def common_training_args(
    collection_name: str,
    epochs: int = 30,
    seed: int = 0,
    batch_size: int = 16,
    lr: str = "1e-4",
) -> List[str]:
    return [
        "run_finetuning.py",
        "--dataset",
        "TUEV",
        "--task_mod",
        "Classification",
        "--subject_mod",
        "fewshot",
        "--finetune_mod",
        "lora",
        "--k_shot",
        "0.05",
        "--epochs",
        str(int(epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--lr",
        str(lr),
        "--weight_decay",
        "0.05",
        "--num_workers",
        "4",
        "--seed",
        str(int(seed)),
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
        "--lora_base_update",
        "full",
        "--lora_rank",
        "4",
        "--lora_alpha",
        "8",
        "--lora_dropout",
        "0.1",
        "--monitor_dynamics",
        "--eval_train_set",
        "--diag_freq",
        "5",
        "--save_epoch_ckpt_freq",
        "999",
        "--adaptive_swa_eval",
        "--adaptive_swa_epoch_min",
        "1",
        "--adaptive_swa_epoch_max",
        str(int(epochs)),
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
        "--short_output_tag_only",
        "--no_auto_resume",
        "--fb_enable",
        "--fb_probe",
        "--fb_split_check",
        "--fb_collect",
        "--fb_collect_name",
        str(collection_name),
    ]


def build_candidate_training_command(
    python_executable: str,
    model_name: str,
    module_id: str,
    run_tag: str,
    collection_name: str,
    epochs: int = 30,
    seed: int = 0,
    extra_args: Sequence[str] = (),
) -> List[str]:
    module_key = parse_module_ids([module_id])[0]
    meta = DEFAULT_CANDIDATE_MODULES[module_key]
    return [
        python_executable,
        *common_training_args(collection_name, epochs=epochs, seed=seed),
        "--model_name",
        model_name,
        "--lora_target",
        str(meta["lora_target"]),
        "--fb_recipe",
        str(meta["fb_recipe"]),
        "--run_tag",
        run_tag,
        *extra_args,
    ]


def build_final_training_command(
    python_executable: str,
    model_name: str,
    run_tag: str,
    collection_name: str,
    selected_modules: Sequence[Any],
    epochs: int = 30,
    seed: int = 0,
    extra_args: Sequence[str] = (),
) -> List[str]:
    selected = ",".join(parse_module_ids(selected_modules))
    blocks = ",".join(selected_blocks(selected_modules))
    return [
        python_executable,
        *common_training_args(collection_name, epochs=epochs, seed=seed),
        "--model_name",
        model_name,
        "--lora_target",
        "module_c",
        "--fb_recipe",
        "manual",
        "--fb_blocks",
        blocks,
        "--module_c_enable",
        "--module_c_candidates",
        ",".join(DEFAULT_CANDIDATE_MODULES.keys()),
        "--module_c_selected",
        selected,
        "--run_tag",
        run_tag,
        *extra_args,
    ]


def model_extra_args(model_name: str) -> Tuple[str, ...]:
    return tuple(MODEL_EXTRA_ARGS.get(str(model_name), ()))


def collected_run_dir(collection_name: str, model_name: str, run_tag: str, seed: int) -> Path:
    return Path(collection_name) / f"{model_name}_{run_tag}_s{int(seed)}"


def quote_command(command: Sequence[str]) -> str:
    parts = []
    for item in command:
        text = str(item)
        if not text or any(ch.isspace() for ch in text):
            parts.append('"' + text.replace('"', '\\"') + '"')
        else:
            parts.append(text)
    return " ".join(parts)


def run_subprocess(command: Sequence[str]) -> int:
    completed = subprocess.run(list(command), check=False)
    return int(completed.returncode)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _decision_to_dict(decision: ModuleCDecision) -> Dict[str, Any]:
    return {
        "selected_modules": list(decision.selected_modules),
        "selected_score": decision.selected_score,
        "module_scores": decision.module_scores,
        "subset_scores": {"+".join(k) if k else "": v for k, v in decision.subset_scores.items()},
        "reason": decision.reason,
        "recipe": decision.recipe,
    }


def auto_collection_name() -> str:
    return "col_module_c_auto_" + datetime.now().strftime("%Y%m%d_%H%M")


def safe_tag_suffix(collection_name: str) -> str:
    text = Path(str(collection_name)).name
    for prefix in ("col_module_c_auto_", "col_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text).strip("_-")
    return safe or datetime.now().strftime("%Y%m%d_%H%M")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run B/D/E diagnostics, let Module C select modules, then run C-selected final training.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--collection", default="")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Write commands/decisions without launching training.")
    parser.add_argument("--skip-candidates", action="store_true", help="Reuse existing collected candidate folders.")
    parser.add_argument("--skip-final", action="store_true", help="Stop after Module C decisions.")
    args = parser.parse_args(argv)

    collection = args.collection or auto_collection_name()
    collection_path = Path(collection)
    collection_path.mkdir(parents=True, exist_ok=True)
    models = [m for m in (x.strip() for x in args.models.split(",")) if m]
    tag_suffix = safe_tag_suffix(collection)

    records: List[AutoRunRecord] = []
    command_lines: List[str] = []
    status_path = collection_path / "module_c_auto_status.csv"
    with status_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "model", "module", "tag", "status"])
        writer.writeheader()

    for model in models:
        prefix = MODEL_TAG_PREFIX.get(model, model.lower())
        for module_id in parse_module_ids(DEFAULT_CANDIDATE_MODULES.keys()):
            suffix = MODULE_TAG_SUFFIX.get(module_id, module_id.lower())
            tag = f"{prefix}_{suffix}_cauto_{tag_suffix}"
            cmd = build_candidate_training_command(
                args.python,
                model,
                module_id,
                tag,
                collection,
                epochs=args.epochs,
                seed=args.seed,
                extra_args=model_extra_args(model),
            )
            records.append(AutoRunRecord(model, module_id, tag, tuple(cmd), collected_run_dir(collection, model, tag, args.seed)))
            command_lines.append(quote_command(cmd))
            if not args.skip_candidates and not args.dry_run:
                code = run_subprocess(cmd)
                with status_path.open("a", encoding="utf-8", newline="") as f:
                    csv.DictWriter(f, fieldnames=["stage", "model", "module", "tag", "status"]).writerow(
                        {"stage": "candidate", "model": model, "module": module_id, "tag": tag, "status": "done" if code == 0 else f"failed:{code}"}
                    )
                if code != 0:
                    return code

    decisions: Dict[str, Any] = {}
    final_commands: List[str] = []
    for model in models:
        model_records = [r for r in records if r.model_name == model]
        run_dirs = {r.module_id: r.collected_dir for r in model_records}
        missing = [str(path) for path in run_dirs.values() if not Path(path).exists()]
        if missing and not args.dry_run:
            raise FileNotFoundError(f"Missing candidate collected folders for {model}: {missing}")
        if missing and args.dry_run:
            continue
        diagnostics = {m: extract_module_c_diagnostics(path, m) for m, path in run_dirs.items()}
        decision = select_from_module_diagnostics(diagnostics, config=ModuleCPolicyConfig(marginal_margin=0.03, min_module_score=0.0))
        decisions[model] = {
            "diagnostics": diagnostics,
            "decision": _decision_to_dict(decision),
        }
        if not decision.selected_modules:
            continue
        prefix = MODEL_TAG_PREFIX.get(model, model.lower())
        final_tag = f"{prefix}_cauto_final_{tag_suffix}"
        final_cmd = build_final_training_command(
            args.python,
            model,
            final_tag,
            collection,
            decision.selected_modules,
            epochs=args.epochs,
            seed=args.seed,
            extra_args=model_extra_args(model),
        )
        final_commands.append(quote_command(final_cmd))
        if not args.skip_final and not args.dry_run:
            code = run_subprocess(final_cmd)
            with status_path.open("a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=["stage", "model", "module", "tag", "status"]).writerow(
                    {"stage": "final", "model": model, "module": "+".join(decision.selected_modules), "tag": final_tag, "status": "done" if code == 0 else f"failed:{code}"}
                )
            if code != 0:
                return code

    write_json(collection_path / "module_c_auto_decisions.json", decisions)
    (collection_path / "module_c_candidate_commands.ps1").write_text("\n".join(command_lines) + "\n", encoding="utf-8")
    (collection_path / "module_c_final_commands.ps1").write_text("\n".join(final_commands) + "\n", encoding="utf-8")
    print(f"[ModuleC-Auto] collection: {collection_path}")
    print(f"[ModuleC-Auto] decisions: {collection_path / 'module_c_auto_decisions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
