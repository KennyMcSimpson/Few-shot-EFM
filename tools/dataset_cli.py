"""Cross-platform dataset preprocessing, split generation, and split auditing."""

from __future__ import annotations

import argparse
import json
import ntpath
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_DIRECTORY_ALIASES = {
    "BCI-4-2A": "BCI-IV-2A",
}
_SPLIT_PAIRS = (
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
)


class DatasetCommandError(ValueError):
    """Raised when a dataset command cannot be mapped to a repository script."""


def load_training_registry(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Load the datasets that are actually runnable through dataset_config."""

    registry: dict[str, dict[str, Any]] = {}
    config_root = Path(repo_root) / "dataset_config"
    for config_path in sorted(config_root.glob("*.json")):
        task = config_path.stem
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise DatasetCommandError(f"Dataset config must contain an object: {config_path}")
        for dataset, metadata in payload.items():
            entry = registry.setdefault(
                str(dataset),
                {"tasks": [], "config_by_task": {}},
            )
            entry["tasks"].append(task)
            entry["config_by_task"][task] = metadata

    for entry in registry.values():
        entry["tasks"] = tuple(sorted(set(entry["tasks"])))
    return dict(sorted(registry.items()))


def discover_preprocessing(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """Discover dataset preprocessing capabilities from the checked-in scripts."""

    preprocessing_root = Path(repo_root) / "preprocessing"
    discovered: dict[str, dict[str, Any]] = {}
    for directory in sorted(path for path in preprocessing_root.iterdir() if path.is_dir()):
        logical_name = _DIRECTORY_ALIASES.get(directory.name, directory.name)
        prepare_script = directory / "data_process.py"
        split_scripts = {
            mode: directory / f"{mode}_json_process.py"
            for mode in ("cross", "multi")
            if (directory / f"{mode}_json_process.py").is_file()
        }
        if not prepare_script.is_file() and not split_scripts:
            continue
        discovered[logical_name] = {
            "directory": directory,
            "prepare_script": prepare_script if prepare_script.is_file() else None,
            "split_scripts": split_scripts,
            "split_modes": tuple(split_scripts),
        }
    return dict(sorted(discovered.items()))


def _canonical_dataset_name(
    dataset: str,
    discovered: Mapping[str, Mapping[str, Any]],
) -> str:
    matches = [name for name in discovered if name.casefold() == dataset.casefold()]
    if not matches:
        available = ", ".join(sorted(discovered))
        raise DatasetCommandError(
            f"Unknown preprocessing dataset '{dataset}'. Available datasets: {available}"
        )
    return matches[0]


def resolve_script(
    repo_root: Path,
    dataset: str,
    action: str,
    split_mode: str | None = None,
) -> Path:
    """Resolve one logical dataset action to its checked-in Python script."""

    discovered = discover_preprocessing(repo_root)
    canonical = _canonical_dataset_name(dataset, discovered)
    entry = discovered[canonical]
    if action == "prepare":
        script = entry["prepare_script"]
        if script is None:
            raise DatasetCommandError(f"Dataset '{canonical}' has no preprocessing script.")
        return Path(script)
    if action != "split":
        raise DatasetCommandError(f"Unknown dataset action '{action}'.")
    if split_mode not in ("cross", "multi"):
        raise DatasetCommandError("Split generation requires split mode 'cross' or 'multi'.")
    script = entry["split_scripts"].get(split_mode)
    if script is None:
        raise DatasetCommandError(
            f"Dataset '{canonical}' does not provide split mode '{split_mode}'."
        )
    return Path(script)


def build_dataset_command(
    repo_root: Path,
    *,
    dataset: str,
    action: str,
    data_root: Path,
    split_mode: str | None = None,
) -> list[str]:
    """Build a shell-free child-Python command for one dataset action."""

    script = resolve_script(repo_root, dataset, action, split_mode)
    return [sys.executable, script.as_posix(), str(data_root)]


def _load_split_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetCommandError(f"Missing split file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("subject_data")
    if not isinstance(payload, list):
        raise DatasetCommandError(
            f"Split JSON must contain a list or a 'subject_data' list: {path}"
        )
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or not isinstance(row.get("file"), str):
            raise DatasetCommandError(
                f"Split row {index} in {path} must contain a string 'file' field."
            )
        rows.append(dict(row))
    return rows


def _normalized_path(value: str) -> str:
    if "\\" in value or (len(value) >= 2 and value[1] == ":"):
        return ntpath.normcase(ntpath.normpath(value))
    return os.path.normcase(os.path.normpath(value))


def _basename(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].casefold()


def _display_overlap(
    overlap: set[str],
    originals: Mapping[str, set[str]],
) -> list[str]:
    return sorted(min(originals[value]) for value in overlap)


def audit_split_directory(split_dir: Path) -> dict[str, Any]:
    """Audit exact-path and basename overlap across train/val/test JSON files."""

    split_dir = Path(split_dir)
    rows = {
        split: _load_split_rows(split_dir / f"{split}.json")
        for split in ("train", "val", "test")
    }
    normalized: dict[str, set[str]] = {}
    basenames: dict[str, set[str]] = {}
    path_originals: dict[str, set[str]] = {}
    basename_originals: dict[str, set[str]] = {}
    for split, split_rows in rows.items():
        normalized[split] = set()
        basenames[split] = set()
        for row in split_rows:
            raw_path = row["file"]
            normalized_path = _normalized_path(raw_path)
            basename = _basename(raw_path)
            normalized[split].add(normalized_path)
            basenames[split].add(basename)
            path_originals.setdefault(normalized_path, set()).add(raw_path)
            basename_originals.setdefault(basename, set()).add(
                raw_path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            )

    overlaps: dict[str, dict[str, list[str]]] = {}
    for left, right in _SPLIT_PAIRS:
        exact_overlap = normalized[left] & normalized[right]
        basename_overlap = basenames[left] & basenames[right]
        overlaps[f"{left}__{right}"] = {
            "exact_paths": _display_overlap(exact_overlap, path_originals),
            "basenames": _display_overlap(basename_overlap, basename_originals),
        }

    return {
        "split_directory": str(split_dir),
        "counts": {split: len(split_rows) for split, split_rows in rows.items()},
        "overlaps": overlaps,
        "ok": all(
            not details["exact_paths"] and not details["basenames"]
            for details in overlaps.values()
        ),
    }


def _dataset_rows(repo_root: Path) -> list[dict[str, str]]:
    training = load_training_registry(repo_root)
    preprocessing = discover_preprocessing(repo_root)
    rows = []
    for dataset in sorted(set(training) | set(preprocessing)):
        tasks = training.get(dataset, {}).get("tasks", ())
        split_modes = preprocessing.get(dataset, {}).get("split_modes", ())
        if tasks and dataset in preprocessing:
            status = "training-configured"
        elif tasks:
            status = "configured-no-preprocessor"
        else:
            status = "preprocessing-only"
        rows.append(
            {
                "dataset": dataset,
                "tasks": ",".join(tasks) or "-",
                "prepare": "yes" if preprocessing.get(dataset, {}).get("prepare_script") else "no",
                "split_modes": ",".join(split_modes) or "-",
                "status": status,
            }
        )
    return rows


def _print_dataset_rows(rows: Sequence[Mapping[str, str]]) -> None:
    headers = ("dataset", "tasks", "prepare", "split_modes", "status")
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def _run_dataset_action(args: argparse.Namespace, action: str) -> int:
    command = build_dataset_command(
        args.repo_root,
        dataset=args.dataset,
        action=action,
        data_root=args.data_root,
        split_mode=getattr(args, "mode", None),
    )
    if args.dry_run:
        print(json.dumps(command, ensure_ascii=False))
        return 0
    subprocess.run(command, cwd=args.repo_root, check=True, shell=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Few-shot EFM cross-platform dataset utility."
    )
    parser.set_defaults(repo_root=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List configured and preprocessing-only datasets.",
    )
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Run dataset-specific raw-data preprocessing.",
    )
    prepare_parser.add_argument("dataset")
    prepare_parser.add_argument("data_root", type=Path)
    prepare_parser.add_argument("--dry-run", action="store_true")

    split_parser = subparsers.add_parser(
        "split",
        help="Generate local train/validation/test JSON indexes.",
    )
    split_parser.add_argument("dataset")
    split_parser.add_argument("data_root", type=Path)
    split_parser.add_argument("--mode", choices=("cross", "multi"), required=True)
    split_parser.add_argument("--dry-run", action="store_true")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Check exact-path and basename overlap across split JSON files.",
    )
    audit_parser.add_argument("split_dir", type=Path)
    audit_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            rows = _dataset_rows(args.repo_root)
            if args.as_json:
                print(json.dumps(rows, indent=2, ensure_ascii=False))
            else:
                _print_dataset_rows(rows)
            return 0
        if args.command == "prepare":
            return _run_dataset_action(args, "prepare")
        if args.command == "split":
            return _run_dataset_action(args, "split")
        if args.command == "audit":
            audit = audit_split_directory(args.split_dir)
            if args.as_json:
                print(json.dumps(audit, indent=2, ensure_ascii=False))
            else:
                print(f"split_directory: {audit['split_directory']}")
                print(f"counts: {audit['counts']}")
                for pair, details in audit["overlaps"].items():
                    print(
                        f"{pair}: exact_paths={len(details['exact_paths'])}, "
                        f"basenames={len(details['basenames'])}"
                    )
                print(f"ok: {audit['ok']}")
            return 0 if audit["ok"] else 1
    except (DatasetCommandError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
