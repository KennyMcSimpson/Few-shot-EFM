"""Expand and optionally execute declarative Few-shot EFM experiment manifests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


class ManifestError(ValueError):
    """Raised when an experiment manifest is incomplete or inconsistent."""


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    seed: int
    run_tag: str
    command: tuple[str, ...]


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"Manifest field '{field}' must be a list of strings.")
    return list(value)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the portable subset of the experiment-manifest schema."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ManifestError("Experiment manifest must contain a JSON object.")
    for field in ("name", "entrypoint"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ManifestError(f"Manifest field '{field}' must be a non-empty string.")
    matrix = payload.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ManifestError("Manifest field 'matrix' must be an object.")
    datasets = _string_list(matrix.get("datasets"), "matrix.datasets")
    models = _string_list(matrix.get("models"), "matrix.models")
    seeds = matrix.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(
        isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
    ):
        raise ManifestError("Manifest field 'matrix.seeds' must be a non-empty list of integers.")
    if not datasets or not models:
        raise ManifestError("Manifest datasets and models cannot be empty.")
    fixed = _string_list(payload.get("fixed_cli_arguments"), "fixed_cli_arguments")
    if len(fixed) % 2 != 0 and not fixed[-1].startswith("--"):
        raise ManifestError("fixed_cli_arguments must contain CLI tokens.")
    overrides = payload.get("model_cli_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ManifestError("model_cli_overrides must be an object.")
    for model, tokens in overrides.items():
        if not isinstance(model, str):
            raise ManifestError("model_cli_overrides keys must be strings.")
        _string_list(tokens, f"model_cli_overrides.{model}")
    return dict(payload)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("._-").lower()
    if not slug:
        raise ManifestError(f"Cannot create a run tag from {value!r}.")
    return slug


def _validate_filter(
    requested: set[Any] | None,
    available: Sequence[Any],
    name: str,
) -> set[Any]:
    if requested is None:
        return set(available)
    unknown = set(requested) - set(available)
    if unknown:
        raise ManifestError(
            f"Unknown {name} filter values: {sorted(unknown)}; "
            f"available: {list(available)}"
        )
    return set(requested)


def expand_manifest(
    manifest: Mapping[str, Any],
    *,
    python_executable: str,
    output_root: Path,
    datasets: set[str] | None = None,
    models: set[str] | None = None,
    seeds: set[int] | None = None,
    preflight_only: bool = False,
) -> list[RunSpec]:
    """Expand one matrix manifest into deterministic, shell-free commands."""

    matrix = manifest["matrix"]
    available_datasets = list(matrix["datasets"])
    available_models = list(matrix["models"])
    available_seeds = list(matrix["seeds"])
    selected_datasets = _validate_filter(datasets, available_datasets, "dataset")
    selected_models = _validate_filter(models, available_models, "model")
    selected_seeds = _validate_filter(seeds, available_seeds, "seed")
    fixed = list(manifest["fixed_cli_arguments"])
    overrides = manifest.get("model_cli_overrides", {})
    name = _slug(str(manifest["name"]))
    entrypoint = str(manifest["entrypoint"])

    runs: list[RunSpec] = []
    for dataset, model, seed in itertools.product(
        available_datasets,
        available_models,
        available_seeds,
    ):
        if (
            dataset not in selected_datasets
            or model not in selected_models
            or seed not in selected_seeds
        ):
            continue
        run_tag = f"{name}__{_slug(dataset)}__{_slug(model)}__s{seed}"
        command = [
            str(python_executable),
            entrypoint,
            *fixed,
            "--dataset",
            dataset,
            "--model_name",
            model,
            "--seed",
            str(seed),
            "--output_dir",
            str(output_root),
            "--short_output_tag_only",
            "--run_tag",
            run_tag,
            *list(overrides.get(model, ())),
        ]
        if preflight_only:
            command.append("--module_c_preflight_only")
        runs.append(
            RunSpec(
                dataset=dataset,
                model=model,
                seed=int(seed),
                run_tag=run_tag,
                command=tuple(command),
            )
        )
    return runs


def _csv_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {token.strip() for token in value.split(",") if token.strip()}


def _seed_set(value: str | None) -> set[int] | None:
    tokens = _csv_set(value)
    return None if tokens is None else {int(token) for token in tokens}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Expand a portable experiment manifest. Commands are printed by default; "
            "training starts only with --execute."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--python", default=sys.executable, dest="python_executable")
    parser.add_argument("--output-root", type=Path, default=Path("finetuning_results"))
    parser.add_argument("--datasets", help="Optional comma-separated dataset filter.")
    parser.add_argument("--models", help="Optional comma-separated model filter.")
    parser.add_argument("--seeds", help="Optional comma-separated integer seed filter.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute sequentially. Without this flag, print commands only.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue executing later matrix entries after a non-zero return code.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        runs = expand_manifest(
            manifest,
            python_executable=args.python_executable,
            output_root=args.output_root,
            datasets=_csv_set(args.datasets),
            models=_csv_set(args.models),
            seeds=_seed_set(args.seeds),
            preflight_only=args.preflight_only,
        )
    except (ManifestError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"manifest error: {exc}") from exc

    for run in runs:
        record = {
            "dataset": run.dataset,
            "model": run.model,
            "seed": run.seed,
            "run_tag": run.run_tag,
            "command": list(run.command),
        }
        print(json.dumps(record, ensure_ascii=False))
        if not args.execute:
            continue
        completed = subprocess.run(
            run.command,
            cwd=REPO_ROOT,
            shell=False,
            check=False,
        )
        if completed.returncode != 0 and not args.continue_on_error:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
