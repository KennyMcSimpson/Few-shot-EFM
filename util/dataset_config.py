"""Validated access to task-specific dataset configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "dataset_config"


class DatasetConfigError(ValueError):
    """Raised when a training task cannot resolve its dataset metadata."""


def load_task_dataset_info(
    task_mod: str,
    dataset: str,
    *,
    config_root: Path = DEFAULT_CONFIG_ROOT,
) -> dict[str, Any] | None:
    """Return validated metadata, or ``None`` for retrieval's separate loader."""

    if task_mod == "Retrieval":
        return None

    config_path = Path(config_root) / f"{task_mod}.json"
    if not config_path.is_file():
        raise DatasetConfigError(
            f"Missing dataset config for task '{task_mod}': {config_path}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise DatasetConfigError(f"Dataset config must contain an object: {config_path}")

    metadata = payload.get(dataset)
    if not isinstance(metadata, Mapping):
        available = ", ".join(sorted(str(name) for name in payload)) or "none"
        raise DatasetConfigError(
            f"Unknown {task_mod} dataset '{dataset}'. Configured datasets: {available}"
        )
    roots = metadata.get("root")
    if not isinstance(roots, Mapping):
        raise DatasetConfigError(
            f"Dataset '{dataset}' in {config_path} must define a 'root' object."
        )
    return dict(metadata)
