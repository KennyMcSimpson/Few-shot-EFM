from __future__ import annotations

import os
from typing import Any, Callable, Optional

import torch

from .fb_probe import (
    save_module_b_config,
    save_module_b_matrix_summary,
    save_signal_alignment_probe,
)


def _cpu_tensor_state(model) -> dict:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if torch.is_tensor(v)
    }


def run_signal_alignment_probe_after_training(
    args: Any,
    model: Any,
    data_loader_val: Any,
    device: Any,
    lifecycle_selection_row: Optional[dict] = None,
    is_main_process: bool = True,
    save_probe_fn: Optional[Callable[..., Any]] = None,
) -> bool:
    save_probe = save_probe_fn or save_signal_alignment_probe
    if not bool(is_main_process):
        return False

    wrote_any = False
    try:
        wrote_any = bool(save_module_b_config(args=args, model=model)) or wrote_any
    except Exception as exc:
        print(f"[FB2][WARN] Module B config export failed: {exc}")
    try:
        wrote_any = bool(save_module_b_matrix_summary(args=args, model=model)) or wrote_any
    except Exception as exc:
        print(f"[FB2][WARN] Module B matrix summary export failed: {exc}")

    if data_loader_val is None:
        return bool(wrote_any)

    signal_probe_written = False
    if lifecycle_selection_row is not None:
        selected_ckpt_path = os.path.join(
            args.output_dir, "monitor_checkpoints", "adaptive_swa_selected.pth"
        )
        if os.path.exists(selected_ckpt_path):
            restore_state_for_probe = _cpu_tensor_state(model)
            try:
                selected_ckpt = torch.load(
                    selected_ckpt_path, map_location="cpu", weights_only=False
                )
                selected_state = selected_ckpt.get("model", selected_ckpt)
                model.load_state_dict(selected_state, strict=False)
                signal_probe_written = bool(
                    save_probe(
                        args=args,
                        model=model,
                        data_loader=data_loader_val,
                        device=device,
                        split="val_lifecycle_selected",
                        selection_row=lifecycle_selection_row,
                    )
                )
            except Exception as exc:
                print(f"[FB2][WARN] signal alignment probe on lifecycle-selected state failed: {exc}")
            finally:
                model.load_state_dict(restore_state_for_probe, strict=False)
        else:
            print(f"[FB2][WARN] selected lifecycle checkpoint not found for signal probe: {selected_ckpt_path}")

    if (
        not signal_probe_written
        and getattr(args, "task_mod", "") == "Classification"
    ):
        signal_probe_written = bool(
            save_probe(
                args=args,
                model=model,
                data_loader=data_loader_val,
                device=device,
                split="val_final_model",
                selection_row=lifecycle_selection_row,
            )
        )

    return bool(signal_probe_written or wrote_any)
