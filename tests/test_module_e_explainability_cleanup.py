import argparse
import csv
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

if torch is not None:
    from util.lora import apply_lora_to_eegfm
from util.module_e_structural_routing import (
    module_e_mode_from_args,
    save_module_e_lora_injection_audit,
)


if torch is not None:
    class TinyEEGPTForModuleE(nn.Module):
        def __init__(self):
            super().__init__()
            self.main_model = nn.Module()
            self.main_model.blocks = nn.ModuleList([nn.Module()])
            self.main_model.blocks[0].attn = nn.Module()
            self.main_model.blocks[0].attn.qkv = nn.Linear(4, 12)
            self.task_head = nn.Linear(4, 2)


class ModuleEExplainabilityCleanupTest(unittest.TestCase):
    def test_old_module_e_modes_resolve_to_dynamic_only(self):
        self.assertEqual(
            module_e_mode_from_args(argparse.Namespace(module_e_mode="static_pressure_topk")),
            "dynamic_pressure_gate",
        )
        self.assertEqual(
            module_e_mode_from_args(argparse.Namespace(module_e_mode="legacy_all_structural")),
            "dynamic_pressure_gate",
        )
        self.assertEqual(
            module_e_mode_from_args(argparse.Namespace(module_e_mode="")),
            "dynamic_pressure_gate",
        )

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_lora_injection_audit_records_pressure_scope(self):
        model = TinyEEGPTForModuleE()
        replaced = apply_lora_to_eegfm(
            model,
            "EEGPT",
            lora_target="struct_mix",
            r=2,
            alpha=4,
            dropout=0.0,
            verbose=False,
        )
        self.assertEqual(replaced, ["main_model.blocks.0.attn.qkv"])

        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(
                fb_enable=True,
                output_dir=tmp,
                model_name="EEGPT",
                run_tag="module_e_audit_case",
                lora_target="struct_mix",
                lora_base_update="freeze",
                module_e_injected_names=";".join(replaced),
                module_e_dynamic_pressure_enabled=True,
                module_e_dynamic_pressure_branches="mixing",
            )

            path = save_module_e_lora_injection_audit(args, model)
            self.assertIsNotNone(path)

            with Path(path).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["module_name"], "main_model.blocks.0.attn.qkv")
        self.assertEqual(rows[0]["structural_branch"], "mixing")
        self.assertEqual(rows[0]["pressure_param_scope"], "lora_b_only")
        self.assertEqual(rows[0]["controlled_param_scope"], "all_lora_params_in_same_structural_branch")
        self.assertEqual(rows[0]["dynamic_pressure_controlled"], "1")


if __name__ == "__main__":
    unittest.main()
