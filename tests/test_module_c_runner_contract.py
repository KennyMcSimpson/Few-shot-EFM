import argparse
import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import RandomSampler, TensorDataset

import run_finetuning
from run_finetuning import (
    _ensure_module_c_preflight_has_no_resume,
    _ensure_module_c_preflight_is_single_process,
)
from util.fb_policy import add_fb_args, resolve_functional_args
from util.module_c_lora_search import build_module_c_recipe, parse_module_ids
from util.utils import auto_load_model, resolve_resume_checkpoint


class ModuleCRunnerContractTests(unittest.TestCase):
    @staticmethod
    def _resume_args(
        output_dir, resume="", auto_resume=True, enable_deepspeed=False
    ):
        return SimpleNamespace(
            output_dir=output_dir,
            resume=resume,
            auto_resume=auto_resume,
            enable_deepspeed=enable_deepspeed,
        )

    def test_public_experiment_manifest_tracks_the_exhaustive_status_contract(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        manifest_path = root / "experiment_manifests" / "module_c_exhaustive_seed0_4datasets.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.assertEqual(manifest["entrypoint"], "run_finetuning.py")
        self.assertEqual(manifest["matrix"]["seeds"], [0])
        self.assertEqual(
            manifest["matrix"]["datasets"],
            ["TUEV", "Sleep-EDF", "BCI-IV-2A", "SEED-IV"],
        )
        self.assertEqual(
            manifest["matrix"]["models"],
            ["EEGPT", "BIOT", "LaBraM", "CBraMod", "Gram", "CSBrain"],
        )
        self.assertIn("selection_status", manifest["decision_artifact"]["status_fields"])
        serialized = json.dumps(manifest).lower()
        self.assertNotIn("evidence_strength", serialized)
        self.assertNotIn("task_aligned", serialized)

    def test_module_c_public_contract_contains_no_tracked_shell_runner(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        self.assertFalse((root / "run_c_exhaustive_seed0_4datasets_5090.sh").exists())
        self.assertFalse((root / "run_c_exhaustive_tuev_3seed_5070.bat").exists())

    def test_pure_resume_resolver_handles_explicit_newest_auto_and_none(self):
        with tempfile.TemporaryDirectory() as output_dir:
            explicit = os.path.join(output_dir, "manual.pth")
            pathlib.Path(explicit).touch()
            args = self._resume_args(output_dir, resume=explicit)
            self.assertEqual(resolve_resume_checkpoint(args), explicit)
            self.assertEqual(args.resume, explicit)

        with tempfile.TemporaryDirectory() as output_dir:
            pathlib.Path(output_dir, "checkpoint-2.pth").touch()
            pathlib.Path(output_dir, "checkpoint-11.pth").touch()
            pathlib.Path(output_dir, "checkpoint-invalid.pth").touch()
            args = self._resume_args(output_dir)
            self.assertEqual(
                resolve_resume_checkpoint(args),
                os.path.join(output_dir, "checkpoint-11.pth"),
            )
            self.assertEqual(args.resume, "")

        with tempfile.TemporaryDirectory() as output_dir:
            self.assertIsNone(resolve_resume_checkpoint(self._resume_args(output_dir)))

    def test_deepspeed_resume_is_auto_only_and_ignores_explicit_path(self):
        with tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as other_dir:
            pathlib.Path(output_dir, "checkpoint-2").mkdir()
            pathlib.Path(output_dir, "checkpoint-11").mkdir()
            pathlib.Path(output_dir, "checkpoint-invalid").mkdir()
            explicit = os.path.join(other_dir, "checkpoint-99")
            pathlib.Path(explicit).mkdir()
            args = self._resume_args(
                output_dir,
                resume=explicit,
                auto_resume=True,
                enable_deepspeed=True,
            )

            self.assertEqual(
                resolve_resume_checkpoint(args),
                os.path.join(output_dir, "checkpoint-11"),
            )
            self.assertEqual(args.resume, explicit)

            class _DeepSpeedModel:
                def __init__(self):
                    self.calls = []

                def load_checkpoint(self, directory, tag):
                    self.calls.append((directory, tag))
                    return None, {"epoch": 4}

            model = _DeepSpeedModel()
            auto_load_model(args, model, None, None, None)
            self.assertEqual(model.calls, [(output_dir, "checkpoint-11")])
            self.assertEqual(args.start_epoch, 5)

    def test_deepspeed_resolver_returns_none_when_auto_resume_is_disabled(self):
        with tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as other_dir:
            pathlib.Path(output_dir, "checkpoint-7").mkdir()
            explicit = os.path.join(other_dir, "checkpoint-99")
            pathlib.Path(explicit).mkdir()
            args = self._resume_args(
                output_dir,
                resume=explicit,
                auto_resume=False,
                enable_deepspeed=True,
            )

            self.assertIsNone(resolve_resume_checkpoint(args))

    def test_automatic_module_c_resume_guard_rejects_only_discovered_checkpoint(self):
        with tempfile.TemporaryDirectory() as output_dir:
            args = self._resume_args(output_dir)
            _ensure_module_c_preflight_has_no_resume(args)
            pathlib.Path(output_dir, "checkpoint.pth").touch()
            with self.assertRaisesRegex(RuntimeError, "new output directory|no resume"):
                _ensure_module_c_preflight_has_no_resume(args)

        with tempfile.TemporaryDirectory() as output_dir:
            pathlib.Path(output_dir, "checkpoint.pth").touch()
            _ensure_module_c_preflight_has_no_resume(
                self._resume_args(output_dir, auto_resume=False)
            )

    def test_automatic_module_c_rejects_distributed_topology_selection(self):
        with patch("run_finetuning.utils.get_world_size", return_value=1):
            _ensure_module_c_preflight_is_single_process()
        with patch("run_finetuning.utils.get_world_size", return_value=2):
            with self.assertRaisesRegex(RuntimeError, "single-process|world_size=1"):
                _ensure_module_c_preflight_is_single_process()

    def test_formal_training_loader_remains_random_and_drop_last(self):
        make_loader = getattr(run_finetuning, "_make_formal_train_loader", None)
        self.assertIsNotNone(
            make_loader, "formal training must use the behavior-tested loader helper"
        )
        dataset = TensorDataset(torch.arange(11))
        generator = torch.Generator().manual_seed(23)
        sampler = RandomSampler(dataset, generator=generator)
        args = SimpleNamespace(
            batch_size=3,
            num_workers=0,
            pin_mem=False,
        )

        loader = make_loader(args, dataset, sampler)

        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertTrue(loader.drop_last)
        visible_ids = [int(value) for batch in loader for value in batch[0].tolist()]
        self.assertEqual(len(visible_ids), 9)
        self.assertEqual(len(set(visible_ids)), 9)
        self.assertTrue(set(visible_ids).issubset(set(range(11))))

    def test_documentation_describes_complete_support_and_validation(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        help_text = (root / "util" / "fb_policy.py").read_text(encoding="utf-8")
        combined = "\n".join((readme, help_text)).lower()
        self.assertIn("complete support", combined)
        self.assertIn("drop_last=false", combined)
        self.assertIn("validation", combined)
        self.assertNotIn("subject-cluster", combined)
        self.assertNotIn("holm", combined)
        self.assertNotIn("hierarchical_forward", combined)

    def test_c_accepts_only_bde_and_never_emits_qv_metadata(self):
        with self.assertRaises(ValueError):
            parse_module_ids("B,qv,E")

        recipe = build_module_c_recipe(("B", "E"))
        self.assertNotIn("qv", str(recipe).lower())

    def test_parser_keeps_batch_caps_as_debug_controls(self):
        parser = argparse.ArgumentParser()
        add_fb_args(parser)
        args = parser.parse_args([])

        self.assertEqual(args.module_c_preflight_train_batches, 0)
        self.assertEqual(args.module_c_preflight_val_batches, 0)
        self.assertFalse(args.module_c_preflight_only)
        self.assertFalse(hasattr(args, "module_c_rgfs_harm_threshold"))
        self.assertFalse(hasattr(args, "module_c_probe_head_steps"))

    def test_parser_exposes_preflight_only_for_no_training_verification(self):
        parser = argparse.ArgumentParser()
        add_fb_args(parser)
        args = parser.parse_args(["--module_c_preflight_only"])

        self.assertTrue(args.module_c_preflight_only)

    def test_disabled_preflight_requires_an_explicit_nonempty_bde_selection(self):
        parser = argparse.ArgumentParser()
        add_fb_args(parser)
        args = parser.parse_args(["--module_c_no_preflight"])
        args.lora_target = "module_c"

        with self.assertRaisesRegex(ValueError, "nonempty --module_c_selected"):
            resolve_functional_args(args)

    def test_automatic_preflight_rejects_a_partial_or_reordered_registry(self):
        parser = argparse.ArgumentParser()
        add_fb_args(parser)
        for candidates in ("B,D", "E,D,B"):
            with self.subTest(candidates=candidates):
                args = parser.parse_args(["--module_c_candidates", candidates])
                args.lora_target = "module_c"
                with self.assertRaisesRegex(ValueError, "exactly B,D,E"):
                    resolve_functional_args(args)


if __name__ == "__main__":
    unittest.main()
