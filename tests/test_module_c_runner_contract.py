import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch.utils.data import RandomSampler, TensorDataset

import run_finetuning
from run_finetuning import _ensure_module_c_preflight_has_no_resume
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

    def test_decision_lookup_follows_remote_results_symlink(self):
        runner = pathlib.Path(__file__).resolve().parents[1] / "run_c_task_aligned_seed0_4datasets_5090.sh"
        text = runner.read_text(encoding="utf-8")

        self.assertIn("find -L finetuning_results", text)

    def test_status_prefers_final_path_evidence_strength(self):
        runner = pathlib.Path(__file__).resolve().parents[1] / "run_c_task_aligned_seed0_4datasets_5090.sh"
        text = runner.read_text(encoding="utf-8")

        self.assertIn(
            'payload.get("final_evidence_strength", payload.get("primary_evidence_strength", "NA"))',
            text,
        )

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

    def test_documentation_describes_formal_visible_support_and_complete_validation(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        help_text = (root / "util" / "fb_policy.py").read_text(encoding="utf-8")
        design = (
            root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-07-12-module-c-task-aligned-search-design.md"
        ).read_text(encoding="utf-8")
        plan = (
            root
            / "docs"
            / "superpowers"
            / "plans"
            / "2026-07-12-module-c-task-aligned-search-plan.md"
        ).read_text(encoding="utf-8")
        combined = "\n".join((readme, help_text, design, plan)).lower()
        self.assertIn("complete formal-visible support epoch", combined)
        self.assertIn("drop_last=true", combined)
        self.assertIn("raw tail", combined)
        self.assertIn("validation", combined)
        self.assertIn("drop_last=false", combined)
        self.assertNotIn("complete support and validation splits", readme.lower())
        self.assertNotIn("full train split", help_text.lower())


if __name__ == "__main__":
    unittest.main()
