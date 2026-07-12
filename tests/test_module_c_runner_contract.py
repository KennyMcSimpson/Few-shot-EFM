import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from run_finetuning import _ensure_module_c_preflight_has_no_resume
from util.utils import resolve_resume_checkpoint


class ModuleCRunnerContractTests(unittest.TestCase):
    @staticmethod
    def _resume_args(output_dir, resume="", auto_resume=True):
        return SimpleNamespace(
            output_dir=output_dir,
            resume=resume,
            auto_resume=auto_resume,
            enable_deepspeed=False,
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
        source = pathlib.Path(__file__).resolve().parents[1] / "run_finetuning.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn(
            "sampler_train = torch.utils.data.RandomSampler(dataset_train)", text
        )
        formal_loader_block = text.split(
            "data_loader_train = torch.utils.data.DataLoader(", 1
        )[1].split("if args.monitor_dynamics", 1)[0]
        self.assertIn("sampler=sampler_train", formal_loader_block)
        self.assertIn("_data_loader_kwargs(args, drop_last=True)", formal_loader_block)

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
