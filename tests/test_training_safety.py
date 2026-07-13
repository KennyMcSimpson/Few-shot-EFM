import csv
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import engine_for_finetuning
import run_finetuning
import torch
from util.lora import (
    validate_full_lora_base_trainability,
    validate_optimizer_parameter_coverage,
)


class GenericTrainingDefaultsTests(unittest.TestCase):
    def test_lora_base_update_has_no_silent_default(self):
        with patch.object(sys, "argv", ["run_finetuning.py"]):
            args, _ = run_finetuning.get_args()
        self.assertIsNone(args.lora_base_update)

    def test_lora_setup_requires_explicit_base_update_mode(self):
        args = SimpleNamespace(
            task_mod="Classification",
            model_name="BIOT",
            lora_base_update=None,
        )
        with self.assertRaisesRegex(ValueError, "explicit.*lora_base_update"):
            run_finetuning._apply_lora_training_setup(torch.nn.Linear(2, 1), args)

    def test_lora_cli_contract_fails_before_output_or_model_setup(self):
        args, _ = run_finetuning.get_args(["--finetune_mod", "lora"])
        with self.assertRaisesRegex(ValueError, "explicit.*lora_base_update"):
            run_finetuning._validate_lora_cli_contract(args)

        args, _ = run_finetuning.get_args(
            ["--finetune_mod", "lora", "--lora_base_update", "full"]
        )
        self.assertIsNone(run_finetuning._validate_lora_cli_contract(args))

    def test_cudnn_autotuner_is_disabled_for_variable_eeg_shapes(self):
        backend = SimpleNamespace(benchmark=True)
        with patch.object(run_finetuning, "cudnn", backend):
            run_finetuning._configure_cudnn_runtime()
            self.assertFalse(backend.benchmark)
        self.assertIn(
            "_configure_cudnn_runtime()", inspect.getsource(run_finetuning.main)
        )

    def test_boundary_anchor_default_and_snapshot_help_are_generic(self):
        with patch.object(sys, "argv", ["run_finetuning.py"]):
            args, _ = run_finetuning.get_args()

        self.assertEqual(args.boundary_anchor_metric, "selection_bacc_worst_std")

        parser_source = inspect.getsource(run_finetuning.get_args)
        self.assertIn("e.g. selection_bacc_worst_std", parser_source)
        self.assertNotIn("e.g. selection_bacc_min02_std", parser_source)

        fallback_source = inspect.getsource(run_finetuning._maybe_update_boundary_anchor)
        self.assertIn("'selection_bacc_worst_std'", fallback_source)
        self.assertNotIn("'selection_bacc_min02_std'", fallback_source)

        selection_doc = inspect.getdoc(run_finetuning._add_selection_metrics)
        snapshot_doc = inspect.getdoc(run_finetuning._run_snapshot_ensemble_report)
        self.assertNotIn("designed for TUEV", selection_doc)
        self.assertNotIn("designed for TUEV", snapshot_doc)

    def test_snapshot_candidate_csv_keeps_generic_and_legacy_metric_fields(self):
        ranked_rows = [
            {
                "epoch": 3,
                "score": 0.61,
                "row": {
                    "val_balanced_accuracy": 0.55,
                    "val_selection_bacc_worst_std": 0.61,
                    "val_selection_bacc_min02_std": 0.58,
                },
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            run_finetuning._write_snapshot_candidates(
                tmp, ranked_rows, "val_selection_bacc_worst_std"
            )
            with Path(tmp, "snapshot_candidates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)

        self.assertIn("val_selection_bacc_worst_std", reader.fieldnames)
        self.assertIn("val_selection_bacc_min02_std", reader.fieldnames)
        self.assertEqual(rows[0]["val_selection_bacc_worst_std"], "0.61")
        self.assertEqual(rows[0]["val_selection_bacc_min02_std"], "0.58")


class FullFTLoRAContractTests(unittest.TestCase):
    def test_full_lora_trainability_accepts_exact_restoration(self):
        model = torch.nn.Linear(2, 1)
        model.bias.requires_grad_(False)
        before = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
        self.assertIsNone(validate_full_lora_base_trainability(model, before))

    def test_full_lora_trainability_rejects_changed_or_missing_base_parameters(self):
        model = torch.nn.Linear(2, 1)
        before = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
        model.weight.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "trainability changed"):
            validate_full_lora_base_trainability(model, before)

        model = torch.nn.Linear(2, 1)
        before = {id(parameter): parameter.requires_grad for parameter in model.parameters()}
        model.weight = torch.nn.Parameter(torch.zeros_like(model.weight))
        with self.assertRaisesRegex(RuntimeError, "parameter objects disappeared"):
            validate_full_lora_base_trainability(model, before)

    def test_optimizer_coverage_accepts_each_trainable_parameter_once(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.assertIsNone(validate_optimizer_parameter_coverage(model, optimizer))

    def test_optimizer_coverage_rejects_missing_duplicate_and_frozen_parameters(self):
        model = torch.nn.Linear(2, 1)
        missing = torch.optim.SGD([model.weight], lr=0.1)
        with self.assertRaisesRegex(RuntimeError, "missing trainable parameters.*bias"):
            validate_optimizer_parameter_coverage(model, missing)

        duplicate = torch.optim.SGD(model.parameters(), lr=0.1)
        duplicate.param_groups.append({**duplicate.param_groups[0], "params": [model.weight]})
        with self.assertRaisesRegex(RuntimeError, "duplicate parameters.*weight"):
            validate_optimizer_parameter_coverage(model, duplicate)

        model.bias.requires_grad_(False)
        includes_frozen = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(RuntimeError, "frozen parameters.*bias"):
            validate_optimizer_parameter_coverage(model, includes_frozen)


class NonFiniteLossSafetyTests(unittest.TestCase):
    def _guard(self):
        guard = getattr(engine_for_finetuning, "_require_finite_loss", None)
        self.assertIsNotNone(guard, "finite-loss guard is missing")
        return guard

    def test_require_finite_loss_returns_finite_values(self):
        guard = self._guard()
        for value in (0.0, -1.25, 3.5e20):
            with self.subTest(value=value):
                self.assertEqual(guard(value, "unit-test"), value)

    def test_require_finite_loss_rejects_all_non_finite_values_with_context(self):
        guard = self._guard()
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(FloatingPointError) as caught:
                    guard(value, "retrieval safety test")
                message = str(caught.exception)
                self.assertIn(str(value), message)
                self.assertIn("retrieval safety test", message)

    def test_classification_and_retrieval_guard_before_backward_paths(self):
        classification_source = inspect.getsource(
            engine_for_finetuning.train_one_epoch
        )
        retrieval_source = inspect.getsource(engine_for_finetuning.train_model)

        self.assertIn("_require_finite_loss", classification_source)
        classification_guard = classification_source.index("_require_finite_loss")
        self.assertLess(
            classification_guard,
            classification_source.index("loss_scaler(loss, optimizer"),
        )
        self.assertNotIn("Warning: Loss is", classification_source)

        self.assertIn("_require_finite_loss", retrieval_source)
        retrieval_guard = retrieval_source.index("_require_finite_loss")
        self.assertLess(
            retrieval_guard, retrieval_source.index("eeg_model.backward(loss)")
        )
        self.assertLess(
            retrieval_guard,
            retrieval_source.index("loss_scaler(loss, optimizer"),
        )
        self.assertNotIn('print("Loss is', retrieval_source)


if __name__ == "__main__":
    unittest.main()
