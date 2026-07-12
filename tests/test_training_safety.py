import csv
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import engine_for_finetuning
import run_finetuning


class GenericTrainingDefaultsTests(unittest.TestCase):
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
