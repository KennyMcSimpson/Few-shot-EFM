import csv
import tempfile
import unittest
from pathlib import Path

from util.module_d_semantic_boundary_summary import build_semantic_boundary_summary
from util.module_d_semantic_refinement import module_d_sbr_rows, write_module_d_sbr_eval


def _write_one_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class ModuleDSemanticBoundarySummaryTest(unittest.TestCase):
    def test_summary_reports_need_effect_and_recovery_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.csv"
            capability = root / "capability.csv"
            adapted = root / "adapted.csv"
            output = root / "summary.csv"

            reference_row = {
                "model_name": "BIOT",
                "reference_name": "fewshot_ref",
                "val_class_0": 0.10,
                "val_class_1": 0.80,
                "val_class_2": 0.20,
                "val_class_3": 0.70,
                "val_balanced_accuracy": 0.45,
                "test_class_0": 0.11,
                "test_class_1": 0.82,
                "test_class_2": 0.19,
                "test_class_3": 0.69,
                "test_balanced_accuracy": 0.452,
            }
            capability_row = {
                "model_name": "BIOT",
                "reference_name": "full_anchor",
                "val_class_0": 0.30,
                "val_class_1": 0.78,
                "val_class_2": 0.35,
                "val_class_3": 0.68,
                "val_balanced_accuracy": 0.5275,
                "test_class_0": 0.20,
                "test_class_1": 0.81,
                "test_class_2": 0.28,
                "test_class_3": 0.68,
                "test_balanced_accuracy": 0.4925,
            }
            adapted_row = {
                "model_name": "BIOT",
                "reference_name": "semantic_d",
                "val_class_0": 0.22,
                "val_class_1": 0.79,
                "val_class_2": 0.30,
                "val_class_3": 0.65,
                "val_balanced_accuracy": 0.49,
                "test_class_0": 0.16,
                "test_class_1": 0.81,
                "test_class_2": 0.24,
                "test_class_3": 0.60,
                "test_balanced_accuracy": 0.4525,
            }
            _write_one_row(reference, reference_row)
            _write_one_row(capability, capability_row)
            _write_one_row(adapted, adapted_row)

            rows = build_semantic_boundary_summary(
                reference_csv=str(reference),
                capability_csv=str(capability),
                adapted_csv=str(adapted),
                output_csv=str(output),
                model_name="BIOT",
                reference_name="fewshot_ref",
                capability_name="full_anchor",
                adapted_name="semantic_d",
                hard_k=2,
                nb_classes=4,
            )

            self.assertEqual(len(rows), 2)
            val = _read_rows(output)[0]
            self.assertEqual(val["split"], "val")
            self.assertEqual(val["hard_classes"], "0,2")
            self.assertAlmostEqual(float(val["need_sbr"]), 0.155)
            self.assertAlmostEqual(float(val["d_effect_sbr"]), 0.08)
            self.assertAlmostEqual(float(val["recovery_ratio"]), 0.08 / 0.155)
            self.assertEqual(val["interpretation"], "partial_recovery_with_stable_tradeoff")

            test = _read_rows(output)[1]
            self.assertEqual(test["split"], "test")
            self.assertEqual(test["hard_classes"], "0,2")
            self.assertEqual(test["test_used_for_hard_class_selection"], "0")

    def test_summary_can_use_existing_module_d_sbr_eval_as_adapted_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.csv"
            capability = root / "capability.csv"
            adapted_sbr = root / "module_d_sbr_eval.csv"
            output = root / "summary.csv"

            reference_row = {
                "model_name": "BIOT",
                "reference_name": "fewshot_ref",
                "val_class_0": 0.10,
                "val_class_1": 0.80,
                "val_class_2": 0.20,
                "val_class_3": 0.70,
                "val_balanced_accuracy": 0.45,
                "test_class_0": 0.11,
                "test_class_1": 0.82,
                "test_class_2": 0.19,
                "test_class_3": 0.69,
                "test_balanced_accuracy": 0.452,
            }
            capability_row = {
                "model_name": "BIOT",
                "reference_name": "full_anchor",
                "val_class_0": 0.30,
                "val_class_1": 0.78,
                "val_class_2": 0.35,
                "val_class_3": 0.68,
                "val_balanced_accuracy": 0.5275,
                "test_class_0": 0.20,
                "test_class_1": 0.81,
                "test_class_2": 0.28,
                "test_class_3": 0.68,
                "test_balanced_accuracy": 0.4925,
            }
            adapted_row = {
                "mode": "semantic_d",
                "val_class_0": 0.22,
                "val_class_1": 0.79,
                "val_class_2": 0.30,
                "val_class_3": 0.65,
                "val_balanced_accuracy": 0.49,
                "test_class_0": 0.16,
                "test_class_1": 0.81,
                "test_class_2": 0.24,
                "test_class_3": 0.60,
                "test_balanced_accuracy": 0.4525,
            }
            _write_one_row(reference, reference_row)
            _write_one_row(capability, capability_row)
            write_module_d_sbr_eval(
                str(adapted_sbr),
                module_d_sbr_rows(
                    reference_row=reference_row,
                    adapted_row=adapted_row,
                    hard_k=2,
                    nb_classes=4,
                    model_name="BIOT",
                    reference_name="fewshot_ref",
                    adapted_source="semantic_d",
                ),
            )

            build_semantic_boundary_summary(
                reference_csv=str(reference),
                capability_csv=str(capability),
                adapted_csv=str(adapted_sbr),
                output_csv=str(output),
                model_name="BIOT",
                reference_name="fewshot_ref",
                capability_name="full_anchor",
                hard_k=2,
                nb_classes=4,
            )

            val = _read_rows(output)[0]
            self.assertAlmostEqual(float(val["d_effect_sbr"]), 0.08)
            self.assertEqual(val["adapted_source"], str(adapted_sbr))


if __name__ == "__main__":
    unittest.main()
