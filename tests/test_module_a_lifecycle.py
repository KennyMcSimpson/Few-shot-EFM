import argparse
import tempfile
import unittest

import numpy as np
import torch

import module_a_lifecycle as lifecycle


def _default_args(*extra):
    parser = argparse.ArgumentParser()
    lifecycle.add_lifecycle_window_args(parser)
    return parser.parse_args(list(extra))


class ModuleALifecycleSafetyTests(unittest.TestCase):
    def test_parser_defaults_to_generic_metric_without_hard_classes(self):
        args = _default_args()

        self.assertEqual(args.adaptive_swa_select_metric, "selection_bacc_worst_std")
        self.assertEqual(args.adaptive_swa_hard_classes, "")

    def test_default_score_is_invariant_to_class_label_permutation(self):
        args = _default_args()
        base_stats = {
            "balanced_accuracy": 0.55,
            "worst_class_recall": 0.10,
            "recall_std": 0.20,
        }
        first_stats = dict(base_stats)
        permuted_stats = dict(base_stats)
        lifecycle._add_selection_metrics(
            first_stats,
            {"per_class_recall": np.asarray([0.10, 0.50, 0.70, 0.40])},
            args,
        )
        lifecycle._add_selection_metrics(
            permuted_stats,
            {"per_class_recall": np.asarray([0.50, 0.10, 0.40, 0.70])},
            args,
        )

        first_score, first_report = lifecycle._adaptive_swa_metric(
            first_stats, None, args, start_epoch=1, end_epoch=1, length=1
        )
        permuted_score, permuted_report = lifecycle._adaptive_swa_metric(
            permuted_stats, None, args, start_epoch=1, end_epoch=1, length=1
        )

        self.assertEqual(first_report["raw_metric_name"], "selection_bacc_worst_std")
        self.assertEqual(permuted_report["raw_metric_name"], "selection_bacc_worst_std")
        self.assertAlmostEqual(first_score, permuted_score)

    def test_empty_hard_classes_disable_hard_score_tie_and_report_influence(self):
        args = _default_args(
            "--adaptive_swa_balance_lambda",
            "10",
            "--adaptive_swa_hard_floor",
            "0.9",
            "--adaptive_swa_hard_floor_lambda",
            "10",
            "--adaptive_swa_tie_mode",
            "hard_stable",
        )
        stats = {
            "balanced_accuracy": 0.50,
            "worst_class_recall": 0.10,
            "recall_std": 0.20,
            "selection_bacc_worst_std": 0.505,
        }
        details = {"per_class_recall": np.asarray([0.10, 0.80, 0.30])}

        score, report = lifecycle._adaptive_swa_metric(
            stats, details, args, start_epoch=1, end_epoch=1, length=1
        )

        self.assertAlmostEqual(score, stats["selection_bacc_worst_std"])
        for field in (
            "hard_imbalance",
            "hard_imbalance_penalty",
            "hard_min",
            "hard_max",
            "hard_floor",
            "hard_floor_penalty",
        ):
            self.assertTrue(field not in report or report[field] == "", field)

        later_with_better_hard_value = {
            "score": score,
            "start_epoch": 2,
            "end_epoch": 2,
            "length": 1,
            "hard_min": 0.9,
        }
        earlier_with_worse_hard_value = {
            "score": score,
            "start_epoch": 1,
            "end_epoch": 1,
            "length": 1,
            "hard_min": 0.1,
        }
        self.assertFalse(
            lifecycle._prefer_adaptive_swa_candidate(
                later_with_better_hard_value,
                earlier_with_worse_hard_value,
                tie_eps=0.01,
                args=args,
            )
        )

        window_details = {
            "y_true": np.asarray([0, 1, 2]),
            "y_pred": np.asarray([0, 1, 0]),
        }
        final_details = {"y_pred": np.asarray([0, 2, 2])}
        args.nb_classes = 3
        summary, class_rows = lifecycle._adaptive_swa_forgetting_rows(
            window_details, final_details, args, 1, 1, 1
        )
        self.assertEqual(summary["hard_classes"], "")
        for field in (
            "hard_total_count",
            "hard_retained_count",
            "hard_forgotten_rate",
            "hard_trajectory_forgetting_asymmetry",
        ):
            self.assertTrue(field not in summary or summary[field] == "", field)
        self.assertTrue(
            all(
                "is_hard_class" not in row or row["is_hard_class"] == ""
                for row in class_rows
            )
        )

    def test_explicit_legacy_min02_and_hard_classes_keep_previous_behavior(self):
        args = _default_args(
            "--adaptive_swa_select_metric",
            "selection_bacc_min02_std",
            "--adaptive_swa_hard_classes",
            "0,2",
            "--adaptive_swa_balance_lambda",
            "0.5",
            "--adaptive_swa_hard_floor",
            "0.4",
            "--adaptive_swa_hard_floor_lambda",
            "2",
            "--adaptive_swa_tie_mode",
            "hard_stable",
        )
        stats = {
            "balanced_accuracy": 0.60,
            "worst_class_recall": 0.20,
            "recall_std": 0.20,
        }
        details = {"per_class_recall": np.asarray([0.20, 0.70, 0.40])}
        lifecycle._add_selection_metrics(stats, details, args)

        score, report = lifecycle._adaptive_swa_metric(
            stats, details, args, start_epoch=1, end_epoch=1, length=1
        )

        self.assertAlmostEqual(report["raw_metric"], 0.63)
        self.assertAlmostEqual(report["hard_imbalance"], 0.20)
        self.assertAlmostEqual(report["hard_imbalance_penalty"], 0.10)
        self.assertAlmostEqual(report["hard_floor_penalty"], 0.40)
        self.assertAlmostEqual(score, 0.13)

        later_with_better_hard_value = {
            "score": score,
            "start_epoch": 2,
            "end_epoch": 2,
            "length": 1,
            "hard_min": 0.5,
        }
        earlier_with_worse_hard_value = {
            "score": score,
            "start_epoch": 1,
            "end_epoch": 1,
            "length": 1,
            "hard_min": 0.1,
        }
        self.assertTrue(
            lifecycle._prefer_adaptive_swa_candidate(
                later_with_better_hard_value,
                earlier_with_worse_hard_value,
                tie_eps=0.01,
                args=args,
            )
        )


    def test_binary_forgetting_rows_cover_both_real_labels(self):
        args = _default_args()
        args.nb_classes = 1
        window_details = {
            "y_true": np.asarray([0, 0, 1, 1]),
            "y_pred": np.asarray([0, 1, 1, 0]),
        }
        final_details = {"y_pred": np.asarray([0, 0, 1, 1])}

        summary, class_rows = lifecycle._adaptive_swa_forgetting_rows(
            window_details, final_details, args, 1, 1, 1
        )

        self.assertEqual(summary["total_count"], 4)
        self.assertEqual([row["class_id"] for row in class_rows], [0, 1])
        self.assertEqual([row["support"] for row in class_rows], [2, 2])

    def test_binary_adaptive_swa_runs_validation_selection_and_one_test(self):
        args = _default_args(
            "--adaptive_swa_eval",
            "--adaptive_swa_min_len",
            "1",
            "--adaptive_swa_max_len",
            "1",
            "--adaptive_swa_epoch_max",
            "1",
            "--adaptive_swa_no_save_selected_ckpt",
        )
        args.task_mod = "Classification"
        args.nb_classes = 1
        args.epochs = 1
        args.lora_target = "module_c"
        args.fb_recipe = ""
        args.eval_logit_adjust = False

        model = torch.nn.Linear(1, 1)
        snapshots = [
            lifecycle.capture_lifecycle_snapshot(
                model, epoch_id=1, trainable_names={"weight", "bias"}
            )
        ]
        calls = []

        def fake_evaluate(
            _args,
            loader,
            _model,
            _device,
            header,
            ch_names,
            metrics,
            return_details,
            logit_bias=None,
        ):
            del _args, _model, _device, ch_names, metrics, logit_bias
            calls.append((loader, header))
            return (
                {
                    "accuracy": 0.75,
                    "balanced_accuracy": 0.75,
                    "worst_class_recall": 0.50,
                    "recall_std": 0.25,
                    "pr_auc": 0.80,
                    "roc_auc": 0.85,
                },
                {
                    "per_class_recall": np.asarray([0.50, 1.00]),
                    "y_true": np.asarray([0, 0, 1, 1]),
                    "y_pred": np.asarray([0, 1, 1, 1]),
                },
            )

        val_loader = object()
        test_loader = object()
        with tempfile.TemporaryDirectory() as tmp:
            args.output_dir = tmp
            result = lifecycle.run_lifecycle_window_search(
                args=args,
                model=model,
                data_loader_val=val_loader,
                data_loader_test=test_loader,
                device=torch.device("cpu"),
                metrics=["accuracy", "balanced_accuracy", "pr_auc", "roc_auc"],
                snapshots=snapshots,
                evaluate_fn=fake_evaluate,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["test_used_for_selection"], 0)
        self.assertEqual(sum(loader is test_loader for loader, _ in calls), 1)


if __name__ == "__main__":
    unittest.main()
