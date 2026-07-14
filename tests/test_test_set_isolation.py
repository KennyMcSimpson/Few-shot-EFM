import ast
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def _epoch_loop(function):
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        if isinstance(node.target, ast.Name) and node.target.id == "epoch":
            return node
    raise AssertionError(f"epoch loop not found in {function.name}")


def _loaded_names(node):
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def _load_pure_function(tree, name, namespace=None):
    function = _function(tree, name)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {} if namespace is None else dict(namespace)
    exec(compile(module, filename=f"<{name}>", mode="exec"), namespace)
    return namespace[name]


class TestSetIsolationTests(unittest.TestCase):
    def test_classification_regression_epoch_loop_never_loads_test_data(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        loop = _epoch_loop(_function(tree, "main"))

        self.assertNotIn("data_loader_test", _loaded_names(loop))

    def test_retrieval_epoch_loop_never_loads_or_selects_on_test_data(self):
        tree = ast.parse(
            (ROOT / "engine_for_finetuning.py").read_text(encoding="utf-8")
        )
        loop = _epoch_loop(_function(tree, "main_train_loop"))
        names = _loaded_names(loop)

        self.assertNotIn("test_dataloader", names)
        self.assertNotIn("test_accuracy", names)
        self.assertNotIn("test_loss", names)

    def test_adapter_calibration_candidates_never_load_test_data(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        function = _function(tree, "_run_adapter_strength_calibration")
        candidate_loop = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "cand"
        )

        self.assertNotIn("data_loader_test", _loaded_names(candidate_loop))

    def test_evaluate_is_gradient_free(self):
        tree = ast.parse(
            (ROOT / "engine_for_finetuning.py").read_text(encoding="utf-8")
        )
        function = _function(tree, "evaluate")
        decorators = [ast.unparse(node) for node in function.decorator_list]

        self.assertIn("torch.no_grad()", decorators)
        called = {
            ast.unparse(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        }
        self.assertNotIn("loss.backward", called)
        self.assertNotIn("optimizer.step", called)

    def test_distributed_evaluate_gathers_predictions_before_metrics(self):
        tree = ast.parse(
            (ROOT / "engine_for_finetuning.py").read_text(encoding="utf-8")
        )
        function = _function(tree, "evaluate")
        called = {
            ast.unparse(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        }

        self.assertIn("torch.distributed.all_gather_object", called)

    def test_snapshot_selection_rejects_non_validation_metrics(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        metric_column = _load_pure_function(tree, "_snapshot_metric_column")

        self.assertEqual(metric_column("balanced_accuracy"), "val_balanced_accuracy")
        self.assertEqual(metric_column("val_balanced_accuracy"), "val_balanced_accuracy")
        for forbidden in (
            "test_balanced_accuracy",
            "train_eval_balanced_accuracy",
            "train_balanced_accuracy",
        ):
            with self.subTest(metric=forbidden):
                with self.assertRaises(ValueError):
                    metric_column(forbidden)

    def test_missing_selection_metric_never_falls_back_to_accuracy(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_metric = _load_pure_function(tree, "_select_metric")

        with self.assertRaises(KeyError):
            select_metric(
                {"accuracy": 0.99, "balanced_accuracy": 0.50},
                "selection_bacc_worst_std",
            )

    def test_only_one_final_test_protocol_can_be_enabled(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        try:
            select_protocol = _load_pure_function(tree, "_select_final_test_protocol")
        except AssertionError as exc:
            self.fail(str(exc))

        class Args:
            snapshot_eval = True
            boundary_anchor_eval = True
            adaptive_swa_eval = False
            proto_eval = False

        with self.assertRaises(ValueError):
            select_protocol(Args())

    def test_final_protocol_is_resolved_after_dataset_class_count(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        main = _function(tree, "main")
        calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
        ]
        protocol_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "_select_final_test_protocol"
        )
        dataset_line = min(
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name)
            and node.func.id == "get_datasets"
        )

        self.assertGreater(protocol_line, dataset_line)

    def test_special_final_protocols_require_multiclass_classification(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_protocol = _load_pure_function(tree, "_select_final_test_protocol")

        class Args:
            snapshot_eval = False
            boundary_anchor_eval = False
            adaptive_swa_eval = False
            proto_eval = True
            adapter_calib_eval = False
            task_mod = "Regression"
            nb_classes = 1

        with self.assertRaises(ValueError):
            select_protocol(Args())

    def test_binary_classification_allows_adaptive_swa_only(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_protocol = _load_pure_function(tree, "_select_final_test_protocol")

        class BinaryAdaptiveSwaArgs:
            snapshot_eval = False
            boundary_anchor_eval = False
            adaptive_swa_eval = True
            proto_eval = False
            adapter_calib_eval = False
            task_mod = "Classification"
            nb_classes = 1
            start_epoch = 0
            distributed = False

        self.assertEqual(select_protocol(BinaryAdaptiveSwaArgs()), "adaptive_swa")

        class BinarySnapshotArgs(BinaryAdaptiveSwaArgs):
            snapshot_eval = True
            adaptive_swa_eval = False
            monitor_dynamics = True

        with self.assertRaises(ValueError):
            select_protocol(BinarySnapshotArgs())

    def test_snapshot_protocol_requires_epoch_monitoring(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_protocol = _load_pure_function(tree, "_select_final_test_protocol")

        class Args:
            snapshot_eval = True
            boundary_anchor_eval = False
            adaptive_swa_eval = False
            proto_eval = False
            adapter_calib_eval = False
            task_mod = "Classification"
            nb_classes = 3
            monitor_dynamics = False

        with self.assertRaises(ValueError):
            select_protocol(Args())

    def test_resume_requires_and_loads_validation_best_checkpoint(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        try:
            load_best = _load_pure_function(
                tree,
                "_load_validation_best_checkpoint",
                namespace={"Path": Path, "torch": torch},
            )
        except AssertionError as exc:
            self.fail(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            class Args:
                output_dir = tmp
                start_epoch = 4
                task_mod = "Classification"
                best_metric = "balanced_accuracy"

            with self.assertRaises(FileNotFoundError):
                load_best(Args())

            torch.save(
                {
                    "model": {"weight": torch.tensor([1.0])},
                    "selected_epoch": 3,
                    "selected_by": "validation",
                    "selection_metric": "balanced_accuracy",
                },
                Path(tmp) / "checkpoint-best.pth",
            )
            state, selected_epoch = load_best(Args())

            self.assertEqual(selected_epoch, 3)
            self.assertTrue(torch.equal(state["weight"], torch.tensor([1.0])))

            torch.save(
                {
                    "model": {"weight": torch.tensor([4.0])},
                    "selected_epoch": 3,
                    "selected_by": "validation",
                    "selection_metric": "accuracy",
                },
                Path(tmp) / "checkpoint-best.pth",
            )
            with self.assertRaises(RuntimeError):
                load_best(Args())

            torch.save(
                {"model": {"weight": torch.tensor([2.0])}, "epoch": "best"},
                Path(tmp) / "checkpoint-best.pth",
            )
            with self.assertRaises(RuntimeError):
                load_best(Args())

    def test_in_memory_adaptive_swa_cannot_resume_mid_protocol(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_protocol = _load_pure_function(tree, "_select_final_test_protocol")

        class Args:
            snapshot_eval = False
            boundary_anchor_eval = False
            adaptive_swa_eval = True
            proto_eval = False
            adapter_calib_eval = False
            task_mod = "Classification"
            nb_classes = 3
            monitor_dynamics = False
            start_epoch = 2

        with self.assertRaises(ValueError):
            select_protocol(Args())

    def test_special_final_protocols_fail_closed_under_distributed_training(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_protocol = _load_pure_function(tree, "_select_final_test_protocol")

        class Args:
            snapshot_eval = False
            boundary_anchor_eval = True
            adaptive_swa_eval = False
            proto_eval = False
            adapter_calib_eval = False
            task_mod = "Classification"
            nb_classes = 3
            monitor_dynamics = False
            start_epoch = 0
            distributed = True

        with self.assertRaises(ValueError):
            select_protocol(Args())

    def test_retrieval_final_suite_fails_closed_under_distributed_training(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_protocol = _load_pure_function(tree, "_select_final_test_protocol")

        class Args:
            snapshot_eval = False
            boundary_anchor_eval = False
            adaptive_swa_eval = False
            proto_eval = False
            adapter_calib_eval = False
            task_mod = "Retrieval"
            nb_classes = 0
            monitor_dynamics = False
            start_epoch = 0
            distributed = True

        with self.assertRaises(ValueError):
            select_protocol(Args())

    def test_temporary_validation_state_protocols_cannot_resume(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        select_protocol = _load_pure_function(tree, "_select_final_test_protocol")

        base = {
            "snapshot_eval": False,
            "boundary_anchor_eval": False,
            "adaptive_swa_eval": False,
            "proto_eval": False,
            "adapter_calib_eval": False,
            "task_mod": "Classification",
            "nb_classes": 3,
            "monitor_dynamics": False,
            "start_epoch": 2,
            "distributed": False,
            "adapter_swa": False,
            "adapter_swa_eval": False,
            "cbra_eval_front_beta": 1.0,
        }
        for override in (
            {"adapter_swa": True},
            {"adapter_swa_eval": True},
            {"cbra_eval_front_beta": 0.5},
        ):
            with self.subTest(override=override):
                attrs = dict(base)
                attrs.update(override)
                Args = type("Args", (), attrs)
                with self.assertRaises(ValueError):
                    select_protocol(Args())

    def test_main_fails_closed_for_deepspeed_or_multiprocess_launch(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))

        class FakeDistributed:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_world_size():
                return 2

        FakeTorch = type("FakeTorch", (), {"distributed": FakeDistributed})
        try:
            ensure_single = _load_pure_function(
                tree,
                "_ensure_single_process_protocol",
                namespace={"os": os, "torch": FakeTorch},
            )
        except AssertionError as exc:
            self.fail(str(exc))

        for attrs in (
            {"enable_deepspeed": True, "distributed": False},
            {"enable_deepspeed": False, "distributed": True},
        ):
            with self.subTest(attrs=attrs):
                Args = type("Args", (), attrs)
                with self.assertRaises(RuntimeError):
                    ensure_single(Args())
        Args = type(
            "Args",
            (),
            {"enable_deepspeed": False, "distributed": False},
        )
        with self.assertRaises(RuntimeError):
            ensure_single(Args())

    def test_module_e_resume_fails_without_persisted_controller_state(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        try:
            ensure_resume = _load_pure_function(
                tree, "_ensure_resume_controller_state"
            )
        except AssertionError as exc:
            self.fail(str(exc))

        Args = type("Args", (), {"start_epoch": 2})
        with self.assertRaises(RuntimeError):
            ensure_resume(Args(), object())
        ensure_resume(Args(), None)

    def test_enabled_final_protocol_requires_a_real_result(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        try:
            require_result = _load_pure_function(
                tree, "_require_final_test_result"
            )
        except AssertionError as exc:
            self.fail(str(exc))

        with self.assertRaises(RuntimeError):
            require_result("adaptive_swa", None, True)
        self.assertEqual(
            require_result("standard", {"test_accuracy": 0.5}, True),
            {"test_accuracy": 0.5},
        )

    def test_new_run_clears_stale_final_test_artifact(self):
        try:
            from run_finetuning import _prepare_final_test_artifact
        except ImportError as exc:
            self.fail(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            class Args:
                output_dir = tmp

            path = Path(tmp) / "diagnostics" / "final_test_metrics.json"
            path.parent.mkdir()
            path.write_text('{"stale": true}', encoding="utf-8")
            _prepare_final_test_artifact(Args())
            self.assertFalse(path.exists())

    def test_snapshot_candidate_artifact_has_no_test_columns(self):
        from run_finetuning import _write_snapshot_candidates

        ranked_rows = [
            {
                "epoch": 1,
                "score": 0.5,
                "row": {"val_balanced_accuracy": 0.5},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            _write_snapshot_candidates(
                tmp, ranked_rows, "val_balanced_accuracy"
            )
            with (Path(tmp) / "snapshot_candidates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                fieldnames = csv.DictReader(handle).fieldnames
        self.assertFalse(any(name.startswith("test_") for name in fieldnames))

    def test_final_checkpoint_loaders_are_topology_strict(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        for function_name in (
            "_load_epoch_checkpoint_into_model",
            "_load_epoch_state_with_front_beta",
            "_run_boundary_anchor_final_eval",
        ):
            with self.subTest(function=function_name):
                function = _function(tree, function_name)
                load_calls = [
                    node
                    for node in ast.walk(function)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load_state_dict"
                ]
                self.assertTrue(load_calls)
                for call in load_calls:
                    strict = next(
                        (
                            keyword.value
                            for keyword in call.keywords
                            if keyword.arg == "strict"
                        ),
                        None,
                    )
                    self.assertIsInstance(strict, ast.Constant)
                    self.assertIs(strict.value, True)

    def test_boundary_anchor_never_falls_back_to_a_different_epoch(self):
        tree = ast.parse((ROOT / "run_finetuning.py").read_text(encoding="utf-8"))
        load_anchor = _load_pure_function(
            tree,
            "_load_anchor_epoch_state",
            namespace={"os": os, "torch": torch},
        )

        with tempfile.TemporaryDirectory() as tmp:
            class Args:
                output_dir = tmp
                boundary_anchor_tag = "boundary_anchor"

            ckpt_dir = Path(tmp) / "monitor_checkpoints"
            ckpt_dir.mkdir()
            torch.save(
                {"model": {"weight": torch.tensor([9.0])}, "epoch": 9},
                ckpt_dir / "boundary_anchor.pth",
            )
            with self.assertRaises(FileNotFoundError):
                load_anchor(Args(), 3)

            exact = ckpt_dir / "boundary_epoch_003.pth"
            torch.save(
                {"model": {"weight": torch.tensor([3.0])}, "epoch": 3},
                exact,
            )
            obj, path = load_anchor(Args(), 3)
            self.assertEqual(obj["epoch"], 3)
            self.assertEqual(Path(path), exact)

    def test_split_integrity_never_exposes_test_label_counts(self):
        from util.fb_probe import save_split_integrity

        with tempfile.TemporaryDirectory() as tmp:
            split_root = Path(tmp) / "splits"
            split_root.mkdir()
            for split, labels in {
                "train": [0, 1],
                "val": [1, 1],
                "test": [7, 7],
            }.items():
                records = [
                    {"path": f"{split}_{idx}.npy", "label": label}
                    for idx, label in enumerate(labels)
                ]
                (split_root / f"{split}.json").write_text(
                    json.dumps(records), encoding="utf-8"
                )

            class Args:
                fb_enable = True
                fb_split_check = True
                fb_integrity_max_json_records = 100
                output_dir = str(Path(tmp) / "output")
                subject_mod = "fewshot"

            save_split_integrity(
                Args(),
                dataset_info={"root": {"fewshot": str(split_root)}},
            )
            diag = Path(Args.output_dir) / "diagnostics"
            with (diag / "fb_class_counts.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                class_rows = list(csv.DictReader(handle))
            self.assertNotIn("test", {row["split"] for row in class_rows})

            with (diag / "fb_split_integrity.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                split_rows = {
                    row["split"]: row for row in csv.DictReader(handle)
                }
            self.assertEqual(split_rows["test"]["json_label_count_total"], "")
            self.assertEqual(
                split_rows["test"]["class_count_summary"],
                "withheld_until_final_evaluation",
            )


if __name__ == "__main__":
    unittest.main()
