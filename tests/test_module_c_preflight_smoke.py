import csv
import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - depends on the training env.
    torch = None
    nn = None

if torch is not None:
    import util.module_c_preflight_policy as module_c_preflight_policy
    from run_finetuning import (
        _make_module_c_preflight_loaders,
        _resolve_module_c_support_batch_limit,
    )
    from util.module_c_preflight_policy import (
        _run_support_pass,
        _validation_losses,
        capture_module_c_rng_state,
        restore_module_c_rng_state,
        run_module_c_preflight_selection,
    )


if torch is not None:
    class _MetadataDataset(Dataset):
        def __init__(self, samples, labels, subjects):
            self.samples = samples
            self.labels = labels
            self.data = [
                {"subject_id": str(subject), "label": int(label), "file": f"sample_{index}.pkl"}
                for index, (subject, label) in enumerate(zip(subjects, labels.tolist()))
            ]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.samples[index], self.labels[index]


    class _TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(4, 4)
            self.key = nn.Linear(4, 4)
            self.value = nn.Linear(4, 4)
            self.proj = nn.Linear(4, 4)


    class _TinyMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)


    class _TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _TinyAttention()
            self.mlp = _TinyMlp()


    class _TinyGram(nn.Module):
        def __init__(self, classes=3):
            super().__init__()
            self.input_channels = 4
            self.main_model = nn.Module()
            self.main_model.model = nn.Module()
            self.main_model.model.blocks = nn.ModuleList([_TinyBlock()])
            self.task_head = nn.Linear(4, classes)
            self.integer_state = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)

        def forward(self, samples):
            hidden = samples.mean(dim=-1)
            block = self.main_model.model.blocks[0]
            hidden = torch.tanh(block.attn.query(hidden) + block.attn.value(hidden))
            hidden = torch.tanh(block.mlp.fc1(hidden))
            hidden = block.mlp.fc2(hidden)
            return self.task_head(hidden)


    class _IdentityLogits(nn.Module):
        def forward(self, samples):
            return samples


    class _ScalarProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.0))

        def forward(self, samples):
            return self.scale.expand(samples.shape[0], 2)


    class _MeanOutputLoss(nn.Module):
        def forward(self, output, _targets):
            return output.mean()


def _args(output_dir, classes=3):
    return SimpleNamespace(
        task_mod="Classification",
        nb_classes=classes,
        norm_method="",
        mv_norm_value=0.01,
        model_name="Gram",
        module_c_candidates="B,D,E",
        module_b_sites="input",
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.2,
        lora_base_update="full",
        lora_train_head=True,
        lora_train_chan_conv=False,
        opt="adamw",
        lr=0.03,
        weight_decay=0.0,
        layer_decay=1.0,
        opt_eps=None,
        opt_betas=None,
        momentum=0.9,
        clip_grad=None,
        update_freq=1,
        module_e_mode="dynamic_pressure_gate",
        module_e_warmup_steps=0,
        module_e_pressure_beta=0.95,
        module_e_gate_temperature=1.0,
        module_e_gate_floor=0.2,
        module_e_scale_min=0.5,
        module_e_scale_max=1.5,
        module_e_diag_freq=1000,
        diag_freq=1000,
        module_c_preflight_train_batches=0,
        module_c_preflight_val_batches=0,
        output_dir=output_dir,
        dataset="tiny",
        subject_mod="fewshot",
        k_shot=1,
        seed=7,
        run_tag="tiny-c",
    )


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class ModuleCPreflightSmokeTests(unittest.TestCase):
    def _loaders(self, classes=3):
        torch.manual_seed(7)
        support_labels = torch.tensor(list(range(classes)) * 4)
        validation_labels = torch.tensor(list(range(classes)) * 6)
        support_samples = torch.randn(len(support_labels), 4, 5)
        validation_samples = torch.randn(len(validation_labels), 4, 5)
        subjects = [f"s{(index // classes) % 3}" for index in range(len(validation_labels))]
        support = DataLoader(TensorDataset(support_samples, support_labels), batch_size=3, shuffle=False)
        validation = DataLoader(
            _MetadataDataset(validation_samples, validation_labels, subjects),
            batch_size=3,
            shuffle=False,
        )
        return support, validation

    def test_matched_search_trains_head_and_measures_direct_validation_loss(self):
        support_loader, validation_loader = self._loaders(classes=3)

        with tempfile.TemporaryDirectory() as output_dir:
            args = _args(output_dir, classes=3)
            model = _TinyGram(classes=3)
            original_trainable_count = sum(
                int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
            )
            result = run_module_c_preflight_selection(
                args=args,
                model=model,
                data_loader_train=support_loader,
                data_loader_val=validation_loader,
                device=torch.device("cpu"),
            )

            self.assertTrue(result.decision.selected_modules)
            self.assertTrue(set(result.decision.selected_modules).issubset({"B", "D", "E"}))
            self.assertEqual(args.module_c_resolved_selected, ",".join(result.decision.selected_modules))
            self.assertGreater(result.head_anchor["parameter_delta_l2"], 0.0)
            self.assertEqual(result.head_anchor["support_passes"], 1)
            self.assertIn(tuple(), result.branch_traces)
            self.assertEqual(
                result.branch_traces[tuple()]["trainable_parameter_count"],
                original_trainable_count,
            )
            for subset, trace in result.branch_traces.items():
                expected_count = original_trainable_count + sum(
                    result.ownership.parameter_counts[action] for action in subset
                )
                self.assertEqual(trace["trainable_parameter_count"], expected_count)
            self.assertTrue(all((action,) in result.branch_traces for action in ("B", "D", "E")))
            self.assertEqual(
                set(result.branch_traces),
                {
                    (),
                    ("B",),
                    ("D",),
                    ("E",),
                    ("B", "D"),
                    ("B", "E"),
                    ("D", "E"),
                    ("B", "D", "E"),
                },
            )
            fingerprints = {trace["support_fingerprint"] for trace in result.branch_traces.values()}
            self.assertEqual(len(fingerprints), 1)
            self.assertTrue(
                all(trace["validation_loss_source"] == "direct_per_example_log_loss" for trace in result.branch_traces.values())
            )
            self.assertEqual(result.diagnostics_by_module["E"]["functional_diagnostics_used_for_ranking"], 0)
            self.assertTrue(os.path.exists(result.score_csv_path))
            self.assertTrue(os.path.exists(result.decision_json_path))

            with open(result.decision_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            serialized = json.dumps(payload).lower()
            self.assertIn("exhaustive", serialized)
            self.assertIn("validation_macro_log_loss", serialized)
            self.assertIn("conditional_contributions", serialized)
            self.assertIn("pair_interactions", serialized)
            self.assertNotIn("first_order", serialized)
            self.assertNotIn("forced_nonempty_least_harm", serialized)
            self.assertNotIn("subject", serialized)
            self.assertNotIn("holm", serialized)
            self.assertNotIn("retired_actions", serialized)
            self.assertNotIn("search_steps", serialized)
            self.assertEqual(payload["runtime"]["branch_count"], 8)
            self.assertEqual(payload["runtime"]["support_pass_count"], 9)
            self.assertEqual(payload["runtime"]["validation_pass_count"], 10)
            self.assertEqual(set(payload["branches"]), {"EMPTY", "B", "D", "E", "B+D", "B+E", "D+E", "B+D+E"})
            with open(result.score_csv_path, "r", encoding="utf-8", newline="") as handle:
                score_rows = list(csv.DictReader(handle))
            self.assertEqual(len(score_rows), 7)
            self.assertEqual(
                {row["branch"] for row in score_rows},
                {"B", "D", "E", "B+D", "B+E", "D+E", "B+D+E"},
            )
            self.assertEqual(payload["probe_training"]["lora_dropout"], 0.2)
            self.assertEqual(
                payload["probe_training"]["full_update_base_control"],
                "same_pre_injection_base_trainability_for_every_branch",
            )

    def test_preflight_loaders_match_formal_optimizer_geometry_and_epoch_zero_schedule(self):
        support_ids = torch.arange(11, dtype=torch.float32)
        support_samples = support_ids.view(-1, 1, 1).expand(-1, 4, 5).clone()
        support_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        validation_ids = torch.arange(8, dtype=torch.float32) + 100.0
        validation_samples = validation_ids.view(-1, 1, 1).expand(-1, 4, 5).clone()
        validation_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        validation_subjects = ["s0", "s0", "s0", "s1", "s1", "s1", "s2", "s2"]
        support_dataset = TensorDataset(support_samples, support_labels)
        validation_dataset = _MetadataDataset(
            validation_samples, validation_labels, validation_subjects
        )

        with tempfile.TemporaryDirectory() as output_dir:
            args = _args(output_dir, classes=3)
            args.batch_size = 3
            args.update_freq = 2
            args.num_workers = 0
            args.pin_mem = False
            args.module_c_preflight_train_batches = 99
            support_loader, validation_loader = _make_module_c_preflight_loaders(
                args, support_dataset, validation_dataset
            )

            self.assertIsInstance(support_loader.sampler, SequentialSampler)
            self.assertFalse(support_loader.drop_last)
            self.assertEqual(
                [batch[0][:, 0, 0].tolist() for batch in support_loader],
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0],
                ],
            )
            self.assertIsInstance(validation_loader.sampler, SequentialSampler)
            self.assertFalse(validation_loader.drop_last)
            self.assertEqual(
                [value for batch in validation_loader for value in batch[0][:, 0, 0].tolist()],
                validation_ids.tolist(),
            )
            self.assertEqual(_resolve_module_c_support_batch_limit(4, 0), 4)
            self.assertEqual(_resolve_module_c_support_batch_limit(4, 1), 1)
            self.assertEqual(_resolve_module_c_support_batch_limit(4, 99), 4)

            seen_support_ids = []

            def recording_support_pass(*call_args, **call_kwargs):
                batches = call_args[2]
                seen_support_ids.append(
                    tuple(
                        int(value)
                        for batch in batches
                        for value in batch[0][:, 0, 0].tolist()
                    )
                )
                return _run_support_pass(*call_args, **call_kwargs)

            with patch(
                "util.module_c_preflight_policy._run_support_pass",
                side_effect=recording_support_pass,
            ):
                result = run_module_c_preflight_selection(
                    args=args,
                    model=_TinyGram(classes=3),
                    data_loader_train=support_loader,
                    data_loader_val=validation_loader,
                    device=torch.device("cpu"),
                    num_training_steps_per_epoch=1,
                    lr_schedule_values=[0.0125],
                    wd_schedule_values=[0.025],
                )

            self.assertTrue(seen_support_ids)
            self.assertEqual(len(seen_support_ids), 9)
            self.assertEqual(set(seen_support_ids), {tuple(range(11))})
            expected_trace = [
                {
                    "optimizer_step": 0,
                    "lr_schedule_value": 0.0125,
                    "weight_decay_schedule_value": 0.025,
                },
                {
                    "optimizer_step": 1,
                    "lr_schedule_value": 0.0125,
                    "weight_decay_schedule_value": 0.025,
                },
            ]
            self.assertEqual(result.head_anchor["optimizer_schedule_trace"], expected_trace)
            self.assertEqual(
                {json.dumps(trace["optimizer_schedule_trace"], sort_keys=True)
                 for trace in result.branch_traces.values()},
                {json.dumps(expected_trace, sort_keys=True)},
            )
            self.assertTrue(
                all(trace["support_examples"] == 11 for trace in result.branch_traces.values())
            )
            self.assertTrue(
                all(trace["optimizer_steps"] == 2 for trace in result.branch_traces.values())
            )

            with open(result.decision_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            support_scope = payload["probe_training"]["support_loader"]
            self.assertEqual(support_scope["sampler_type"], "SequentialSampler")
            self.assertFalse(support_scope["drop_last"])
            self.assertEqual(support_scope["raw_dataset_size"], 11)
            self.assertEqual(support_scope["visible_example_count"], 11)
            self.assertAlmostEqual(support_scope["coverage_fraction"], 1.0)
            validation_scope = payload["probe_training"]["validation_loader"]
            self.assertEqual(validation_scope["sampler_type"], "SequentialSampler")
            self.assertFalse(validation_scope["drop_last"])
            self.assertEqual(validation_scope["raw_dataset_size"], 8)
            self.assertEqual(validation_scope["visible_example_count"], 8)
            self.assertEqual(validation_scope["coverage_fraction"], 1.0)

    def test_preflight_support_keeps_a_dataset_smaller_than_one_batch(self):
        args = SimpleNamespace(batch_size=3, num_workers=0, pin_mem=False)
        support = TensorDataset(torch.arange(2), torch.tensor([0, 1]))
        validation = TensorDataset(torch.arange(2), torch.tensor([0, 1]))

        support_loader, validation_loader = _make_module_c_preflight_loaders(
            args, support, validation
        )

        self.assertFalse(support_loader.drop_last)
        self.assertEqual([value for batch in support_loader for value in batch[0].tolist()], [0, 1])
        self.assertEqual([value for batch in validation_loader for value in batch[0].tolist()], [0, 1])

    def test_automatic_preflight_requires_the_complete_bde_registry(self):
        support_loader, validation_loader = self._loaders(classes=3)
        with tempfile.TemporaryDirectory() as output_dir:
            args = _args(output_dir, classes=3)
            args.module_c_candidates = "B,D"

            with self.assertRaisesRegex(ValueError, "exactly.*B.*D.*E"):
                run_module_c_preflight_selection(
                    args=args,
                    model=_TinyGram(classes=3),
                    data_loader_train=support_loader,
                    data_loader_val=validation_loader,
                    device=torch.device("cpu"),
                )

    def test_validation_metrics_report_macro_and_micro_multiclass_log_loss(self):
        logits = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
        labels = torch.tensor([0, 0, 0, 1])
        batches = list(DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False))
        args = _args("", classes=2)

        losses, observed, per_class, macro_loss, micro_loss = _validation_losses(
            args, _IdentityLogits(), batches, torch.device("cpu")
        )
        expected = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

        self.assertEqual(observed, tuple(labels.tolist()))
        self.assertTrue(torch.allclose(torch.tensor(losses), expected))
        self.assertAlmostEqual(per_class[0], float(expected[:3].mean()))
        self.assertAlmostEqual(per_class[1], float(expected[3]))
        self.assertAlmostEqual(macro_loss, float((expected[:3].mean() + expected[3]) / 2.0))
        self.assertAlmostEqual(micro_loss, float(expected.mean()))
        self.assertNotAlmostEqual(macro_loss, micro_loss)

    def test_validation_metrics_report_macro_and_micro_binary_log_loss(self):
        logits = torch.tensor([[-2.0], [-1.0], [0.0], [2.0]])
        labels = torch.tensor([0, 0, 0, 1])
        batches = list(DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False))
        args = _args("", classes=1)

        losses, observed, per_class, macro_loss, micro_loss = _validation_losses(
            args, _IdentityLogits(), batches, torch.device("cpu")
        )
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0], labels.float(), reduction="none"
        )

        self.assertEqual(observed, tuple(labels.tolist()))
        self.assertTrue(torch.allclose(torch.tensor(losses), expected))
        self.assertAlmostEqual(per_class[0], float(expected[:3].mean()))
        self.assertAlmostEqual(per_class[1], float(expected[3]))
        self.assertAlmostEqual(macro_loss, float((expected[:3].mean() + expected[3]) / 2.0))
        self.assertAlmostEqual(micro_loss, float(expected.mean()))
        self.assertNotAlmostEqual(macro_loss, micro_loss)

    def test_partial_accumulation_tail_uses_its_actual_microbatch_count(self):
        args = _args("", classes=2)
        args.lr = 0.1
        args.update_freq = 2
        model = _ScalarProbe()
        batches = [
            (torch.zeros(1, 1), torch.tensor([0])),
            (torch.zeros(1, 1), torch.tensor([1])),
            (torch.zeros(1, 1), torch.tensor([0])),
        ]

        with patch(
            "util.module_c_preflight_policy._create_probe_optimizer",
            return_value=torch.optim.SGD(model.parameters(), lr=0.1),
        ):
            _run_support_pass(
                args,
                model,
                batches,
                torch.device("cpu"),
                _MeanOutputLoss(),
                lr_schedule_values=[0.1],
                wd_schedule_values=[0.0],
                formal_epoch_zero_steps=1,
            )

        self.assertAlmostEqual(float(model.scale.detach()), -0.2, places=6)

    def test_e_controller_attaches_only_to_the_four_e_containing_branches(self):
        support_loader, validation_loader = self._loaders(classes=3)
        original_attach = module_c_preflight_policy.attach_module_e_dynamic_pressure_controller
        attached_models = []

        def recording_attach(args, model):
            attached_models.append(model)
            return original_attach(args, model)

        with tempfile.TemporaryDirectory() as output_dir, patch(
            "util.module_c_preflight_policy.attach_module_e_dynamic_pressure_controller",
            side_effect=recording_attach,
        ):
            model = _TinyGram(classes=3)
            result = run_module_c_preflight_selection(
                args=_args(output_dir, classes=3),
                model=model,
                data_loader_train=support_loader,
                data_loader_val=validation_loader,
                device=torch.device("cpu"),
            )

        self.assertEqual(len(attached_models), 4)
        self.assertEqual(len(result.branch_traces), 8)
        self.assertFalse(hasattr(model, "_module_e_dynamic_pressure_controller"))

    def test_binary_bce_search_uses_both_labels(self):
        support_loader, validation_loader = self._loaders(classes=2)

        with tempfile.TemporaryDirectory() as output_dir:
            args = _args(output_dir, classes=1)
            args.dataset = "tiny_binary"
            result = run_module_c_preflight_selection(
                args=args,
                model=_TinyGram(classes=1),
                data_loader_train=support_loader,
                data_loader_val=validation_loader,
                device=torch.device("cpu"),
            )

            self.assertTrue(result.decision.selected_modules)
            observed_classes = set(result.branch_traces[("B",)]["validation_per_class_loss"])
            self.assertEqual(observed_classes, {0, 1})

    def test_validation_does_not_require_subject_metadata(self):
        support_loader, _ = self._loaders(classes=3)
        samples = torch.randn(9, 4, 5)
        labels = torch.tensor([0, 1, 2] * 3)
        validation_loader = DataLoader(TensorDataset(samples, labels), batch_size=3, shuffle=False)

        with tempfile.TemporaryDirectory() as output_dir:
            result = run_module_c_preflight_selection(
                args=_args(output_dir, classes=3),
                model=_TinyGram(classes=3),
                data_loader_train=support_loader,
                data_loader_val=validation_loader,
                device=torch.device("cpu"),
            )
            self.assertEqual(len(result.branch_traces), 8)

    def test_rng_snapshot_restores_python_numpy_and_torch(self):
        random.seed(19)
        torch.manual_seed(19)
        np_state = __import__("numpy").random
        np_state.seed(19)
        state = capture_module_c_rng_state()
        expected = (random.random(), float(np_state.random()), float(torch.rand(1).item()))

        random.random()
        np_state.random()
        torch.rand(4)
        restore_module_c_rng_state(state)
        observed = (random.random(), float(np_state.random()), float(torch.rand(1).item()))

        self.assertEqual(observed, expected)

    def test_unmirrored_training_controls_fail_instead_of_silently_changing_the_probe(self):
        support_loader, validation_loader = self._loaders(classes=3)
        with tempfile.TemporaryDirectory() as output_dir:
            args = _args(output_dir, classes=3)
            args.lora_delta_lambda = 0.1
            with self.assertRaisesRegex(ValueError, "lora_delta_lambda"):
                run_module_c_preflight_selection(
                    args=args,
                    model=_TinyGram(classes=3),
                    data_loader_train=support_loader,
                    data_loader_val=validation_loader,
                    device=torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
