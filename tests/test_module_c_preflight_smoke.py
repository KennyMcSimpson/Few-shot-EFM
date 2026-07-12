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
    from torch.utils.data import DataLoader, Dataset, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - depends on the training env.
    torch = None
    nn = None

if torch is not None:
    from util.module_c_preflight_policy import (
        capture_module_c_rng_state,
        restore_module_c_rng_state,
        run_module_c_preflight_selection,
    )
    from util.module_c_risk_policy import SearchDecision


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
            self.main_model.blocks = nn.ModuleList([_TinyBlock()])
            self.task_head = nn.Linear(4, classes)
            self.integer_state = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)

        def forward(self, samples):
            hidden = samples.mean(dim=-1)
            block = self.main_model.blocks[0]
            hidden = torch.tanh(block.attn.query(hidden) + block.attn.value(hidden))
            hidden = torch.tanh(block.mlp.fc1(hidden))
            hidden = block.mlp.fc2(hidden)
            return self.task_head(hidden)


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
            self.assertTrue(any(len(subset) == 2 for subset in result.branch_traces))
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
            self.assertIn("paired_validation_log_loss", serialized)
            self.assertNotIn("first_order", serialized)
            self.assertNotIn("forced_nonempty_least_harm", serialized)
            self.assertEqual(payload["probe_training"]["lora_dropout"], 0.2)
            self.assertEqual(
                payload["probe_training"]["full_update_base_control"],
                "same_pre_injection_base_trainability_for_every_branch",
            )

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
            primary_step = result.decision.search_steps[0]
            observed_classes = set(primary_step["trial_diagnostics"]["B"]["class_gain"])
            self.assertEqual(observed_classes, {0, 1})

    def test_dead_primary_path_compares_the_alternative_pair_without_building_the_triple(self):
        support_loader, validation_loader = self._loaders(classes=3)
        choose_calls = []

        def scripted_choice(trials, require_nonempty, alpha):
            del alpha
            trials = tuple(trials)
            subsets = {tuple(trial.candidate_subset) for trial in trials}
            choose_calls.append(subsets)
            if len(choose_calls) == 1:
                self.assertTrue(require_nonempty)
                selected = next(trial for trial in trials if trial.candidate_subset == ("B",))
                return SearchDecision(selected, "nonempty_weak_best_observed_gain", "weak", {})
            if len(choose_calls) == 2:
                self.assertFalse(require_nonempty)
                self.assertEqual(subsets, {("B", "D"), ("B", "E")})
                return SearchDecision(None, "no_supported_safe_gain", "none", {})
            if len(choose_calls) == 3:
                self.assertFalse(require_nonempty)
                self.assertEqual(subsets, {("D", "E")})
                selected = next(iter(trials))
                return SearchDecision(selected, "supported_alternative_pair_gain", "supported", {})
            self.fail(f"Unexpected additional search stage: {subsets}")

        with tempfile.TemporaryDirectory() as output_dir:
            with patch("util.module_c_preflight_policy.choose_action", side_effect=scripted_choice):
                result = run_module_c_preflight_selection(
                    args=_args(output_dir, classes=3),
                    model=_TinyGram(classes=3),
                    data_loader_train=support_loader,
                    data_loader_val=validation_loader,
                    device=torch.device("cpu"),
                )

            self.assertEqual(result.decision.selected_modules, ("D", "E"))
            self.assertEqual(result.decision.evidence_strength, "supported")
            self.assertNotIn(("B", "D", "E"), result.branch_traces)
            self.assertEqual(
                [step["stage"] for step in result.decision.search_steps],
                ["primary", "conditional_addition", "alternative_pair_rescue"],
            )
            self.assertEqual(
                result.decision.search_steps[-1]["reason"],
                "supported_alternative_pair_gain",
            )
            with open(result.decision_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(
                payload["search"]["strategy"],
                "hierarchical_forward_addition_and_alternative_pair_rescue",
            )
            self.assertEqual(payload["primary_evidence_strength"], "weak")
            self.assertEqual(payload["final_evidence_strength"], "supported")
            self.assertEqual(payload["search"]["retired_actions"], ["B"])
            self.assertNotIn("floating_deletion", json.dumps(payload).lower())

    def test_supported_forward_chain_can_reach_all_three_actions(self):
        support_loader, validation_loader = self._loaders(classes=3)
        choose_calls = []

        def scripted_choice(trials, require_nonempty, alpha):
            del alpha
            trials = tuple(trials)
            subsets = {tuple(trial.candidate_subset) for trial in trials}
            choose_calls.append(subsets)
            if len(choose_calls) == 1:
                self.assertTrue(require_nonempty)
                selected = next(trial for trial in trials if trial.candidate_subset == ("B",))
                return SearchDecision(selected, "supported_primary_gain", "supported", {})
            if len(choose_calls) == 2:
                self.assertFalse(require_nonempty)
                self.assertEqual(subsets, {("B", "D"), ("B", "E")})
                selected = next(trial for trial in trials if trial.candidate_subset == ("B", "D"))
                return SearchDecision(selected, "supported_conditional_gain", "supported", {})
            if len(choose_calls) == 3:
                self.assertFalse(require_nonempty)
                self.assertEqual(subsets, {("B", "D", "E")})
                selected = next(iter(trials))
                return SearchDecision(selected, "supported_conditional_gain", "supported", {})
            self.fail(f"Unexpected additional search stage: {subsets}")

        with tempfile.TemporaryDirectory() as output_dir:
            with patch("util.module_c_preflight_policy.choose_action", side_effect=scripted_choice):
                result = run_module_c_preflight_selection(
                    args=_args(output_dir, classes=3),
                    model=_TinyGram(classes=3),
                    data_loader_train=support_loader,
                    data_loader_val=validation_loader,
                    device=torch.device("cpu"),
                )

            self.assertEqual(result.decision.selected_modules, ("B", "D", "E"))
            self.assertEqual(
                [step["stage"] for step in result.decision.search_steps],
                ["primary", "conditional_addition", "conditional_addition"],
            )
            with open(result.decision_json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["search"]["retired_actions"], [])

    def test_subject_metadata_is_required_for_clustered_evidence(self):
        support_loader, _ = self._loaders(classes=3)
        samples = torch.randn(9, 4, 5)
        labels = torch.tensor([0, 1, 2] * 3)
        validation_loader = DataLoader(TensorDataset(samples, labels), batch_size=3, shuffle=False)

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaisesRegex(ValueError, "subject_id"):
                run_module_c_preflight_selection(
                    args=_args(output_dir, classes=3),
                    model=_TinyGram(classes=3),
                    data_loader_train=support_loader,
                    data_loader_val=validation_loader,
                    device=torch.device("cpu"),
                )

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
