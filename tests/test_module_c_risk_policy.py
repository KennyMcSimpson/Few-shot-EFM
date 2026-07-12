import argparse
import math
import unittest

import util.module_c_risk_policy as module_c_risk_policy
from util.fb_policy import add_fb_args, resolve_functional_args
from util.module_c_lora_search import build_module_c_recipe, parse_module_ids
from util.module_c_risk_policy import (
    ActionTrial,
    PairedRiskEvidence,
    choose_action,
    cluster_jackknife_evidence,
    holm_adjust,
)


def _evidence(
    overall_gain,
    class_gains,
    gain_p,
    harm_p=None,
    status="cluster_jackknife",
):
    harm_p = harm_p or {class_id: 0.5 for class_id in class_gains}
    return PairedRiskEvidence(
        subject_class_gain={},
        class_gain={int(class_id): float(value) for class_id, value in class_gains.items()},
        overall_gain=float(overall_gain),
        worst_class_gain=min(float(value) for value in class_gains.values()),
        cluster_count=6,
        class_cluster_counts={int(class_id): 6 for class_id in class_gains},
        overall_standard_error=0.1,
        class_standard_error={int(class_id): 0.1 for class_id in class_gains},
        overall_gain_p_value=float(gain_p),
        class_harm_p_values={int(class_id): float(value) for class_id, value in harm_p.items()},
        confidence_status=status,
    )


def _trial(label, subset, evidence, parameter_count, base=()):
    base = tuple(base)
    subset = tuple(subset)
    return ActionTrial(
        label=label,
        base_subset=base,
        candidate_subset=subset,
        added_actions=tuple(action for action in subset if action not in base),
        parameter_count=int(parameter_count),
        evidence=evidence,
    )


class ModuleCRiskPolicyTests(unittest.TestCase):
    def test_aggregates_windows_then_subjects_then_classes(self):
        evidence = cluster_jackknife_evidence(
            {
                "s1": {0: [1.0, 3.0], 1: [2.0]},
                "s2": {0: [0.0], 1: [4.0, 4.0, 4.0]},
                "s3": {0: [1.0], 1: [3.0]},
            }
        )

        self.assertAlmostEqual(evidence.subject_class_gain["s1"][0], 2.0)
        self.assertAlmostEqual(evidence.class_gain[0], 1.0)
        self.assertAlmostEqual(evidence.class_gain[1], 3.0)
        self.assertAlmostEqual(evidence.overall_gain, 2.0)
        self.assertEqual(evidence.cluster_count, 3)

    def test_cluster_jackknife_reports_finite_one_sided_evidence(self):
        evidence = cluster_jackknife_evidence(
            {
                "s1": {0: [0.30], 1: [0.20]},
                "s2": {0: [0.20], 1: [0.10]},
                "s3": {0: [0.40], 1: [0.20]},
                "s4": {0: [0.30], 1: [0.10]},
            }
        )

        self.assertEqual(evidence.confidence_status, "cluster_jackknife")
        self.assertGreater(evidence.overall_standard_error, 0.0)
        self.assertTrue(math.isfinite(evidence.overall_gain_p_value))
        self.assertLess(evidence.overall_gain_p_value, 0.05)
        self.assertTrue(all(value > 0.05 for value in evidence.class_harm_p_values.values()))

    def test_holm_adjustment_controls_a_stage_family(self):
        adjusted = holm_adjust({"B": 0.01, "D": 0.04, "E": 0.03})

        self.assertAlmostEqual(adjusted["B"], 0.03)
        self.assertAlmostEqual(adjusted["D"], 0.06)
        self.assertAlmostEqual(adjusted["E"], 0.06)

    def test_primary_uses_supported_gain_and_vetoes_supported_class_harm(self):
        decision = choose_action(
            [
                _trial("B", ("B",), _evidence(0.10, {0: 0.10, 1: 0.10}, 0.01), 10),
                _trial(
                    "D",
                    ("D",),
                    _evidence(0.30, {0: 0.50, 1: -0.10}, 0.001, harm_p={0: 0.9, 1: 0.001}),
                    12,
                ),
                _trial("E", ("E",), _evidence(0.20, {0: 0.20, 1: 0.20}, 0.20), 8),
            ],
            require_nonempty=True,
        )

        self.assertEqual(decision.selected_subset, ("B",))
        self.assertEqual(decision.evidence_strength, "supported")
        self.assertEqual(decision.trial_diagnostics["D"]["gate"], "supported_class_harm")

    def test_nonempty_fallback_prefers_point_gain_without_supported_harm(self):
        decision = choose_action(
            [
                _trial("B", ("B",), _evidence(-0.10, {0: -0.10, 1: -0.10}, 0.8), 5),
                _trial("D", ("D",), _evidence(0.04, {0: 0.03, 1: 0.05}, 0.2), 20),
                _trial(
                    "E",
                    ("E",),
                    _evidence(0.10, {0: 0.30, 1: -0.10}, 0.2, harm_p={0: 0.9, 1: 0.001}),
                    10,
                ),
            ],
            require_nonempty=True,
        )

        self.assertEqual(decision.selected_subset, ("D",))
        self.assertEqual(decision.evidence_strength, "weak")
        self.assertEqual(decision.reason, "nonempty_weak_best_observed_gain")

    def test_nonempty_fallback_is_explicit_minimax_when_every_action_has_harm(self):
        decision = choose_action(
            [
                _trial("B", ("B",), _evidence(-0.20, {0: -0.20, 1: -0.10}, 0.9, {0: 0.001, 1: 0.01}), 5),
                _trial("D", ("D",), _evidence(-0.10, {0: -0.10, 1: -0.10}, 0.9, {0: 0.001, 1: 0.001}), 20),
                _trial("E", ("E",), _evidence(-0.25, {0: -0.05, 1: -0.45}, 0.9, {0: 0.01, 1: 0.001}), 10),
            ],
            require_nonempty=True,
        )

        self.assertEqual(decision.selected_subset, ("D",))
        self.assertEqual(decision.evidence_strength, "mandatory")
        self.assertEqual(decision.reason, "nonempty_mandatory_minimax_harm")

    def test_parameter_count_breaks_only_an_exact_evidence_tie(self):
        shared = _evidence(0.20, {0: 0.20, 1: 0.20}, 0.001)
        decision = choose_action(
            [
                _trial("B", ("B",), shared, 20),
                _trial("D", ("D",), shared, 10),
            ],
            require_nonempty=True,
        )

        self.assertEqual(decision.selected_subset, ("D",))

    def test_conditional_addition_uses_measured_subset_gain(self):
        decision = choose_action(
            [
                _trial("B+D", ("B", "D"), _evidence(0.12, {0: 0.10, 1: 0.14}, 0.005), 20, base=("B",)),
                _trial("B+E", ("B", "E"), _evidence(-0.02, {0: 0.02, 1: -0.06}, 0.8), 15, base=("B",)),
            ],
            require_nonempty=False,
        )

        self.assertEqual(decision.selected_subset, ("B", "D"))
        self.assertEqual(decision.reason, "supported_conditional_gain")

    def test_alternative_pair_uses_the_same_supported_rule_on_a_joint_branch(self):
        decision = choose_action(
            [
                _trial(
                    "B+D+E",
                    ("B", "D", "E"),
                    _evidence(0.15, {0: 0.12, 1: 0.18}, 0.01),
                    30,
                    base=("B",),
                )
            ],
            require_nonempty=False,
        )

        self.assertEqual(decision.selected_subset, ("B", "D", "E"))
        self.assertEqual(decision.evidence_strength, "supported")

    def test_policy_does_not_expose_backward_deletion(self):
        self.assertFalse(hasattr(module_c_risk_policy, "choose_floating_deletion"))

    def test_selector_supports_binary_classes(self):
        evidence = cluster_jackknife_evidence(
            {
                "s1": {0: [0.20], 1: [0.10]},
                "s2": {0: [0.10], 1: [0.20]},
                "s3": {0: [0.30], 1: [0.10]},
            }
        )

        self.assertEqual(set(evidence.class_gain), {0, 1})

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


if __name__ == "__main__":
    unittest.main()
