import argparse
import unittest

import torch

from util.module_c_lora_search import build_module_c_recipe, parse_module_ids
from util.module_c_preflight_policy import (
    first_order_effects_from_snapshots,
    prune_harmful_confirmation_additions,
)
from util.module_c_risk_policy import select_validation_risk_subset
from util.fb_policy import add_fb_args, resolve_functional_args


class ModuleCRiskPolicyTests(unittest.TestCase):
    def test_prefers_the_largest_safe_class_balanced_effect(self):
        decision = select_validation_risk_subset(
            module_effects={
                "B": {0: 0.00, 1: 0.00, 2: 0.00},
                "D": {0: 0.30, 1: 0.20, 2: 0.10},
                "E": {0: 0.40, 1: 0.40, 2: -0.05},
            },
            parameter_counts={"B": 10, "D": 12, "E": 8},
            allow_empty=False,
        )

        self.assertEqual(decision.selected_modules, ("D",))
        self.assertEqual(decision.candidate_decisions["E"]["gate"], "unsafe_class_harm")

    def test_adds_only_actions_that_improve_mean_without_lowering_worst_class(self):
        decision = select_validation_risk_subset(
            module_effects={
                "B": {0: 0.20, 1: 0.20, 2: 0.20},
                "D": {0: 0.10, 1: 0.10, 2: 0.10},
                "E": {0: 0.30, 1: -0.30, 2: 0.30},
            },
            parameter_counts={"B": 10, "D": 12, "E": 8},
            allow_empty=False,
        )

        self.assertEqual(decision.selected_modules, ("B", "D"))
        self.assertNotIn("E", decision.selected_modules)
        self.assertGreater(decision.overall_effect, 0.20)
        self.assertGreaterEqual(decision.worst_class_effect, 0.20)

    def test_forces_the_least_harmful_action_when_nonempty_is_required(self):
        decision = select_validation_risk_subset(
            module_effects={
                "B": {0: -0.10, 1: -0.10, 2: -0.10},
                "D": {0: -0.20, 1: 0.00, 2: 0.00},
                "E": {0: -0.10, 1: -0.20, 2: 0.10},
            },
            parameter_counts={"B": 10, "D": 8, "E": 6},
            allow_empty=False,
        )

        self.assertEqual(decision.selected_modules, ("B",))
        self.assertTrue(decision.forced_nonempty)
        self.assertEqual(decision.reason, "forced_nonempty_least_harm")

    def test_selector_rejects_an_empty_selection_request(self):
        with self.assertRaisesRegex(ValueError, "never permits empty"):
            select_validation_risk_subset(
                module_effects={
                    "B": {0: -0.10, 1: -0.10, 2: -0.10},
                    "D": {0: -0.20, 1: -0.20, 2: -0.20},
                    "E": {0: -0.30, 1: -0.30, 2: -0.30},
                },
                parameter_counts={"B": 10, "D": 12, "E": 8},
                allow_empty=True,
            )

    def test_selection_is_invariant_to_the_e_label(self):
        original = select_validation_risk_subset(
            module_effects={
                "B": {0: 0.00, 1: 0.00, 2: 0.00},
                "D": {0: 0.20, 1: 0.20, 2: 0.20},
                "E": {0: 0.30, 1: 0.30, 2: -0.05},
            },
            parameter_counts={"B": 10, "D": 12, "E": 8},
            allow_empty=False,
        )
        swapped = select_validation_risk_subset(
            module_effects={
                "B": {0: 0.30, 1: 0.30, 2: -0.05},
                "D": {0: 0.20, 1: 0.20, 2: 0.20},
                "E": {0: 0.00, 1: 0.00, 2: 0.00},
            },
            parameter_counts={"B": 8, "D": 12, "E": 10},
            allow_empty=False,
        )

        self.assertEqual(original.selected_modules, ("D",))
        self.assertEqual(swapped.selected_modules, ("D",))

    def test_c_accepts_only_bde_and_never_emits_qv_metadata(self):
        with self.assertRaises(ValueError):
            parse_module_ids("B,qv,E")

        recipe = build_module_c_recipe(("B", "E"))
        self.assertNotIn("qv", str(recipe).lower())

    def test_parser_exposes_only_batch_caps_for_the_c_preflight(self):
        parser = argparse.ArgumentParser()
        add_fb_args(parser)
        args = parser.parse_args([])

        self.assertEqual(args.module_c_preflight_train_batches, 0)
        self.assertEqual(args.module_c_preflight_val_batches, 0)
        self.assertFalse(hasattr(args, "module_c_rgfs_harm_threshold"))
        self.assertFalse(hasattr(args, "module_c_probe_head_steps"))

    def test_disabled_preflight_requires_an_explicit_nonempty_bde_selection(self):
        parser = argparse.ArgumentParser()
        add_fb_args(parser)
        args = parser.parse_args(["--module_c_no_preflight"])
        args.lora_target = "module_c"

        with self.assertRaisesRegex(ValueError, "nonempty --module_c_selected"):
            resolve_functional_args(args)

    def test_first_order_effect_uses_validation_loss_change_units(self):
        effects = first_order_effects_from_snapshots(
            validation_gradients={
                "B": {
                    0: {"adapter": torch.tensor([2.0, 0.0])},
                    1: {"adapter": torch.tensor([-1.0, 0.0])},
                    2: {"adapter": torch.tensor([0.0, 1.0])},
                }
            },
            virtual_updates={
                "B": {"adapter": torch.tensor([-0.5, 0.0])},
            },
        )

        self.assertAlmostEqual(effects["B"][0], 1.0)
        self.assertAlmostEqual(effects["B"][1], -0.5)
        self.assertAlmostEqual(effects["B"][2], 0.0)

    def test_confirmation_prunes_only_an_added_branch_that_is_not_needed(self):
        selected, diagnostics = prune_harmful_confirmation_additions(
            selected_modules=("B", "D", "E"),
            primary_module="B",
            full_per_class_loss={0: 1.10, 1: 1.00, 2: 1.20},
            masked_per_class_loss={
                "D": {0: 1.00, 1: 0.95, 2: 1.20},
                "E": {0: 1.20, 1: 0.90, 2: 1.20},
            },
        )

        self.assertEqual(selected, ("B", "E"))
        self.assertFalse(diagnostics["B"]["checked"])
        self.assertTrue(diagnostics["D"]["pruned"])
        self.assertFalse(diagnostics["E"]["pruned"])


if __name__ == "__main__":
    unittest.main()
