import unittest

from util.module_c_rgfs_policy import RGFSConfig, select_rgfs_subset
from util.module_c_preflight_policy import _collect_probe_batches, _focused_burden_cap, _low_rank_fit_from_snapshot


class ModuleCRGFSPolicyTests(unittest.TestCase):
    def test_zero_batch_cap_means_full_probe_split(self):
        loader = [("x0", "y0"), ("x1", "y1"), ("x2", "y2")]

        self.assertEqual(_collect_probe_batches(loader, 0), loader)
        self.assertEqual(_collect_probe_batches(loader, -1), loader)
        self.assertEqual(_collect_probe_batches(loader, 2), loader[:2])

    def test_missing_low_rank_measurement_is_not_positive_evidence(self):
        self.assertEqual(
            _low_rank_fit_from_snapshot({"lrf_weighted": 0.0, "lrf_energy": 0.0}),
            0.0,
        )
        self.assertAlmostEqual(
            _low_rank_fit_from_snapshot(
                {"lrf_weighted": 4.0, "lrf_energy": 5.0, "lrf_total_energy": 10.0}
            ),
            0.4,
        )

    def test_structural_cap_uses_median_above_uniform_burden(self):
        burden = {0: 0.10, 1: 0.20, 2: 0.30, 3: 0.40}

        self.assertAlmostEqual(_focused_burden_cap(None, burden), 0.35)

    def test_default_focus_rule_is_at_or_above_uniform_burden(self):
        decision = select_rgfs_subset(
            module_ids=["B", "D"],
            class_ids=[0, 1, 2],
            burden={0: 0.34, 1: 0.33, 2: 0.33},
            relief_lcb={
                "B": {0: 0.20, 1: 0.00, 2: 0.00},
                "D": {0: 0.00, 1: 0.90, 2: 0.90},
            },
            harm_lcb={"B": {}, "D": {}},
            complexity={"B": 1.0, "D": 1.0},
            config=RGFSConfig(),
        )

        self.assertEqual(decision.focus_classes, (0,))
        self.assertEqual(decision.selected_modules, ("B",))

    def test_zero_harm_threshold_blocks_only_positive_harm(self):
        decision = select_rgfs_subset(
            module_ids=["B", "E"],
            class_ids=[0, 1],
            burden={0: 0.60, 1: 0.40},
            relief_lcb={
                "B": {0: 0.20, 1: 0.00},
                "E": {0: 0.20, 1: 0.00},
            },
            harm_lcb={
                "B": {0: 0.00},
                "E": {0: 0.01},
            },
            complexity={"B": 1.0, "E": 1.0},
            config=RGFSConfig(harm_veto_threshold=0.0),
        )

        self.assertEqual(decision.selected_modules, ("B",))
        self.assertEqual(decision.candidate_decisions["E"]["gate"], "blocked_harm_high_burden")

    def test_selects_module_covering_high_burden_class_over_global_pressure(self):
        decision = select_rgfs_subset(
            module_ids=["B", "E"],
            class_ids=[0, 1],
            burden={0: 0.20, 1: 0.80},
            relief_lcb={
                "B": {0: 0.90, 1: 0.00},
                "E": {0: 0.00, 1: 0.55},
            },
            harm_lcb={"B": {}, "E": {}},
            complexity={"B": 1.0, "E": 1.2},
            config=RGFSConfig(min_marginal_gain=0.01),
        )

        self.assertEqual(decision.selected_modules, ("E",))
        self.assertGreater(decision.class_coverage[1], 0.0)

    def test_adds_complementary_modules_but_not_redundant_modules(self):
        decision = select_rgfs_subset(
            module_ids=["B", "D", "E"],
            class_ids=[0, 1],
            burden={0: 0.50, 1: 0.50},
            relief_lcb={
                "B": {0: 0.60, 1: 0.02},
                "D": {0: 0.58, 1: 0.01},
                "E": {0: 0.00, 1: 0.50},
            },
            harm_lcb={"B": {}, "D": {}, "E": {}},
            complexity={"B": 1.0, "D": 0.9, "E": 1.2},
            config=RGFSConfig(min_marginal_gain=0.01, tie_tolerance=0.005),
        )

        self.assertEqual(decision.selected_modules, ("B", "E"))
        self.assertNotIn("D", decision.selected_modules)

    def test_reliable_harm_on_high_burden_class_blocks_candidate(self):
        decision = select_rgfs_subset(
            module_ids=["B", "E"],
            class_ids=[0, 1],
            burden={0: 0.60, 1: 0.40},
            relief_lcb={
                "B": {0: 0.10, 1: 0.10},
                "E": {0: 0.00, 1: 0.90},
            },
            harm_lcb={
                "B": {},
                "E": {0: 0.20},
            },
            complexity={"B": 1.0, "E": 1.2},
            config=RGFSConfig(min_marginal_gain=0.01, harm_veto_threshold=0.05),
        )

        self.assertEqual(decision.selected_modules, ("B",))
        self.assertEqual(decision.candidate_decisions["E"]["gate"], "blocked_harm_high_burden")

    def test_returns_empty_when_no_reliable_positive_relief_exists(self):
        decision = select_rgfs_subset(
            module_ids=["B", "D", "E"],
            class_ids=[0, 1, 2],
            burden={0: 0.33, 1: 0.33, 2: 0.34},
            relief_lcb={
                "B": {0: 0.00, 1: 0.00, 2: 0.00},
                "D": {0: 0.00, 1: 0.00, 2: 0.00},
                "E": {0: 0.00, 1: 0.00, 2: 0.00},
            },
            harm_lcb={"B": {}, "D": {}, "E": {}},
            complexity={"B": 1.0, "D": 1.0, "E": 1.0},
            config=RGFSConfig(min_marginal_gain=0.01),
        )

        self.assertEqual(decision.selected_modules, tuple())
        self.assertEqual(decision.reason, "no reliable positive marginal relief")

    def test_forces_nonempty_when_formal_lora_search_disallows_empty(self):
        decision = select_rgfs_subset(
            module_ids=["B", "D", "E"],
            class_ids=[0, 1, 2],
            burden={0: 0.33, 1: 0.33, 2: 0.34},
            relief_lcb={
                "B": {0: 0.02, 1: 0.00, 2: 0.00},
                "D": {0: 0.00, 1: 0.00, 2: 0.00},
                "E": {0: 0.00, 1: 0.00, 2: 0.00},
            },
            harm_lcb={"B": {}, "D": {}, "E": {}},
            complexity={"B": 1.0, "D": 1.0, "E": 1.0},
            config=RGFSConfig(min_marginal_gain=0.05, allow_empty=False),
        )

        self.assertEqual(decision.selected_modules, ("B",))
        self.assertTrue(decision.candidate_decisions["B"]["forced_nonempty"])
        self.assertIn("forced non-empty", decision.reason)

    def test_structural_residual_can_select_e_without_class_shortcut(self):
        decision = select_rgfs_subset(
            module_ids=["B", "E"],
            class_ids=[0, 1],
            burden={0: 0.50, 1: 0.50},
            relief_lcb={
                "B": {0: 0.01, 1: 0.01},
                "E": {0: 0.00, 1: 0.00},
            },
            harm_lcb={"B": {}, "E": {}},
            complexity={"B": 1.0, "E": 1.2},
            functional_burden={"E:structural_balance": 0.40},
            functional_relief_lcb={
                "B": {"E:structural_balance": 0.00},
                "E": {"E:structural_balance": 0.80},
            },
            config=RGFSConfig(min_marginal_gain=0.01),
        )

        self.assertEqual(decision.selected_modules, ("E",))
        self.assertGreater(decision.functional_coverage["E:structural_balance"], 0.0)
        self.assertGreater(decision.candidate_decisions["E"]["functional_marginal_gain"], 0.0)

    def test_structural_residual_does_not_bypass_high_burden_harm_gate(self):
        decision = select_rgfs_subset(
            module_ids=["B", "E"],
            class_ids=[0, 1],
            burden={0: 0.70, 1: 0.30},
            relief_lcb={
                "B": {0: 0.20, 1: 0.10},
                "E": {0: 0.00, 1: 0.00},
            },
            harm_lcb={
                "B": {},
                "E": {0: 0.20},
            },
            complexity={"B": 1.0, "E": 1.2},
            functional_burden={"E:structural_balance": 0.50},
            functional_relief_lcb={
                "B": {"E:structural_balance": 0.00},
                "E": {"E:structural_balance": 0.90},
            },
            config=RGFSConfig(min_marginal_gain=0.01, harm_veto_threshold=0.05),
        )

        self.assertEqual(decision.selected_modules, ("B",))
        self.assertEqual(decision.candidate_decisions["E"]["gate"], "blocked_harm_high_burden")


if __name__ == "__main__":
    unittest.main()
