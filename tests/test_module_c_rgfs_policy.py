import unittest

from util.module_c_rgfs_policy import RGFSConfig, select_rgfs_subset


class ModuleCRGFSPolicyTests(unittest.TestCase):
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
            burden={0: 0.55, 1: 0.45},
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


if __name__ == "__main__":
    unittest.main()
