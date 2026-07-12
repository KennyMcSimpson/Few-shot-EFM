import math
import unittest

from util.module_c_exhaustive_policy import (
    SubsetRisk,
    enumerate_action_subsets,
    select_exhaustive_subset,
)


def _risk(subset, macro_loss, *, micro_loss=None, per_class=None, params=0):
    return SubsetRisk(
        subset=tuple(subset),
        macro_loss=float(macro_loss),
        micro_loss=float(macro_loss if micro_loss is None else micro_loss),
        per_class_loss=dict(per_class or {0: macro_loss, 1: macro_loss}),
        adapter_parameter_count=int(params),
    )


class ModuleCExhaustivePolicyTests(unittest.TestCase):
    def test_enumerates_every_nonempty_bde_subset_once_in_canonical_order(self):
        self.assertEqual(
            enumerate_action_subsets(("B", "D", "E")),
            (
                ("B",),
                ("D",),
                ("E",),
                ("B", "D"),
                ("B", "E"),
                ("D", "E"),
                ("B", "D", "E"),
            ),
        )

    def test_global_minimum_wins_without_a_forward_search_path(self):
        branches = {
            (): _risk((), 1.00),
            ("B",): _risk(("B",), 0.80, params=10),
            ("D",): _risk(("D",), 0.82, params=12),
            ("E",): _risk(("E",), 0.81, params=14),
            ("B", "D"): _risk(("B", "D"), 0.79, params=22),
            ("B", "E"): _risk(("B", "E"), 0.78, params=24),
            ("D", "E"): _risk(("D", "E"), 0.77, params=26),
            ("B", "D", "E"): _risk(("B", "D", "E"), 0.60, params=36),
        }

        decision = select_exhaustive_subset(branches, ("B", "D", "E"))

        self.assertEqual(decision.selected_subset, ("B", "D", "E"))
        self.assertEqual(decision.selection_status, "positive_gain")
        self.assertEqual(decision.runner_up_subset, ("D", "E"))
        self.assertAlmostEqual(decision.selection_gap, 0.17)
        self.assertAlmostEqual(decision.observed_gain, 0.40)

    def test_exact_tie_prefers_fewer_actions_but_near_tie_has_no_tolerance(self):
        exact = {
            (): _risk((), 1.0),
            ("B",): _risk(("B",), 0.4, params=10),
            ("D",): _risk(("D",), 0.8, params=10),
            ("E",): _risk(("E",), 0.9, params=10),
            ("B", "D"): _risk(("B", "D"), 0.4, params=20),
            ("B", "E"): _risk(("B", "E"), 0.7, params=20),
            ("D", "E"): _risk(("D", "E"), 0.6, params=20),
            ("B", "D", "E"): _risk(("B", "D", "E"), 0.5, params=30),
        }
        near = dict(exact)
        near[("B",)] = _risk(("B",), math.nextafter(0.4, math.inf), params=10)

        self.assertEqual(
            select_exhaustive_subset(exact, ("B", "D", "E")).selected_subset,
            ("B",),
        )
        self.assertEqual(
            select_exhaustive_subset(near, ("B", "D", "E")).selected_subset,
            ("B", "D"),
        )

    def test_exact_ties_use_adapter_count_then_canonical_order(self):
        branches = {
            (): _risk((), 1.0),
            ("B",): _risk(("B",), 0.4, params=20),
            ("D",): _risk(("D",), 0.4, params=10),
            ("E",): _risk(("E",), 0.8, params=5),
            ("B", "D"): _risk(("B", "D"), 0.7, params=30),
            ("B", "E"): _risk(("B", "E"), 0.7, params=25),
            ("D", "E"): _risk(("D", "E"), 0.7, params=15),
            ("B", "D", "E"): _risk(("B", "D", "E"), 0.9, params=35),
        }

        self.assertEqual(
            select_exhaustive_subset(branches, ("B", "D", "E")).selected_subset,
            ("D",),
        )
        branches[("B",)] = _risk(("B",), 0.4, params=10)
        self.assertEqual(
            select_exhaustive_subset(branches, ("B", "D", "E")).selected_subset,
            ("B",),
        )

    def test_nonempty_winner_is_explicit_when_empty_is_better(self):
        branches = {
            (): _risk((), 0.10),
            ("B",): _risk(("B",), 0.20),
            ("D",): _risk(("D",), 0.30),
            ("E",): _risk(("E",), 0.40),
            ("B", "D"): _risk(("B", "D"), 0.50),
            ("B", "E"): _risk(("B", "E"), 0.60),
            ("D", "E"): _risk(("D", "E"), 0.70),
            ("B", "D", "E"): _risk(("B", "D", "E"), 0.80),
        }

        decision = select_exhaustive_subset(branches, ("B", "D", "E"))

        self.assertEqual(decision.selected_subset, ("B",))
        self.assertEqual(decision.selection_status, "forced_nonempty_best_observed")
        self.assertAlmostEqual(decision.observed_gain, -0.10)

    def test_certificate_reuses_cached_subsets_for_contributions_and_interactions(self):
        branches = {
            (): _risk((), 1.00, per_class={0: 1.20, 1: 0.80}),
            ("B",): _risk(("B",), 0.90, per_class={0: 1.05, 1: 0.75}),
            ("D",): _risk(("D",), 0.80, per_class={0: 0.95, 1: 0.65}),
            ("E",): _risk(("E",), 0.85, per_class={0: 1.00, 1: 0.70}),
            ("B", "D"): _risk(("B", "D"), 0.65),
            ("B", "E"): _risk(("B", "E"), 0.70),
            ("D", "E"): _risk(("D", "E"), 0.60),
            ("B", "D", "E"): _risk(("B", "D", "E"), 0.50),
        }

        decision = select_exhaustive_subset(branches, ("B", "D", "E"))

        self.assertAlmostEqual(decision.conditional_contributions["B"], 0.10)
        self.assertAlmostEqual(decision.conditional_contributions["D"], 0.20)
        self.assertAlmostEqual(decision.conditional_contributions["E"], 0.15)
        self.assertAlmostEqual(decision.pair_interactions["B+D"], 0.05)
        self.assertAlmostEqual(decision.pair_interactions["B+E"], 0.05)
        self.assertAlmostEqual(decision.pair_interactions["D+E"], 0.05)
        self.assertAlmostEqual(decision.triple_interaction, -0.10)
        self.assertAlmostEqual(decision.per_class_gain["B"][0], 0.15)
        self.assertAlmostEqual(decision.per_class_gain["B"][1], 0.05)

    def test_missing_branch_fails_instead_of_silently_pruning_the_grid(self):
        branches = {
            (): _risk((), 1.0),
            ("B",): _risk(("B",), 0.9),
        }

        with self.assertRaisesRegex(ValueError, "missing exhaustive branches"):
            select_exhaustive_subset(branches, ("B", "D", "E"))

    def test_unexpected_or_noncanonical_branch_fails_instead_of_polluting_the_grid(self):
        branches = {
            (): _risk((), 1.0),
            **{
                subset: _risk(subset, 0.5 + index / 100.0)
                for index, subset in enumerate(enumerate_action_subsets(("B", "D", "E")))
            },
            ("D", "B"): _risk(("D", "B"), 0.1),
        }

        with self.assertRaisesRegex(ValueError, "unexpected exhaustive branches"):
            select_exhaustive_subset(branches, ("B", "D", "E"))


if __name__ == "__main__":
    unittest.main()
