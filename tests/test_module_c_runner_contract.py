import pathlib
import unittest


class ModuleCRunnerContractTests(unittest.TestCase):
    def test_decision_lookup_follows_remote_results_symlink(self):
        runner = pathlib.Path(__file__).resolve().parents[1] / "run_c_task_aligned_seed0_4datasets_5090.sh"
        text = runner.read_text(encoding="utf-8")

        self.assertIn("find -L finetuning_results", text)

    def test_status_prefers_final_path_evidence_strength(self):
        runner = pathlib.Path(__file__).resolve().parents[1] / "run_c_task_aligned_seed0_4datasets_5090.sh"
        text = runner.read_text(encoding="utf-8")

        self.assertIn(
            'payload.get("final_evidence_strength", payload.get("primary_evidence_strength", "NA"))',
            text,
        )


if __name__ == "__main__":
    unittest.main()
