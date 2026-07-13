import contextlib
import io
import pathlib
import sys
import unittest

from tools.run_manifest import expand_manifest, load_manifest
from run_finetuning import get_args, resolve_output_root


ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiment_manifests" / "module_c_exhaustive_seed0_4datasets.json"


class ExperimentManifestCliTests(unittest.TestCase):
    def test_training_parser_rejects_removed_backbone(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                get_args(["--model_name", "NeurI" + "PT"])

    def test_module_c_manifest_expands_to_24_unique_runs(self):
        runs = expand_manifest(
            load_manifest(MANIFEST),
            python_executable=sys.executable,
            output_root=pathlib.Path("outputs"),
        )

        self.assertEqual(len(runs), 24)
        self.assertEqual(len({run.run_tag for run in runs}), 24)
        self.assertEqual(
            {run.dataset for run in runs},
            {"TUEV", "Sleep-EDF", "BCI-IV-2A", "SEED-IV"},
        )
        self.assertEqual(
            {run.model for run in runs},
            {"EEGPT", "BIOT", "LaBraM", "CBraMod", "Gram", "CSBrain"},
        )

    def test_commands_include_matrix_fields_and_model_overrides(self):
        runs = expand_manifest(
            load_manifest(MANIFEST),
            python_executable=sys.executable,
            output_root=pathlib.Path("outputs"),
        )
        gram = next(
            run for run in runs
            if run.dataset == "TUEV" and run.model == "Gram" and run.seed == 0
        )
        eegpt = next(
            run for run in runs
            if run.dataset == "TUEV" and run.model == "EEGPT" and run.seed == 0
        )

        self.assertEqual(gram.command[0], sys.executable)
        self.assertIn("--dataset", gram.command)
        self.assertIn("TUEV", gram.command)
        self.assertIn("--model_name", gram.command)
        self.assertIn("Gram", gram.command)
        self.assertIn("--gram_vqgan_ckpt", gram.command)
        self.assertIn("--short_output_tag_only", gram.command)
        self.assertIn("--run_tag", gram.command)
        self.assertIn("--sampling_rate", eegpt.command)
        self.assertNotIn("--gram_ckpt", eegpt.command)

    def test_filters_reduce_the_expanded_matrix(self):
        runs = expand_manifest(
            load_manifest(MANIFEST),
            python_executable=sys.executable,
            output_root=pathlib.Path("outputs"),
            datasets={"BCI-IV-2A"},
            models={"BIOT", "EEGPT"},
            seeds={0},
            preflight_only=True,
        )

        self.assertEqual(len(runs), 2)
        self.assertTrue(all("--module_c_preflight_only" in run.command for run in runs))

    def test_expanded_command_is_accepted_by_training_parser(self):
        run = expand_manifest(
            load_manifest(MANIFEST),
            python_executable=sys.executable,
            output_root=pathlib.Path("portable-outputs"),
            datasets={"BCI-IV-2A"},
            models={"CBraMod"},
            seeds={0},
            preflight_only=True,
        )[0]

        args, ds_init = get_args(list(run.command[2:]))

        self.assertIsNone(ds_init)
        self.assertEqual(args.dataset, "BCI-IV-2A")
        self.assertEqual(args.model_name, "CBraMod")
        self.assertTrue(args.module_c_preflight_only)
        self.assertEqual(args.output_dir, "portable-outputs")
        self.assertEqual(resolve_output_root(args), "portable-outputs")


if __name__ == "__main__":
    unittest.main()
