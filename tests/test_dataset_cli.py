import json
import pathlib
import sys
import tempfile
import unittest

from tools.dataset_cli import (
    DatasetCommandError,
    audit_split_directory,
    build_dataset_command,
    discover_preprocessing,
    load_training_registry,
    resolve_script,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class DatasetCliTests(unittest.TestCase):
    def test_training_registry_reports_only_configured_tasks(self):
        registry = load_training_registry(ROOT)

        self.assertEqual(
            set(registry),
            {
                "BCI-IV-2A",
                "EEGMAT",
                "HMC",
                "SEED-IV",
                "SEED-VIG",
                "Siena",
                "Sleep-EDF",
                "TUEV",
            },
        )
        self.assertEqual(registry["TUEV"]["tasks"], ("Classification",))
        self.assertEqual(registry["SEED-VIG"]["tasks"], ("Regression",))

    def test_preprocessing_discovery_keeps_logical_bci_name(self):
        discovered = discover_preprocessing(ROOT)

        self.assertIn("BCI-IV-2A", discovered)
        self.assertEqual(
            discovered["BCI-IV-2A"]["directory"].as_posix().split("/")[-1],
            "BCI-4-2A",
        )
        self.assertEqual(
            discovered["BCI-IV-2A"]["split_modes"],
            ("cross", "multi"),
        )
        self.assertIn("SHU", discovered)
        self.assertNotIn("SHU", load_training_registry(ROOT))

    def test_resolve_script_rejects_a_missing_split_mode(self):
        script = resolve_script(ROOT, "BCI-IV-2A", "prepare")
        self.assertEqual(script.name, "data_process.py")

        with self.assertRaisesRegex(DatasetCommandError, "does not provide split mode 'multi'"):
            resolve_script(ROOT, "TUEV", "split", split_mode="multi")

    def test_bci_cross_split_reads_the_preprocessor_output_directory(self):
        prepare_source = (
            ROOT / "preprocessing" / "BCI-4-2A" / "data_process.py"
        ).read_text(encoding="utf-8")
        cross_source = (
            ROOT / "preprocessing" / "BCI-4-2A" / "cross_json_process.py"
        ).read_text(encoding="utf-8")

        self.assertIn("BCI-4-2A/processed_data", prepare_source)
        self.assertIn("BCI-4-2A/processed_data", cross_source)
        self.assertIn("preprocessing/BCI-4-2A/cross_subject_json", cross_source)

    def test_build_command_is_cross_platform_and_never_uses_a_shell(self):
        command = build_dataset_command(
            ROOT,
            dataset="TUEV",
            action="split",
            data_root=pathlib.Path("example-data"),
            split_mode="cross",
        )

        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[1].endswith("preprocessing/TUEV/cross_json_process.py"))
        self.assertEqual(command[2], "example-data")

    def test_split_audit_detects_path_and_basename_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = pathlib.Path(tmp)
            payloads = {
                "train": [
                    {"file": "C:/data/train/shared.pkl", "label": 0},
                    {"file": "C:/data/train/only-train.pkl", "label": 1},
                ],
                "val": [
                    {"file": "C:/data/train/shared.pkl", "label": 0},
                    {"file": "D:/copy/only-train.pkl", "label": 1},
                ],
                "test": [{"file": "C:/data/test/held-out.pkl", "label": 0}],
            }
            for split, rows in payloads.items():
                (split_dir / f"{split}.json").write_text(
                    json.dumps(rows),
                    encoding="utf-8",
                )

            audit = audit_split_directory(split_dir)

        self.assertFalse(audit["ok"])
        self.assertEqual(
            audit["overlaps"]["train__val"]["exact_paths"],
            ["C:/data/train/shared.pkl"],
        )
        self.assertEqual(
            audit["overlaps"]["train__val"]["basenames"],
            ["only-train.pkl", "shared.pkl"],
        )
        self.assertEqual(audit["overlaps"]["train__test"]["exact_paths"], [])

    def test_split_audit_normalizes_windows_paths_on_every_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = pathlib.Path(tmp)
            payloads = {
                "train": [{"file": r"C:\EEG\subject\sample.pkl", "label": 0}],
                "val": [{"file": "c:/eeg/subject/sample.pkl", "label": 0}],
                "test": [{"file": "/data/held-out.pkl", "label": 1}],
            }
            for split, rows in payloads.items():
                (split_dir / f"{split}.json").write_text(
                    json.dumps(rows),
                    encoding="utf-8",
                )

            audit = audit_split_directory(split_dir)

        self.assertFalse(audit["ok"])
        self.assertEqual(
            audit["overlaps"]["train__val"]["exact_paths"],
            [r"C:\EEG\subject\sample.pkl"],
        )

    def test_split_audit_accepts_the_training_loader_json_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = pathlib.Path(tmp)
            for split, file_name in (
                ("train", "train.pkl"),
                ("val", "val.pkl"),
                ("test", "test.pkl"),
            ):
                (split_dir / f"{split}.json").write_text(
                    json.dumps(
                        {
                            "dataset_info": {"sampling_rate": 200},
                            "subject_data": [
                                {"file": f"C:/data/{file_name}", "label": 0}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            audit = audit_split_directory(split_dir)

        self.assertTrue(audit["ok"])
        self.assertEqual(audit["counts"], {"train": 1, "val": 1, "test": 1})


if __name__ == "__main__":
    unittest.main()
