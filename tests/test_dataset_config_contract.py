import json
from pathlib import Path
import tempfile
import unittest

from util.dataset_config import DatasetConfigError, load_task_dataset_info


ROOT = Path(__file__).resolve().parents[1]


class DatasetConfigContractTests(unittest.TestCase):
    def test_bci_subject_modes_consume_the_matching_generated_splits(self):
        payload = json.loads(
            (ROOT / "dataset_config" / "Classification.json").read_text(
                encoding="utf-8"
            )
        )
        roots = payload["BCI-IV-2A"]["root"]

        self.assertEqual(
            roots,
            {
                "multi": "./preprocessing/BCI-4-2A/multi_subject_json",
                "cross": "./preprocessing/BCI-4-2A/cross_subject_json",
                "fewshot": "./preprocessing/BCI-4-2A/cross_subject_json",
            },
        )

    def test_retrieval_does_not_require_a_task_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                load_task_dataset_info(
                    "Retrieval",
                    "Things-EEG",
                    config_root=Path(tmp),
                )
            )

    def test_unknown_configured_dataset_fails_with_available_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_root = Path(tmp)
            (config_root / "Classification.json").write_text(
                json.dumps({"Known": {"root": {}, "num_classes": 2, "num_t": 1}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                DatasetConfigError,
                "Unknown Classification dataset 'Missing'.*Known",
            ):
                load_task_dataset_info(
                    "Classification",
                    "Missing",
                    config_root=config_root,
                )


if __name__ == "__main__":
    unittest.main()
