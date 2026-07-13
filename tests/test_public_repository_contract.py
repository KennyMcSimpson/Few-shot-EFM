import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PublicRepositoryContractTests(unittest.TestCase):
    def test_public_tree_has_no_shell_or_batch_runners(self):
        forbidden = sorted(
            path.relative_to(ROOT).as_posix()
            for suffix in ("*.bat", "*.sh")
            for path in ROOT.rglob(suffix)
            if ".git" not in path.parts
        )
        self.assertEqual(forbidden, [])

    def test_public_guides_are_present(self):
        required = (
            "docs/architecture.md",
            "docs/datasets.md",
            "docs/reproducibility.md",
            "docs/adding-a-backbone.md",
            "THIRD_PARTY_NOTICES.md",
            "licenses/Apache-2.0.txt",
            "licenses/BSD-3-Clause-Salesforce.txt",
        )
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual(missing, [])

    def test_internal_plans_and_unused_image_bundle_are_not_public(self):
        forbidden = [
            path
            for path in ("docs/superpowers", "image")
            if (ROOT / path).is_file()
            or any(candidate.is_file() for candidate in (ROOT / path).glob("**/*"))
        ]
        self.assertEqual(forbidden, [])

    def test_experiment_manifests_are_portable_json(self):
        manifests = sorted((ROOT / "experiment_manifests").glob("*.json"))
        self.assertTrue(manifests)
        for manifest in manifests:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["entrypoint"], "run_finetuning.py")
            serialized = json.dumps(payload)
            self.assertNotIn("D:\\", serialized)
            self.assertNotIn("/home/", serialized)

    def test_removed_backbone_has_no_public_integration(self):
        removed_name = "NeurI" + "PT"
        removed_lower = removed_name.lower()
        self.assertFalse((ROOT / "external" / removed_name).exists())
        self.assertFalse((ROOT / "models" / f"{removed_lower}_ada.py").exists())

        searchable_suffixes = {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml"}
        references = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path == pathlib.Path(__file__).resolve() or path.suffix.lower() not in searchable_suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue
            if removed_lower in text or removed_lower in path.as_posix().lower():
                references.append(path.relative_to(ROOT).as_posix())

        self.assertEqual(sorted(references), [])


if __name__ == "__main__":
    unittest.main()
