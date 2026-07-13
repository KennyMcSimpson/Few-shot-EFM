import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch.nn as nn

from util.backbone_contracts import (
    SUPPORTED_BD_BACKBONES,
    BackboneContractError,
    backbone_bd_contract_hash,
    get_backbone_bd_contract,
    resolve_backbone_bd_sites,
    resolve_canonical_head,
    save_backbone_bd_contract_audit,
)
from util.lora import apply_lora_to_eegfm


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)


class _TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()


class _BIOTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(4, 8)
        self.w2 = nn.Linear(8, 4)


class _LinearBlock(nn.Module):
    def __init__(self, include_global=False):
        super().__init__()
        self.linear1 = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 4)
        if include_global:
            self.global_fc = nn.Linear(4, 4)


class _Container(nn.Module):
    pass


def _base_wrapper():
    model = _Container()
    model.task_head = nn.Linear(4, 3)
    model.main_model = _Container()
    return model


def _biot():
    model = _base_wrapper()
    model.chan_conv = nn.Conv1d(4, 4, 1)
    model.main_model.layers = nn.ModuleList([_BIOTLayer()])
    return model


def _merged_transformer(with_bridge):
    model = _base_wrapper()
    if with_bridge:
        model.chan_conv = nn.Conv1d(4, 4, 1)
    model.main_model.blocks = nn.ModuleList([_TransformerBlock()])
    return model


def _linear_transformer(include_global=False, bridge=None):
    model = _base_wrapper()
    if bridge is not None:
        model.chan_conv = bridge
    model.main_model.blocks = nn.ModuleList([_LinearBlock(include_global=include_global)])
    return model


def _gram():
    model = _Container()
    model.task_head = nn.Identity()
    model.main_model = _Container()
    model.main_model.model = _Container()
    model.main_model.model.blocks = nn.ModuleList([_TransformerBlock()])
    model.main_model.model.proj_layers = nn.ModuleList([nn.Linear(4, 4)])
    model.main_model.model.linear_projection = nn.Linear(4, 4)
    model.main_model.model.cls_head = nn.Linear(4, 3)
    model.main_model.model.decoder = nn.Linear(4, 4)
    return model


class BackboneContractTests(unittest.TestCase):
    def test_registry_contains_exactly_six_supported_backbones(self):
        self.assertEqual(
            SUPPORTED_BD_BACKBONES,
            ("BIOT", "EEGPT", "LaBraM", "CBraMod", "CSBrain", "Gram"),
        )
        hashes = {backbone_bd_contract_hash(name) for name in SUPPORTED_BD_BACKBONES}
        self.assertEqual(len(hashes), len(SUPPORTED_BD_BACKBONES))

    def test_declared_sites_match_expected_contract(self):
        expected = {
            "BIOT": (("chan_conv",), ("*.w1", "*.w2")),
            "EEGPT": (("chan_conv",), ("*.mlp.fc1", "*.mlp.fc2")),
            "LaBraM": ((), ("*.mlp.fc1", "*.mlp.fc2")),
            "CBraMod": ((), ("*.linear1", "*.linear2")),
            "CSBrain": (("chan_conv",), ("*.linear1", "*.linear2")),
            "Gram": ((), ("main_model.model.blocks.*.mlp.fc1", "main_model.model.blocks.*.mlp.fc2")),
        }
        for model_name, (bridge_globs, semantic_globs) in expected.items():
            with self.subTest(model=model_name):
                contract = get_backbone_bd_contract(model_name)
                self.assertTrue(contract.raw_input_site)
                self.assertEqual(tuple(site.path_glob for site in contract.bridge_sites), bridge_globs)
                self.assertEqual(tuple(site.path_glob for site in contract.semantic_ffn_sites), semantic_globs)

    def test_resolver_returns_exact_b_and_d_sites(self):
        fixtures = {
            "BIOT": (_biot(), ("chan_conv",), ("main_model.layers.0.w1", "main_model.layers.0.w2")),
            "EEGPT": (_merged_transformer(True), ("chan_conv",), ("main_model.blocks.0.mlp.fc1", "main_model.blocks.0.mlp.fc2")),
            "LaBraM": (_merged_transformer(False), (), ("main_model.blocks.0.mlp.fc1", "main_model.blocks.0.mlp.fc2")),
            "CBraMod": (_linear_transformer(), (), ("main_model.blocks.0.linear1", "main_model.blocks.0.linear2")),
            "CSBrain": (_linear_transformer(True, nn.Conv1d(4, 4, 1)), ("chan_conv",), ("main_model.blocks.0.linear1", "main_model.blocks.0.linear2")),
            "Gram": (_gram(), (), ("main_model.model.blocks.0.mlp.fc1", "main_model.model.blocks.0.mlp.fc2")),
        }
        for model_name, (model, expected_b, expected_d) in fixtures.items():
            with self.subTest(model=model_name):
                actual_b = tuple(site.module_path for site in resolve_backbone_bd_sites(model, model_name, "B"))
                actual_d = tuple(site.module_path for site in resolve_backbone_bd_sites(model, model_name, "D"))
                self.assertEqual(actual_b, expected_b)
                self.assertEqual(actual_d, expected_d)

    def test_csbrain_identity_bridge_is_supported_absence(self):
        model = _linear_transformer(True, nn.Identity())
        self.assertEqual(resolve_backbone_bd_sites(model, "CSBrain", "B"), ())

    def test_csbrain_global_fc_and_gram_non_mlp_linears_are_not_d(self):
        csbrain = _linear_transformer(True, nn.Conv1d(4, 4, 1))
        gram = _gram()
        csbrain_d = {site.module_path for site in resolve_backbone_bd_sites(csbrain, "CSBrain", "D")}
        gram_d = {site.module_path for site in resolve_backbone_bd_sites(gram, "Gram", "D")}
        self.assertNotIn("main_model.blocks.0.global_fc", csbrain_d)
        self.assertNotIn("main_model.model.proj_layers.0", gram_d)
        self.assertNotIn("main_model.model.linear_projection", gram_d)
        self.assertNotIn("main_model.model.cls_head", gram_d)
        self.assertNotIn("main_model.model.decoder", gram_d)

    def test_canonical_head_uses_internal_gram_classifier(self):
        biot = _biot()
        gram = _gram()
        self.assertIs(resolve_canonical_head(biot, "BIOT"), biot.task_head)
        self.assertIs(resolve_canonical_head(gram, "Gram"), gram.main_model.model.cls_head)

    def test_missing_required_d_site_fails_closed(self):
        broken = _base_wrapper()
        with self.assertRaisesRegex(BackboneContractError, "BIOT.*semantic"):
            resolve_backbone_bd_sites(broken, "BIOT", "D")

    def test_wrong_required_module_type_fails_closed(self):
        broken = _biot()
        broken.main_model.layers[0].w1 = nn.Identity()
        with self.assertRaisesRegex(BackboneContractError, "w1.*linear"):
            resolve_backbone_bd_sites(broken, "BIOT", "D")

    def test_unknown_backbone_and_action_fail_closed(self):
        with self.assertRaises(BackboneContractError):
            get_backbone_bd_contract("Unknown")
        with self.assertRaises(BackboneContractError):
            resolve_backbone_bd_sites(_biot(), "BIOT", "E")

    def test_existing_four_backbones_keep_exact_semantic_injection_names(self):
        fixtures = {
            "BIOT": (_biot(), ("main_model.layers.0.w1", "main_model.layers.0.w2")),
            "EEGPT": (_merged_transformer(True), ("main_model.blocks.0.mlp.fc1", "main_model.blocks.0.mlp.fc2")),
            "LaBraM": (_merged_transformer(False), ("main_model.blocks.0.mlp.fc1", "main_model.blocks.0.mlp.fc2")),
            "CBraMod": (_linear_transformer(), ("main_model.blocks.0.linear1", "main_model.blocks.0.linear2")),
        }
        for model_name, (model, expected) in fixtures.items():
            with self.subTest(model=model_name):
                actual = apply_lora_to_eegfm(
                    model,
                    model_name,
                    lora_target="semantic",
                    r=2,
                    alpha=4.0,
                    dropout=0.0,
                    verbose=False,
                )
                self.assertEqual(tuple(actual), expected)

    def test_contract_driven_bridge_injection_includes_csbrain(self):
        fixtures = {
            "BIOT": _biot(),
            "EEGPT": _merged_transformer(True),
            "CSBrain": _linear_transformer(True, nn.Conv1d(4, 4, 1)),
        }
        for model_name, model in fixtures.items():
            with self.subTest(model=model_name):
                actual = apply_lora_to_eegfm(
                    model,
                    model_name,
                    lora_target="signal_align",
                    module_b_sites="bridge",
                    r=2,
                    alpha=4.0,
                    dropout=0.0,
                    verbose=False,
                )
                self.assertEqual(actual, ["chan_conv"])

    def test_csbrain_semantic_injection_excludes_global_fc(self):
        actual = apply_lora_to_eegfm(
            _linear_transformer(True, nn.Conv1d(4, 4, 1)),
            "CSBrain",
            lora_target="semantic",
            r=2,
            alpha=4.0,
            dropout=0.0,
            verbose=False,
        )
        self.assertEqual(
            tuple(actual),
            ("main_model.blocks.0.linear1", "main_model.blocks.0.linear2"),
        )

    def test_gram_semantic_injection_is_limited_to_encoder_mlp(self):
        actual = apply_lora_to_eegfm(
            _gram(),
            "Gram",
            lora_target="semantic",
            r=2,
            alpha=4.0,
            dropout=0.0,
            verbose=False,
        )
        self.assertEqual(
            tuple(actual),
            (
                "main_model.model.blocks.0.mlp.fc1",
                "main_model.model.blocks.0.mlp.fc2",
            ),
        )

    def test_injection_audit_records_contract_and_realized_d_sites(self):
        model = _biot()
        apply_lora_to_eegfm(
            model,
            "BIOT",
            lora_target="semantic",
            r=2,
            alpha=4.0,
            dropout=0.0,
            verbose=False,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            path = save_backbone_bd_contract_audit(
                SimpleNamespace(output_dir=output_dir),
                model,
            )
            payload = json.loads(Path(path).read_text(encoding="utf-8"))

        self.assertEqual(payload["model_name"], "BIOT")
        self.assertEqual(payload["contract_hash"], backbone_bd_contract_hash("BIOT"))
        self.assertEqual(payload["resolved_bridge_sites"], [])
        self.assertEqual(
            payload["resolved_semantic_ffn_sites"],
            ["main_model.layers.0.w1", "main_model.layers.0.w2"],
        )
        self.assertEqual(payload["injected_d_sites"], payload["resolved_semantic_ffn_sites"])


if __name__ == "__main__":
    unittest.main()
