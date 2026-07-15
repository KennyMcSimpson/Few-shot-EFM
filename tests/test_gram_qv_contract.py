import unittest

import torch.nn as nn

from util.lora import LoRALinear, apply_lora_to_eegfm


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.key = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()
        self.mlp = _Mlp()


def _gram():
    model = nn.Module()
    model.main_model = nn.Module()
    model.main_model.model = nn.Module()
    model.main_model.model.blocks = nn.ModuleList([_Block()])
    model.main_model.model.proj_layers = nn.ModuleList([nn.Linear(4, 4)])
    model.task_head = nn.Linear(4, 3)
    return model


QV_SITES = [
    "main_model.model.blocks.0.attn.query",
    "main_model.model.blocks.0.attn.value",
]
FFN_SITES = [
    "main_model.model.blocks.0.mlp.fc1",
    "main_model.model.blocks.0.mlp.fc2",
]
STRUCTURAL_SITES = [
    "main_model.model.blocks.0.attn.query",
    "main_model.model.blocks.0.attn.key",
    "main_model.model.blocks.0.attn.value",
    "main_model.model.blocks.0.attn.proj",
    "main_model.model.proj_layers.0",
]


def _inject(model, target, module_c_selected=None):
    return apply_lora_to_eegfm(
        model,
        "Gram",
        lora_target=target,
        module_c_selected=module_c_selected,
        r=2,
        alpha=4.0,
        dropout=0.0,
        verbose=False,
    )


class GramQvContractTests(unittest.TestCase):
    def test_qv_wraps_query_and_value_only(self):
        model = _gram()

        actual = apply_lora_to_eegfm(
            model,
            "Gram",
            lora_target="qv",
            r=2,
            alpha=4.0,
            dropout=0.0,
            verbose=False,
        )

        self.assertEqual(
            actual,
            [
                "main_model.model.blocks.0.attn.query",
                "main_model.model.blocks.0.attn.value",
            ],
        )
        self.assertIsInstance(model.main_model.model.blocks[0].attn.query, LoRALinear)
        self.assertIsInstance(model.main_model.model.blocks[0].attn.value, LoRALinear)
        self.assertIsInstance(model.main_model.model.blocks[0].attn.key, nn.Linear)
        self.assertIsInstance(model.main_model.model.blocks[0].attn.proj, nn.Linear)

    def test_qv_ffn_combines_qv_and_existing_d_sites(self):
        self.assertEqual(_inject(_gram(), "qv_ffn"), QV_SITES + FFN_SITES)

    def test_module_d_keeps_existing_ffn_surface(self):
        self.assertEqual(_inject(_gram(), "semantic"), FFN_SITES)

    def test_module_e_keeps_existing_structural_surface(self):
        self.assertEqual(_inject(_gram(), "structural"), STRUCTURAL_SITES)

    def test_module_c_e_matches_module_e_structural_surface(self):
        self.assertEqual(
            _inject(_gram(), "module_c", module_c_selected=("E",)),
            STRUCTURAL_SITES,
        )

    def test_module_e_spatial_temporal_variants_keep_prior_gram_surface(self):
        for target in ("spatial_attn", "temporal_attn"):
            with self.subTest(target=target):
                self.assertEqual(_inject(_gram(), target), [])


if __name__ == "__main__":
    unittest.main()
