import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - depends on the training env.
    torch = None
    nn = None

if torch is not None:
    from util.lora import apply_lora_to_eegfm


if torch is not None:
    class _TinyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(4, 4)
            self.key = nn.Linear(4, 4)
            self.value = nn.Linear(4, 4)
            self.proj = nn.Linear(4, 4)


    class _TinyMlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)


    class _TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _TinyAttention()
            self.mlp = _TinyMlp()


    class _TinyGram(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_channels = 4
            self.main_model = nn.Module()
            self.main_model.blocks = nn.ModuleList([_TinyBlock()])
            self.task_head = nn.Linear(4, 3)

        def forward(self, x):
            return self.task_head(torch.zeros(x.shape[0], 4, device=x.device))


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class ModuleCLoraInterfaceTests(unittest.TestCase):
    def test_module_c_bde_injects_b_d_and_e_surfaces_for_gram(self):
        model = _TinyGram()

        replaced = apply_lora_to_eegfm(
            model=model,
            model_name="Gram",
            lora_target="module_c",
            module_c_selected="B,D,E",
            r=2,
            alpha=4.0,
            dropout=0.0,
            verbose=False,
        )

        self.assertIn("input_side_lora", replaced)
        self.assertTrue(any("attn.query" in name for name in replaced), replaced)
        self.assertTrue(any("mlp.fc1" in name for name in replaced), replaced)


if __name__ == "__main__":
    unittest.main()
