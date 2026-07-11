import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - depends on the training env.
    torch = None
    nn = None

if torch is not None:
    from run_finetuning import _apply_lora_training_setup
    from util.module_c_preflight_policy import (
        _configure_branch_trainability,
        install_module_c_action_registry,
    )


if torch is not None:
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
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)


    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attention()
            self.mlp = _Mlp()


    class _TinyGram(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_channels = 4
            self.main_model = nn.Module()
            self.main_model.blocks = nn.ModuleList([_Block()])
            self.task_head = nn.Linear(4, 3)


    class _BiotLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(4, 4)
            self.to_k = nn.Linear(4, 4)
            self.to_v = nn.Linear(4, 4)
            self.to_out = nn.Linear(4, 4)
            self.w1 = nn.Linear(4, 8)
            self.w2 = nn.Linear(8, 4)


    class _TinyBIOT(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_channels = 4
            self.chan_conv = nn.Conv1d(4, 4, kernel_size=1)
            self.main_model = nn.Module()
            self.main_model.layers = nn.ModuleList([_BiotLayer()])
            self.task_head = nn.Linear(4, 3)
            self.integer_state = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)


    class _MergedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 12)
            self.proj = nn.Linear(4, 4)


    class _MergedBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _MergedAttention()
            self.mlp = _Mlp()


    class _TinyMergedTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_channels = 4
            self.main_model = nn.Module()
            self.main_model.blocks = nn.ModuleList([_MergedBlock()])
            self.task_head = nn.Linear(4, 3)


    class _DualAttentionLayer(nn.Module):
        def __init__(self, csbrain=False):
            super().__init__()
            if csbrain:
                self.inter_window_attn = nn.MultiheadAttention(4, 1, batch_first=True)
                self.inter_region_attn = nn.MultiheadAttention(4, 1, batch_first=True)
            else:
                self.self_attn_s = nn.MultiheadAttention(4, 1, batch_first=True)
                self.self_attn_t = nn.MultiheadAttention(4, 1, batch_first=True)
            self.linear1 = nn.Linear(4, 8)
            self.linear2 = nn.Linear(8, 4)


    class _TinyDualAttention(nn.Module):
        def __init__(self, csbrain=False):
            super().__init__()
            self.input_channels = 4
            self.main_model = nn.Module()
            self.main_model.layers = nn.ModuleList([_DualAttentionLayer(csbrain=csbrain)])
            self.task_head = nn.Linear(4, 3)


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class ModuleCLoraInterfaceTests(unittest.TestCase):
    def _audit(self, model, model_name):
        ownership = install_module_c_action_registry(
            model=model,
            model_name=model_name,
            candidate_modules=("B", "D", "E"),
            module_b_sites="both",
            r=2,
            alpha=4.0,
            dropout=0.0,
        )
        self.assertEqual(set(ownership.action_parameter_names), {"B", "D", "E"})
        self.assertTrue(all(ownership.action_parameter_names[action] for action in ("B", "D", "E")))
        self.assertTrue(all(ownership.action_replacement_names[action] for action in ("B", "D", "E")))
        all_names = [
            name
            for action in ("B", "D", "E")
            for name in ownership.action_parameter_names[action]
        ]
        self.assertEqual(len(all_names), len(set(all_names)))
        self.assertEqual(set(all_names), set(ownership.adapter_parameter_owner))
        return ownership

    def test_biot_assigns_qv_to_e_and_ffn_to_d(self):
        model = _TinyBIOT()
        ownership = self._audit(model, "BIOT")

        self.assertTrue(any("to_q" in name or "to_v" in name for name in ownership.action_replacement_names["E"]))
        self.assertFalse(any("w1" in name or "w2" in name for name in ownership.action_replacement_names["E"]))
        self.assertTrue(any("w1" in name or "w2" in name for name in ownership.action_replacement_names["D"]))

        args = SimpleNamespace(
            lora_base_update="full",
            lora_train_head=True,
            lora_train_chan_conv=False,
            model_name="BIOT",
        )
        _configure_branch_trainability(model, args, ("D",), ownership)
        named = dict(model.named_parameters())
        self.assertTrue(ownership.action_wrapped_base_parameter_names["D"])
        self.assertTrue(ownership.action_wrapped_base_parameter_names["E"])
        self.assertTrue(
            all(named[name].requires_grad for name in ownership.action_wrapped_base_parameter_names["D"])
        )
        self.assertTrue(
            all(named[name].requires_grad for name in ownership.action_wrapped_base_parameter_names["E"])
        )
        self.assertFalse(model.integer_state.requires_grad)

    def test_formal_full_update_restores_wrapped_base_trainability(self):
        model = _TinyBIOT()
        original_trainability = {id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()}
        args = SimpleNamespace(
            task_mod="Classification",
            model_name="BIOT",
            lora_target="module_c",
            module_c_selected="D",
            module_c_resolved_selected="D",
            module_b_sites="both",
            lora_base_update="full",
            lora_rank=2,
            lora_alpha=4.0,
            lora_dropout=0.0,
            lora_train_head=True,
            lora_train_chan_conv=False,
        )

        _apply_lora_training_setup(model, args)

        original_parameters = [
            parameter for parameter in model.parameters() if id(parameter) in original_trainability
        ]
        self.assertTrue(
            all(
                parameter.requires_grad == original_trainability[id(parameter)]
                for parameter in original_parameters
            )
        )
        self.assertTrue(
            all(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" in name)
        )
        self.assertFalse(model.integer_state.requires_grad)

    def test_gram_actions_are_nonempty_and_disjoint(self):
        ownership = self._audit(_TinyGram(), "Gram")

        self.assertTrue(any("attn" in name for name in ownership.action_replacement_names["E"]))
        self.assertTrue(any("mlp" in name for name in ownership.action_replacement_names["D"]))

    def test_eegpt_and_labram_actions_are_nonempty_and_disjoint(self):
        for model_name in ("EEGPT", "LaBraM"):
            with self.subTest(model_name=model_name):
                self._audit(_TinyMergedTransformer(), model_name)

    def test_cbramod_actions_are_nonempty_and_disjoint(self):
        self._audit(_TinyDualAttention(csbrain=False), "CBraMod")

    def test_csbrain_actions_are_nonempty_and_disjoint(self):
        self._audit(_TinyDualAttention(csbrain=True), "CSBrain")


if __name__ == "__main__":
    unittest.main()
