import os
import tempfile
import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # pragma: no cover - depends on the training env.
    torch = None
    nn = None

if torch is not None:
    from util.module_c_preflight_policy import run_module_c_preflight_selection


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

        def forward(self, samples):
            hidden = samples.mean(dim=-1)
            block = self.main_model.blocks[0]
            hidden = torch.tanh(block.attn.query(hidden) + block.attn.value(hidden))
            hidden = torch.tanh(block.mlp.fc1(hidden))
            hidden = block.mlp.fc2(hidden)
            return self.task_head(hidden)


    class _TinyBinaryGram(_TinyGram):
        def __init__(self):
            super().__init__()
            self.task_head = nn.Linear(4, 1)


@unittest.skipIf(torch is None, "torch is not installed in this Python environment")
class ModuleCPreflightSmokeTests(unittest.TestCase):
    def test_bde_preflight_runs_through_real_lora_injection_without_test_data(self):
        torch.manual_seed(7)
        samples = torch.randn(12, 4, 5)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        support_loader = DataLoader(TensorDataset(samples[:6], labels[:6]), batch_size=3, shuffle=False)
        validation_loader = DataLoader(TensorDataset(samples[6:], labels[6:]), batch_size=3, shuffle=False)

        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                task_mod="Classification",
                nb_classes=3,
                norm_method="",
                mv_norm_value=0.01,
                model_name="Gram",
                module_c_candidates="B,D,E",
                module_b_sites="input",
                lora_rank=2,
                lora_alpha=4.0,
                opt="adamw",
                lr=0.05,
                weight_decay=0.0,
                opt_eps=None,
                opt_betas=None,
                momentum=0.9,
                clip_grad=None,
                module_c_preflight_train_batches=0,
                module_c_preflight_val_batches=0,
                output_dir=output_dir,
                dataset="tiny",
                subject_mod="fewshot",
                k_shot=1,
                seed=7,
            )

            result = run_module_c_preflight_selection(
                args=args,
                model=_TinyGram(),
                data_loader_train=support_loader,
                data_loader_val=validation_loader,
                device=torch.device("cpu"),
            )

            self.assertTrue(result.decision.selected_modules)
            self.assertTrue(set(result.decision.selected_modules).issubset({"B", "D", "E"}))
            self.assertEqual(args.module_c_resolved_selected, ",".join(result.decision.selected_modules))
            self.assertEqual(result.diagnostics_by_module["E"]["structural_reference_used_for_ranking"], 0)
            self.assertTrue(os.path.exists(result.score_csv_path))
            self.assertTrue(os.path.exists(result.decision_json_path))

    def test_bce_binary_preflight_uses_both_validation_labels(self):
        torch.manual_seed(11)
        samples = torch.randn(8, 4, 5)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        support_loader = DataLoader(TensorDataset(samples[:4], labels[:4]), batch_size=2, shuffle=False)
        validation_loader = DataLoader(TensorDataset(samples[4:], labels[4:]), batch_size=2, shuffle=False)

        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                task_mod="Classification",
                nb_classes=1,
                norm_method="",
                mv_norm_value=0.01,
                model_name="Gram",
                module_c_candidates="B,D,E",
                module_b_sites="input",
                lora_rank=2,
                lora_alpha=4.0,
                opt="adamw",
                lr=0.05,
                weight_decay=0.0,
                opt_eps=None,
                opt_betas=None,
                momentum=0.9,
                clip_grad=None,
                module_c_preflight_train_batches=0,
                module_c_preflight_val_batches=0,
                output_dir=output_dir,
                dataset="tiny_binary",
                subject_mod="fewshot",
                k_shot=1,
                seed=11,
            )

            result = run_module_c_preflight_selection(
                args=args,
                model=_TinyBinaryGram(),
                data_loader_train=support_loader,
                data_loader_val=validation_loader,
                device=torch.device("cpu"),
            )

            self.assertTrue(result.decision.selected_modules)
            self.assertEqual(set(result.decision.per_class_effect), {0, 1})
            self.assertEqual(args.module_c_resolved_selected, ",".join(result.decision.selected_modules))


if __name__ == "__main__":
    unittest.main()
