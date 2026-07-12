import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from util.utils import _validate_optimizer_resume_schema, auto_load_model


class OptimizerResumeCompatibilityTests(unittest.TestCase):
    def test_legacy_group_mismatch_fails_before_model_state_is_mutated(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()

        extra = torch.nn.Parameter(torch.zeros(()))
        current_optimizer = torch.optim.AdamW([
            {
                'params': [model.weight],
                'param_group_tag': 'module_e:mixing',
            },
            {
                'params': [model.bias],
                'param_group_tag': 'non_module_e',
            },
            {
                'params': [extra],
                'param_group_tag': 'module_e:spatial',
            },
        ])

        legacy_model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            legacy_model.weight.fill_(1.0)
            legacy_model.bias.fill_(1.0)
        legacy_optimizer = torch.optim.AdamW([
            {'params': [legacy_model.weight]},
            {'params': [legacy_model.bias]},
        ])

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'checkpoint.pth'
            torch.save(
                {
                    'model': legacy_model.state_dict(),
                    'optimizer': legacy_optimizer.state_dict(),
                    'epoch': 3,
                },
                checkpoint_path,
            )
            args = SimpleNamespace(
                output_dir=tmp_dir,
                enable_deepspeed=False,
                auto_resume=False,
                resume=str(checkpoint_path),
                start_epoch=0,
                model_ema=False,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                r'legacy optimizer.*saved_groups=2.*current_groups=3.*--no_auto_resume',
            ):
                auto_load_model(
                    args,
                    model,
                    model,
                    current_optimizer,
                    loss_scaler=None,
                )

        self.assertTrue(torch.equal(model.weight, torch.zeros_like(model.weight)))
        self.assertTrue(torch.equal(model.bias, torch.zeros_like(model.bias)))

    def test_missing_module_e_group_tags_are_rejected_even_when_counts_match(self):
        p1 = torch.nn.Parameter(torch.zeros(()))
        p2 = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([
            {'params': [p1], 'param_group_tag': 'module_e:mixing'},
            {'params': [p2], 'param_group_tag': 'non_module_e'},
        ])
        legacy_state = optimizer.state_dict()
        for group in legacy_state['param_groups']:
            group.pop('param_group_tag', None)

        with self.assertRaisesRegex(RuntimeError, r'legacy optimizer.*group tags'):
            _validate_optimizer_resume_schema(
                optimizer,
                legacy_state,
                'legacy-checkpoint.pth',
            )

    def test_current_tagged_optimizer_schema_is_accepted(self):
        p1 = torch.nn.Parameter(torch.zeros(()))
        p2 = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([
            {'params': [p1], 'param_group_tag': 'module_e:mixing'},
            {'params': [p2], 'param_group_tag': 'non_module_e'},
        ])

        self.assertIsNone(
            _validate_optimizer_resume_schema(
                optimizer,
                optimizer.state_dict(),
                'current-checkpoint.pth',
            )
        )


if __name__ == '__main__':
    unittest.main()
