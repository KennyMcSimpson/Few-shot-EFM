import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

import run_finetuning
from util.fb_registry import classify_param_name
from util.lora import apply_lora_to_eegfm
from util.module_c_preflight_policy import _create_probe_optimizer, _run_support_pass
from util.module_e_structural_routing import (
    ModuleEDynamicPressureController,
    attach_module_e_dynamic_pressure_controller,
)
from util.optim_factory import get_parameter_groups
from util.utils import NativeScalerWithGradNormCount


class _TinyBIOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_model = nn.Module()
        self.main_model.transformer = nn.Module()
        self.main_model.transformer.layers = nn.ModuleList([nn.Module()])
        attention = nn.Module()
        attention.to_q = nn.Linear(4, 4)
        attention.to_k = nn.Linear(4, 4)
        attention.to_v = nn.Linear(4, 4)
        attention.to_out = nn.Linear(4, 4)
        self.main_model.transformer.layers[0].attention = attention
        self.task_head = nn.Linear(4, 2)


class _TwoBranchAdapters(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.lora_A = nn.Parameter(torch.ones(2, 2))
        self.attention.lora_B = nn.Parameter(torch.ones(2, 2))
        self.spatial_attention = nn.Module()
        self.spatial_attention.lora_A = nn.Parameter(torch.ones(2, 2))
        self.spatial_attention.lora_B = nn.Parameter(torch.ones(2, 2))
        self.head = nn.Parameter(torch.ones(2))


class _ProbeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 2)

    def forward(self, samples):
        return self.projection(samples)


class _RecordingController:
    def __init__(self):
        self.events = []
        self.bound_optimizer = None

    def optimizer_group_tag(self, _name):
        return "non_module_e"

    def bind_optimizer(self, optimizer):
        self.events.append("bind")
        self.bound_optimizer = optimizer

    def prepare_optimizer_step(self, optimizer, global_step=None, epoch=None):
        self.events.append(("prepare", optimizer, global_step, epoch))

    def finish_optimizer_step(self, optimizer, step_applied=True):
        self.events.append(("finish", optimizer, step_applied))


class _FailingPrepareController(_RecordingController):
    def prepare_optimizer_step(self, optimizer, global_step=None, epoch=None):
        self.events.append(("prepare", optimizer, global_step, epoch))
        raise RuntimeError("probe prepare failed")


class _FakeScaledLoss:
    def __init__(self, events, parameter):
        self.events = events
        self.parameter = parameter

    def backward(self, create_graph=False):
        del create_graph
        self.events.append("backward")
        self.parameter.grad = torch.ones_like(self.parameter)


class _FakeGradScaler:
    def __init__(self, events, parameter, fail_step=False, skip_step=False, fail_update=False):
        self.events = events
        self.parameter = parameter
        self.fail_step = fail_step
        self.skip_step = skip_step
        self.fail_update = fail_update

    def scale(self, _loss):
        return _FakeScaledLoss(self.events, self.parameter)

    def unscale_(self, _optimizer):
        self.events.append("unscale")

    def step(self, optimizer):
        self.events.append("step")
        if self.fail_step:
            raise RuntimeError("step failed")
        if self.skip_step:
            return None
        return optimizer.step()

    def update(self):
        self.events.append("update")
        if self.fail_update:
            raise RuntimeError("update failed")


class _NoPostHookOptimizer:
    def __init__(self, parameter):
        self.param_groups = [{"params": [parameter], "lr": 0.1}]

    def step(self):
        raise AssertionError("optimizer.step must not run without post-hook support")


class _FailOnceLrGroup(dict):
    def __init__(self, source):
        super().__init__(source)
        self.fail_next_lr_write = True

    def __setitem__(self, key, value):
        if key == "lr" and self.fail_next_lr_write:
            self.fail_next_lr_write = False
            raise RuntimeError("prepare lr write failed")
        return super().__setitem__(key, value)


def _module_e_args(replaced):
    return SimpleNamespace(
        model_name="BIOT",
        output_dir="",
        run_tag="test",
        module_e_injected_names=";".join(replaced),
        module_e_pressure_beta=0.0,
        module_e_gate_temperature=0.1,
        module_e_gate_floor=0.0,
        module_e_scale_min=0.5,
        module_e_scale_max=1.5,
        module_e_warmup_steps=0,
        module_e_diag_freq=1,
    )


class ModuleEDynamicPressureTests(unittest.TestCase):
    def test_biot_attention_projection_segments_map_to_mixing(self):
        for segment in ("to_q", "to_k", "to_v", "to_out"):
            primary, hits = classify_param_name(
                "BIOT", f"main_model.transformer.layers.0.attention.{segment}.weight"
            )
            self.assertEqual(primary, "mixing")
            self.assertIn("mixing", hits)

        primary, hits = classify_param_name("BIOT", "main_model.not_to_query.weight")
        self.assertNotEqual(primary, "mixing")
        self.assertNotIn("mixing", hits)

    def test_real_biot_structural_attachment_controls_lora_a_and_b(self):
        model = _TinyBIOT()
        replaced = apply_lora_to_eegfm(
            model,
            "BIOT",
            lora_target="struct_mix",
            r=2,
            alpha=4.0,
            dropout=0.0,
            verbose=False,
        )
        self.assertEqual(
            replaced,
            [
                "main_model.transformer.layers.0.attention.to_q",
                "main_model.transformer.layers.0.attention.to_v",
            ],
        )

        controller = attach_module_e_dynamic_pressure_controller(
            _module_e_args(replaced), model
        )
        controlled = set(controller.controlled_parameter_names())
        for replacement in replaced:
            self.assertIn(f"{replacement}.lora_A", controlled)
            self.assertIn(f"{replacement}.lora_B", controlled)

    def test_requested_e_without_structural_coverage_raises(self):
        model = nn.Linear(4, 2)
        with self.assertRaisesRegex(RuntimeError, "Module E.*replacement|coverage"):
            attach_module_e_dynamic_pressure_controller(
                _module_e_args(["main_model.missing.to_q"]), model
            )

    def test_gradient_observer_returns_original_values(self):
        controller = ModuleEDynamicPressureController("BIOT", branches=("mixing",))
        parameter = nn.Parameter(torch.ones(2, 2))
        controller.register_param(
            "attention.to_q.lora_B", "mixing", parameter, is_pressure_param=True
        )
        controller._branch_scale["mixing"] = 0.5
        gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])

        observed = controller.make_gradient_hook("attention.to_q.lora_B")(gradient)

        self.assertTrue(torch.equal(observed, gradient))
        self.assertEqual(controller._pending_param_tensors["mixing"], 1)

    def test_optimizer_groups_isolate_e_branches_and_non_e_parameters(self):
        model = _TwoBranchAdapters()
        controller = ModuleEDynamicPressureController("EEGPT")
        named = dict(model.named_parameters())
        controller.register_param(
            "attention.lora_A", "mixing", named["attention.lora_A"]
        )
        controller.register_param(
            "attention.lora_B",
            "mixing",
            named["attention.lora_B"],
            is_pressure_param=True,
        )
        controller.register_param(
            "spatial_attention.lora_A", "spatial", named["spatial_attention.lora_A"]
        )
        controller.register_param(
            "spatial_attention.lora_B",
            "spatial",
            named["spatial_attention.lora_B"],
            is_pressure_param=True,
        )

        groups = get_parameter_groups(
            model,
            weight_decay=0.01,
            get_param_group_tag=controller.optimizer_group_tag,
        )
        tags = {group["param_group_tag"] for group in groups}

        self.assertEqual(tags, {"module_e:mixing", "module_e:spatial", "non_module_e"})
        for group in groups:
            owned_branches = {
                controller.branch_for_parameter(parameter)
                for parameter in group["params"]
                if controller.branch_for_parameter(parameter) is not None
            }
            self.assertLessEqual(len(owned_branches), 1)
            if owned_branches:
                self.assertTrue(all(parameter is not model.head for parameter in group["params"]))

    def test_one_active_branch_uses_exact_unit_multiplier(self):
        parameter_a = nn.Parameter(torch.ones(2, 2))
        parameter_b = nn.Parameter(torch.ones(2, 2))
        controller = ModuleEDynamicPressureController(
            "BIOT", beta=0.0, branches=("mixing",)
        )
        controller.register_param("attention.to_q.lora_A", "mixing", parameter_a)
        controller.register_param(
            "attention.to_q.lora_B", "mixing", parameter_b, is_pressure_param=True
        )
        optimizer = torch.optim.SGD(
            [{"params": [parameter_a, parameter_b], "param_group_tag": "module_e:mixing"}],
            lr=0.1,
        )
        controller.bind_optimizer(optimizer)
        parameter_a.grad = torch.ones_like(parameter_a)
        parameter_b.grad = torch.full_like(parameter_b, 9.0)

        controller.prepare_optimizer_step(optimizer)
        self.assertEqual(controller.scale_for_branch("mixing"), 1.0)
        optimizer.step()
        row = controller.finish_optimizer_step(optimizer)

        self.assertEqual(row["optimizer_lr_multiplier_mixing"], 1.0)
        self.assertEqual(row["allocation_degenerate"], 1)

    def test_lr_multiplier_controls_real_adamw_delta_and_is_restored(self):
        model = _TwoBranchAdapters()
        controller = ModuleEDynamicPressureController(
            "EEGPT",
            beta=0.0,
            temperature=0.01,
            gate_floor=0.0,
            scale_min=0.5,
            scale_max=1.5,
        )
        named = dict(model.named_parameters())
        for name, branch in (
            ("attention.lora_A", "mixing"),
            ("attention.lora_B", "mixing"),
            ("spatial_attention.lora_A", "spatial"),
            ("spatial_attention.lora_B", "spatial"),
        ):
            controller.register_param(
                name, branch, named[name], is_pressure_param=name.endswith("lora_B")
            )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [named["attention.lora_A"], named["attention.lora_B"]],
                    "lr": 0.1,
                    "weight_decay": 0.0,
                    "param_group_tag": "module_e:mixing",
                },
                {
                    "params": [
                        named["spatial_attention.lora_A"],
                        named["spatial_attention.lora_B"],
                    ],
                    "lr": 0.1,
                    "weight_decay": 0.0,
                    "param_group_tag": "module_e:spatial",
                },
                {
                    "params": [model.head],
                    "lr": 0.1,
                    "weight_decay": 0.0,
                    "param_group_tag": "non_module_e",
                },
            ]
        )
        controller.bind_optimizer(optimizer)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        named["attention.lora_B"].grad.fill_(0.01)
        named["spatial_attention.lora_B"].grad.fill_(10.0)
        before_mixing = named["attention.lora_A"].detach().clone()
        before_spatial = named["spatial_attention.lora_A"].detach().clone()

        controller.prepare_optimizer_step(optimizer, global_step=3, epoch=2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.15)
        optimizer.step()
        row = controller.finish_optimizer_step(optimizer)

        mixing_delta = torch.linalg.vector_norm(
            named["attention.lora_A"].detach() - before_mixing
        ).item()
        spatial_delta = torch.linalg.vector_norm(
            named["spatial_attention.lora_A"].detach() - before_spatial
        ).item()
        self.assertGreater(spatial_delta, mixing_delta)
        self.assertAlmostEqual(spatial_delta / mixing_delta, 3.0, delta=0.05)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.1, 0.1, 0.1])
        self.assertGreater(row["actual_update_norm_mixing"], 0.0)
        self.assertGreater(row["actual_update_norm_spatial"], 0.0)
        self.assertNotIn("gradient_multiplier", " ".join(row))

        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        named["attention.lora_B"].grad.fill_(0.01)
        named["spatial_attention.lora_B"].grad.fill_(10.0)
        controller.prepare_optimizer_step(optimizer)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        controller.finish_optimizer_step(optimizer, step_applied=False)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.1, 0.1, 0.1])

    def test_native_scaler_callback_order_and_non_update_contract(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = []
        scaler = object.__new__(NativeScalerWithGradNormCount)
        scaler._scaler = _FakeGradScaler(events, parameter)

        def before():
            events.append("before")

        def after(step_applied=True):
            events.append(f"after:{int(step_applied)}")

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=lambda *_args, **_kwargs: events.append("clip") or torch.tensor(1.0)):
            scaler(
                object(),
                optimizer,
                clip_grad=1.0,
                parameters=[parameter],
                update_grad=True,
                before_optimizer_step=before,
                after_optimizer_step=after,
            )
        self.assertEqual(
            events,
            ["backward", "unscale", "before", "clip", "step", "after:1", "update"],
        )

        events.clear()
        parameter.grad = None
        scaler(
            object(),
            optimizer,
            parameters=[parameter],
            update_grad=False,
            before_optimizer_step=before,
            after_optimizer_step=after,
        )
        self.assertEqual(events, ["backward"])

    def test_native_scaler_runs_after_callback_when_step_raises(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = []
        scaler = object.__new__(NativeScalerWithGradNormCount)
        scaler._scaler = _FakeGradScaler(events, parameter, fail_step=True)

        def before():
            events.append("before")
            optimizer.param_groups[0]["lr"] = 0.5

        def after(step_applied=True):
            events.append(f"after:{int(step_applied)}")
            optimizer.param_groups[0]["lr"] = 0.1

        with self.assertRaisesRegex(RuntimeError, "step failed"):
            scaler(
                object(),
                optimizer,
                parameters=[parameter],
                update_grad=True,
                before_optimizer_step=before,
                after_optimizer_step=after,
            )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(events, ["backward", "unscale", "before", "step", "after:0"])

    def test_native_scaler_reports_silent_amp_skip_and_restores_lr(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = []
        scaler = object.__new__(NativeScalerWithGradNormCount)
        scaler._scaler = _FakeGradScaler(events, parameter, skip_step=True)

        def before():
            events.append("before")
            optimizer.param_groups[0]["lr"] = 0.5

        def after(step_applied=True):
            events.append(f"after:{int(step_applied)}")
            optimizer.param_groups[0]["lr"] = 0.1

        scaler(
            object(),
            optimizer,
            parameters=[parameter],
            update_grad=True,
            before_optimizer_step=before,
            after_optimizer_step=after,
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(
            events, ["backward", "unscale", "before", "step", "after:0", "update"]
        )

    def test_native_scaler_observes_real_optimizer_step_with_none_return(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = []
        scaler = object.__new__(NativeScalerWithGradNormCount)
        scaler._scaler = _FakeGradScaler(events, parameter)

        scaler(
            object(),
            optimizer,
            parameters=[parameter],
            update_grad=True,
            before_optimizer_step=lambda: events.append("before"),
            after_optimizer_step=lambda step_applied=True: events.append(
                f"after:{int(step_applied)}"
            ),
        )

        self.assertEqual(events[-2:], ["after:1", "update"])

    def test_native_scaler_restores_before_update_failure_after_real_step(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = []
        scaler = object.__new__(NativeScalerWithGradNormCount)
        scaler._scaler = _FakeGradScaler(events, parameter, fail_update=True)

        def before():
            events.append("before")
            optimizer.param_groups[0]["lr"] = 0.5

        def after(step_applied=True):
            events.append(f"after:{int(step_applied)}")
            optimizer.param_groups[0]["lr"] = 0.1

        with self.assertRaisesRegex(RuntimeError, "update failed"):
            scaler(
                object(),
                optimizer,
                parameters=[parameter],
                update_grad=True,
                before_optimizer_step=before,
                after_optimizer_step=after,
            )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(
            events, ["backward", "unscale", "before", "step", "after:1", "update"]
        )

    def test_native_scaler_requires_public_post_step_hook_for_after_callback(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = _NoPostHookOptimizer(parameter)
        events = []
        scaler = object.__new__(NativeScalerWithGradNormCount)
        scaler._scaler = _FakeGradScaler(events, parameter)

        def before():
            events.append("before")
            optimizer.param_groups[0]["lr"] = 0.5

        def after(step_applied=True):
            events.append(f"after:{int(step_applied)}")
            optimizer.param_groups[0]["lr"] = 0.1

        with self.assertRaisesRegex(RuntimeError, "register_step_post_hook"):
            scaler(
                object(),
                optimizer,
                parameters=[parameter],
                update_grad=True,
                before_optimizer_step=before,
                after_optimizer_step=after,
            )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.1)
        self.assertEqual(events, ["backward", "unscale", "before", "after:0"])

    def test_native_scaler_does_not_finish_when_prepare_callback_raises(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        events = []
        scaler = object.__new__(NativeScalerWithGradNormCount)
        scaler._scaler = _FakeGradScaler(events, parameter)

        def before():
            events.append("before")
            raise RuntimeError("prepare callback failed")

        with self.assertRaisesRegex(RuntimeError, "prepare callback failed"):
            scaler(
                object(),
                optimizer,
                parameters=[parameter],
                update_grad=True,
                before_optimizer_step=before,
                after_optimizer_step=lambda step_applied=True: events.append("after"),
            )

        self.assertEqual(events, ["backward", "unscale", "before"])

    def test_prepare_optimizer_step_rolls_back_partial_lr_writes(self):
        model = _TwoBranchAdapters()
        controller = ModuleEDynamicPressureController(
            "EEGPT",
            beta=0.0,
            temperature=0.01,
            gate_floor=0.0,
            scale_min=0.5,
            scale_max=1.5,
        )
        named = dict(model.named_parameters())
        for name, branch in (
            ("attention.lora_A", "mixing"),
            ("attention.lora_B", "mixing"),
            ("spatial_attention.lora_A", "spatial"),
            ("spatial_attention.lora_B", "spatial"),
        ):
            controller.register_param(
                name, branch, named[name], is_pressure_param=name.endswith("lora_B")
            )
        optimizer = torch.optim.SGD(
            [
                {
                    "params": [named["attention.lora_A"], named["attention.lora_B"]],
                    "lr": 0.1,
                    "param_group_tag": "module_e:mixing",
                },
                {
                    "params": [
                        named["spatial_attention.lora_A"],
                        named["spatial_attention.lora_B"],
                    ],
                    "lr": 0.1,
                    "param_group_tag": "module_e:spatial",
                },
                {
                    "params": [model.head],
                    "lr": 0.1,
                    "param_group_tag": "non_module_e",
                },
            ]
        )
        controller.bind_optimizer(optimizer)
        optimizer.param_groups[1] = _FailOnceLrGroup(optimizer.param_groups[1])
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        named["attention.lora_B"].grad.fill_(0.01)
        named["spatial_attention.lora_B"].grad.fill_(10.0)

        with self.assertRaisesRegex(RuntimeError, "prepare lr write failed"):
            controller.prepare_optimizer_step(optimizer)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.1, 0.1, 0.1])
        self.assertIsNone(controller._prepared_optimizer)
        self.assertEqual(controller._stored_group_lrs, [])
        self.assertEqual(controller._parameter_snapshots, {})

    def test_module_c_probe_uses_controller_group_and_step_contract(self):
        args = SimpleNamespace(
            model_name="Tiny",
            layer_decay=1.0,
            opt="sgd",
            lr=0.1,
            weight_decay=0.01,
            opt_eps=None,
            opt_betas=None,
            momentum=0.9,
            update_freq=1,
            clip_grad=None,
            norm_method="",
            nb_classes=2,
        )
        model = _ProbeModel()
        controller = _RecordingController()

        optimizer = _create_probe_optimizer(args, model, controller=controller)
        self.assertIs(controller.bound_optimizer, optimizer)
        self.assertTrue(
            all(group["param_group_tag"] == "non_module_e" for group in optimizer.param_groups)
        )

        controller.events.clear()
        batches = [(torch.randn(3, 4), torch.tensor([0, 1, 0]))]
        _run_support_pass(
            args,
            model,
            batches,
            torch.device("cpu"),
            nn.CrossEntropyLoss(),
            controller=controller,
        )
        self.assertEqual(controller.events[0], "bind")
        self.assertEqual(controller.events[1][0], "prepare")
        self.assertEqual(controller.events[2][0], "finish")
        self.assertTrue(controller.events[2][2])

    def test_module_c_probe_preserves_prepare_failure_without_finishing(self):
        args = SimpleNamespace(
            model_name="Tiny",
            layer_decay=1.0,
            opt="sgd",
            lr=0.1,
            weight_decay=0.01,
            opt_eps=None,
            opt_betas=None,
            momentum=0.9,
            update_freq=1,
            clip_grad=None,
            norm_method="",
            nb_classes=2,
        )
        model = _ProbeModel()
        controller = _FailingPrepareController()
        batches = [(torch.randn(3, 4), torch.tensor([0, 1, 0]))]

        with self.assertRaisesRegex(RuntimeError, "probe prepare failed"):
            _run_support_pass(
                args,
                model,
                batches,
                torch.device("cpu"),
                nn.CrossEntropyLoss(),
                controller=controller,
            )

        self.assertEqual(controller.events[0], "bind")
        self.assertEqual(controller.events[1][0], "prepare")
        self.assertFalse(any(event[0] == "finish" for event in controller.events[1:]))

    def test_formal_optimizer_creation_uses_controller_group_and_bind_contract(self):
        args = argparse.Namespace(
            opt="sgd",
            lr=0.1,
            weight_decay=0.01,
            opt_eps=None,
            opt_betas=None,
            momentum=0.9,
        )
        model = _ProbeModel()
        controller = _RecordingController()

        optimizer = run_finetuning._create_formal_optimizer(
            args,
            model,
            skip_weight_decay_list=[],
            assigner=None,
            controller=controller,
        )

        self.assertIs(controller.bound_optimizer, optimizer)
        self.assertTrue(
            all(group["param_group_tag"] == "non_module_e" for group in optimizer.param_groups)
        )

    def test_requested_formal_e_attachment_failure_is_fatal(self):
        args = argparse.Namespace(
            finetune_mod="lora",
            lora_target="struct_mix",
            module_e_mode="dynamic_pressure_gate",
            module_c_selected="",
            module_c_resolved_selected="",
            enable_deepspeed=False,
        )
        with patch.object(
            run_finetuning, "attach_module_e_dynamic_pressure_controller", return_value=None
        ):
            with self.assertRaisesRegex(RuntimeError, "Module E"):
                run_finetuning._attach_requested_module_e_controller(args, _ProbeModel())

        args.enable_deepspeed = True
        with self.assertRaisesRegex(RuntimeError, "DeepSpeed"):
            run_finetuning._attach_requested_module_e_controller(args, _ProbeModel())


if __name__ == "__main__":
    unittest.main()
