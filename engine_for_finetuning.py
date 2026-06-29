# --------------------------------------------------------
# Based on LaBraM, EEGPT, CBraMod, BIOT, EEG_Image_decode, BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/935963004/LaBraM
# https://github.com/BINE022/EEGPT/tree/main/downstream
# https://github.com/wjq-learning/CBraMod
# https://github.com/ycq091044/BIOT
# https://github.com/ncclab-sustech/EEG_Image_decode
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import os
import math
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
from timm.loss import LabelSmoothingCrossEntropy
import util.utils as utils
from util.utils import wandb_logger
import random
import matplotlib.pyplot as plt
import numpy as np


def _get_class_weight_tensor(args, device):
    """Return class-weight tensor prepared in run_finetuning.py, or None.

    args.class_weights is intentionally a plain Python list so it can be saved
    into args/log files safely.
    """
    weights = getattr(args, "class_weights", None)
    if weights is None:
        return None
    if len(weights) == 0:
        return None
    return torch.tensor(weights, dtype=torch.float32, device=device)


class SoftBalancedCrossEntropy(torch.nn.Module):
    """CE mixed with a class-weighted CE term.

    loss = (1 - lambda) * CE + lambda * WeightedCE
    lambda=0 gives plain CE; lambda=1 gives weighted CE.
    """

    def __init__(self, weight=None, balance_lambda=1.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.balance_lambda = float(balance_lambda)
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, target):
        lam = max(0.0, min(1.0, self.balance_lambda))
        if self.weight is None or lam <= 0.0:
            return self.ce(logits, target)
        ce = self.ce(logits, target)
        wce = torch.nn.functional.cross_entropy(logits, target, weight=self.weight)
        return (1.0 - lam) * ce + lam * wce


def _annealed_balance_lambda(args, epoch_id: int) -> float:
    base = float(getattr(args, "class_balance_lambda", 1.0))
    floor = float(getattr(args, "class_balance_anneal_floor", 0.0))
    start = int(getattr(args, "class_balance_anneal_start_epoch", 3))
    end = int(getattr(args, "class_balance_anneal_end_epoch", 8))
    base = max(0.0, min(1.0, base))
    floor = max(0.0, min(base, floor))
    if epoch_id <= start:
        return base
    if epoch_id >= end:
        return floor
    denom = max(float(end - start), 1.0)
    ratio = float(end - epoch_id) / denom
    return floor + (base - floor) * max(0.0, min(1.0, ratio))


def _build_classification_criterion(args, device, is_binary=False, epoch_id: int = 1):
    """Build criterion for classification.

    Added loss_type options for model-aware LoRA experiments:
    - soft_sqrt_balanced_ce: CE + lambda * sqrt-weighted CE.
    - anneal_sqrt_balanced_ce: same, but lambda decays after a warm period.
    """
    loss_type = getattr(args, "loss_type", "ce")

    if is_binary:
        return torch.nn.BCEWithLogitsLoss()

    if getattr(args, "smoothing", 0.0) > 0.0 and loss_type == "ce":
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)

    if loss_type in ["balanced_ce", "sqrt_balanced_ce"]:
        weight = _get_class_weight_tensor(args, device)
        if weight is None:
            print(f"[Loss] loss_type={loss_type}, but no class_weights were found. Falling back to unweighted CE.")
            return torch.nn.CrossEntropyLoss()
        print(f"[Loss] Using {loss_type} with class_weights={weight.detach().cpu().tolist()}")
        return torch.nn.CrossEntropyLoss(weight=weight)

    if loss_type == "soft_sqrt_balanced_ce":
        weight = _get_class_weight_tensor(args, device)
        lam = float(getattr(args, "class_balance_lambda", 0.5))
        print(f"[Loss] Using soft_sqrt_balanced_ce, lambda={lam:.4f}, weights={None if weight is None else weight.detach().cpu().tolist()}")
        return SoftBalancedCrossEntropy(weight=weight, balance_lambda=lam)

    if loss_type == "anneal_sqrt_balanced_ce":
        weight = _get_class_weight_tensor(args, device)
        lam = _annealed_balance_lambda(args, epoch_id)
        print(f"[Loss] Using anneal_sqrt_balanced_ce, epoch={epoch_id}, lambda={lam:.4f}, weights={None if weight is None else weight.detach().cpu().tolist()}")
        return SoftBalancedCrossEntropy(weight=weight, balance_lambda=lam)

    if loss_type == "ce":
        return torch.nn.CrossEntropyLoss()

    raise ValueError(
        f"Unknown loss_type={loss_type}. Supported: ce, balanced_ce, sqrt_balanced_ce, "
        "soft_sqrt_balanced_ce, anneal_sqrt_balanced_ce"
    )



# ------------------------------- LoRA update control --------------------------------
def _as_lora_dict(x):
    """Return iterable (key, tensor) for Parameter / ParameterDict LoRA fields."""
    if isinstance(x, torch.nn.ParameterDict):
        return list(x.items())
    if isinstance(x, torch.nn.Parameter):
        return [("", x)]
    return []


def compute_lora_delta_control_loss(model: torch.nn.Module, args) -> tuple:
    """Compute a normalized LoRA delta penalty.

    This regularizes the *effective low-rank update* delta_W = B @ A rather
    than the raw LoRA parameters. It is intended for EEG few-shot runs where
    LoRA can reach a useful early checkpoint but then over-write useful FM
    representations or class boundaries.

    Returned values:
        penalty_loss: tensor added to the training loss
        penalty_raw: python float before multiplying by lambda
        n_terms: number of LoRA delta matrices included
    """
    lam = float(getattr(args, "lora_delta_lambda", 0.0))
    if lam <= 0.0:
        # Build a device-safe zero from model parameters when possible.
        try:
            ref = next(model.parameters())
            return ref.new_tensor(0.0), 0.0, 0
        except StopIteration:
            return torch.tensor(0.0), 0.0, 0

    mode = str(getattr(args, "lora_delta_mode", "relative_l2")).lower()
    eps = 1e-12
    penalties = []

    for module in model.modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        A_items = dict(_as_lora_dict(module.lora_A))
        B_items = dict(_as_lora_dict(module.lora_B))
        for key, A in A_items.items():
            if key not in B_items:
                continue
            B = B_items[key]
            # Linear / merged-QKV / MHA: A [r, in], B [out, r]
            # Conv1d 1x1 bridge: A [r, in, 1], B [out, r, 1]
            if A.dim() == 2 and B.dim() == 2:
                delta = B @ A
                base = getattr(module, "base", None)
                if base is not None and hasattr(base, "weight"):
                    base_norm = torch.norm(base.weight.detach().float()).to(delta.device)
                else:
                    base_norm = delta.detach().new_tensor(1.0)
            elif A.dim() == 3 and B.dim() == 3 and A.shape[-1] == 1 and B.shape[-1] == 1:
                delta = torch.matmul(B.squeeze(-1), A.squeeze(-1))
                base = getattr(module, "base", None)
                if base is not None and hasattr(base, "weight"):
                    base_norm = torch.norm(base.weight.detach().float().reshape(base.weight.shape[0], -1)).to(delta.device)
                else:
                    base_norm = delta.detach().new_tensor(1.0)
            else:
                continue

            delta = delta.float() * float(getattr(module, "scaling", 1.0))
            if mode == "absolute_l2":
                penalties.append(delta.pow(2).mean())
            else:
                # Default: normalized Frobenius penalty, scale-stable across layers.
                rel = torch.norm(delta) / (base_norm + eps)
                penalties.append(rel.pow(2))

    if len(penalties) == 0:
        try:
            ref = next(model.parameters())
            return ref.new_tensor(0.0), 0.0, 0
        except StopIteration:
            return torch.tensor(0.0), 0.0, 0

    raw = torch.stack(penalties).mean()
    return raw * lam, float(raw.detach().cpu().item()), len(penalties)



def _is_lora_param_name(name: str) -> bool:
    return "lora_" in name


def _is_head_param_name(name: str) -> bool:
    return "task_head" in name


def _ensure_lifecycle_grad_hooks(model: torch.nn.Module):
    """Register lightweight gradient-scale hooks for LoRA and task head.

    The hook values are changed epoch-by-epoch by apply_lora_lifecycle_controls.
    This avoids rebuilding optimizer parameter groups in the middle of training.
    """
    holder = getattr(model, "_lora_life_grad_holder", None)
    if holder is not None:
        return holder

    holder = {"lora_scale": 1.0, "head_scale": 1.0}
    lora_n = 0
    head_n = 0

    def make_hook(group_key):
        def _hook(grad):
            return grad * float(holder[group_key])
        return _hook

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_lora_param_name(name):
            param.register_hook(make_hook("lora_scale"))
            lora_n += 1
        elif _is_head_param_name(name):
            param.register_hook(make_hook("head_scale"))
            head_n += 1

    model._lora_life_grad_holder = holder
    print(f"[LoRA-Life] gradient hooks registered: lora={lora_n}, head={head_n}")
    return holder



def _is_bridge_lora_param_name(name: str) -> bool:
    return _is_lora_param_name(name) and "chan_conv" in name


def _is_ffn_lora_param_name(name: str) -> bool:
    if not _is_lora_param_name(name):
        return False
    lower = name.lower()
    return (
        "mlp.fc1" in lower or "mlp.fc2" in lower
        or "linear1" in lower or "linear2" in lower
        or lower.endswith("w1.lora_a") or lower.endswith("w1.lora_b")
        or lower.endswith("w2.lora_a") or lower.endswith("w2.lora_b")
    )


def _ensure_staged_bridge_grad_hooks(model: torch.nn.Module):
    holder = getattr(model, '_staged_bridge_grad_holder', None)
    if holder is not None:
        return holder
    holder = {'bridge_scale': 1.0}
    n = 0
    def make_hook():
        def _hook(grad):
            return grad * float(holder.get('bridge_scale', 1.0))
        return _hook
    for name, param in model.named_parameters():
        if param.requires_grad and _is_bridge_lora_param_name(name):
            param.register_hook(make_hook())
            n += 1
    model._staged_bridge_grad_holder = holder
    print(f'[Stage-LoRA] bridge gradient hooks registered: n={n}')
    return holder



def _is_cbra_norm_bias_name(name: str) -> bool:
    lower = str(name).lower()
    return (
        "norm" in lower
        or "layernorm" in lower
        or ".ln" in lower
        or lower.endswith(".bias")
    )


def _is_cbra_front_name(name: str) -> bool:
    lower = str(name).lower()
    return (
        "patch_embedding" in lower
        or "spectral" in lower
        or "positional" in lower
        or "position" in lower
    )


def _apply_cbra_staged_controls(model: torch.nn.Module, args, epoch_id: int) -> bool:
    """
    CBraMod-specific two-stage control.

    Motivation:
      c3pf showed patch/front + late FFN base can open class0/class2,
      but it hurts class1/class3. These staged modes test whether a second
      conservative deployment phase can keep hard-class gains while repairing
      global decision boundaries.

    Modes:
      - two_stage_lora_norm:
          Stage1: use the normal trainable mask from run_finetuning.py.
          Stage2: freeze wrapped base and patch/front, keep LoRA + norm/bias + head.
      - head_refit:
          Stage1: use the normal trainable mask.
          Stage2: freeze backbone/LoRA/base/front, keep head + norm/bias.
    """
    if getattr(args, "model_name", None) != "CBraMod":
        return False

    mode = str(getattr(args, "cbra_stage_mode", "none")).lower()
    if mode in ("", "none"):
        return False

    switch_epoch = int(getattr(args, "cbra_stage_epoch", -1))
    if switch_epoch <= 0 or int(epoch_id) <= switch_epoch:
        return False

    if mode not in ("two_stage_lora_norm", "head_refit"):
        return False

    last_stage = getattr(model, "_cbra_stage_last_stage", None)
    stage_name = f"{mode}_after_{switch_epoch}"
    if last_stage == stage_name:
        return True

    kept = 0
    frozen = 0
    kept_examples = []

    for name, param in model.named_parameters():
        trainable = False

        if _is_head_param_name(name):
            trainable = True
        elif _is_cbra_norm_bias_name(name):
            trainable = True
        elif mode == "two_stage_lora_norm" and _is_lora_param_name(name):
            trainable = True
        else:
            trainable = False

        param.requires_grad = trainable
        if trainable:
            kept += param.numel()
            if len(kept_examples) < 80:
                kept_examples.append(name)
        else:
            frozen += param.numel()

    model._cbra_stage_last_stage = stage_name
    print(
        f"[CBraStage] epoch {epoch_id}: mode={mode}, switch_epoch={switch_epoch}, "
        f"kept_trainable={kept}, frozen={frozen}"
    )
    for n in kept_examples:
        print(f"  [CBraStage-keep] {n}")
    if len(kept_examples) == 0:
        print("  [CBraStage-keep] none")
    return True


def _apply_staged_lora_controls(model: torch.nn.Module, args, epoch_id: int) -> bool:
    """Stage-wise LoRA lifecycle.

    Currently used for EEGPT bridge mismatch diagnosis:
      - eegpt_bridge_then_ffn:
          Stage 1 (epoch <= staged_lora_bridge_epochs): train bridge LoRA + head only.
          Stage 2: freeze bridge LoRA, train last2 FFN LoRA + head.
      - eegpt_bridge_only:
          Train only bridge LoRA + head for all epochs. This tests whether a stronger
          input-side LoRA alone can align EEGPT before FFN starts absorbing task bias.

    This keeps the method inside the LoRA framework while preventing FFN LoRA
    from immediately absorbing the task bias before the input bridge has aligned.
    """
    if _apply_cbra_staged_controls(model, args, epoch_id):
        return True

    mode = str(getattr(args, "staged_lora_mode", "none")).lower()
    if mode in ("", "none"):
        return False
    if mode not in ("eegpt_bridge_then_ffn", "eegpt_bridge_only"):
        return False

    bridge_epochs = int(getattr(args, "staged_lora_bridge_epochs", 5))
    if mode == "eegpt_bridge_only":
        stage = "bridge"
    else:
        stage = "bridge" if int(epoch_id) <= bridge_epochs else "ffn"

    # Optional high-LR bridge-first behavior without rebuilding optimizer groups.
    # During bridge stage, bridge LoRA gradients are multiplied; afterwards they
    # are frozen by the staged trainable mask below.
    bridge_mult = float(getattr(args, "staged_lora_bridge_grad_mult", 1.0))
    if bridge_mult != 1.0:
        holder = _ensure_staged_bridge_grad_hooks(model)
        holder["bridge_scale"] = bridge_mult if stage == "bridge" else 1.0
        if stage == "bridge":
            print(f"[Stage-LoRA] bridge gradient multiplier = {bridge_mult:g}")

    last_stage = getattr(model, "_staged_lora_last_stage", None)
    if last_stage == stage:
        return True

    kept = 0
    frozen = 0
    kept_examples = []
    for name, param in model.named_parameters():
        trainable = False
        if _is_head_param_name(name):
            trainable = bool(getattr(args, "lora_train_head", True))
        elif stage == "bridge" and _is_bridge_lora_param_name(name):
            trainable = True
        elif stage == "ffn" and _is_ffn_lora_param_name(name):
            trainable = True
        else:
            trainable = False

        param.requires_grad = trainable
        if trainable:
            kept += param.numel()
            if len(kept_examples) < 40:
                kept_examples.append(name)
        else:
            frozen += param.numel()

    model._staged_lora_last_stage = stage
    print(
        f"[Stage-LoRA] epoch {epoch_id}: mode={mode}, stage={stage}, "
        f"bridge_epochs={bridge_epochs}, kept_trainable={kept}, frozen={frozen}"
    )
    for n in kept_examples:
        print(f"  [stage-keep] {n}")
    if not kept_examples:
        print("  [stage-keep] none")
    return True


def apply_lora_lifecycle_controls(model: torch.nn.Module, args, epoch_id: int):
    """Minimal LoRA lifecycle control for few-shot training.

    Controls implemented here:
      1) freeze_non_lora_after_epoch:
         after epoch K, freeze non-LoRA backbone parameters while keeping LoRA/head
         and explicitly selected small modules trainable.
      2) lora_head_grad_decay_after_epoch:
         after epoch K, scale LoRA/head gradients by a factor, e.g. 0.1.

    This implements the training-side idea tested after BIOT guard-v0:
    controlling only non-LoRA drift was insufficient, so LoRA/head also need a
    lifecycle slow-down rather than continuing to write at full strength.
    """
    epoch_id = int(epoch_id)

    # Stage-wise LoRA control first. For staged modes this deliberately overrides
    # trainable flags before each epoch, while leaving optimizer groups intact.
    _apply_staged_lora_controls(model, args, epoch_id)

    # Gradient lifecycle for LoRA/head.
    holder = None
    decay_after = int(getattr(args, "lora_head_grad_decay_after_epoch", -1))
    decay_factor = float(getattr(args, "lora_head_grad_decay_factor", 1.0))
    if decay_after > 0:
        holder = _ensure_lifecycle_grad_hooks(model)
        scale = decay_factor if epoch_id > decay_after else 1.0
        holder["lora_scale"] = scale
        holder["head_scale"] = scale
        if scale != 1.0:
            print(f"[LoRA-Life] epoch {epoch_id}: LoRA/head gradient scale = {scale:g}")

    # Freeze non-LoRA backbone after a fixed epoch.
    freeze_after = int(getattr(args, "freeze_non_lora_after_epoch", -1))
    if freeze_after <= 0 or epoch_id <= freeze_after:
        return False

    if bool(getattr(model, "_lora_lifecycle_frozen", False)):
        return True

    keep_lora = True
    keep_head = bool(getattr(args, "lora_train_head", True))
    keep_chan = bool(getattr(args, "lora_train_chan_conv", False))
    keep_cbra_front = bool(getattr(args, "cbra_train_patch_embed_when_frozen", False))

    kept = 0
    frozen = 0
    kept_examples = []

    for name, param in model.named_parameters():
        trainable = False
        if keep_lora and _is_lora_param_name(name):
            trainable = True
        elif keep_head and _is_head_param_name(name):
            trainable = True
        elif keep_chan and "chan_conv" in name:
            trainable = True
        elif keep_cbra_front and "main_model.patch_embedding" in name:
            trainable = True

        param.requires_grad = trainable
        if trainable:
            kept += param.numel()
            if len(kept_examples) < 30:
                kept_examples.append(name)
        else:
            frozen += param.numel()

    model._lora_lifecycle_frozen = True
    print(
        f"[LoRA-Life] epoch {epoch_id}: freeze_non_lora_after_epoch={freeze_after}; "
        f"kept_trainable={kept}, frozen={frozen}"
    )
    for n in kept_examples:
        print(f"  [keep] {n}")
    if not kept_examples:
        print("  [keep] none")
    return True


def compute_l2sp_control_loss(model, args):
    """
    L2-SP regularization toward initialization for selected trainable params.

    The references are registered in run_finetuning.py as model._l2sp_ref.
    We average mean-squared drift across tracked tensors, so the lambda scale is
    not dominated by tensor size.
    """
    lam = float(getattr(args, "cbra_l2sp_lambda", 0.0))
    refs = getattr(model, "_l2sp_ref", None)
    if lam <= 0.0 or not refs:
        device = next(model.parameters()).device
        return torch.zeros((), device=device), 0.0, 0

    reg = None
    terms = 0
    for name, param in model.named_parameters():
        ref = refs.get(name, None)
        if ref is None:
            continue
        if not param.requires_grad:
            continue
        ref = ref.to(device=param.device, dtype=param.dtype, non_blocking=True)
        term = torch.mean((param - ref) ** 2)
        reg = term if reg is None else reg + term
        terms += 1

    if terms <= 0 or reg is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device), 0.0, 0

    raw = reg / float(terms)
    loss = lam * raw
    return loss, float(raw.detach().cpu().item()), terms


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)

    return loss, outputs


def _module_e_dynamic_pressure_controller(model: torch.nn.Module):
    controller = getattr(model, "_module_e_dynamic_pressure_controller", None)
    if controller is not None:
        return controller
    wrapped = getattr(model, "module", None)
    if wrapped is not None:
        return getattr(wrapped, "_module_e_dynamic_pressure_controller", None)
    return None


def train_one_epoch(args, model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, ch_names=None):
    is_binary = (args.nb_classes == 1)
    
    # loss foundation
    if args.task_mod == 'Regression':
        criterion = torch.nn.MSELoss()
    else:
        criterion = _build_classification_criterion(args, device, is_binary=is_binary, epoch_id=epoch + 1)
    
    model.train(True)
    if args.finetune_mod == 'linear':
        model.main_model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    update_freq = args.update_freq
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if args.norm_method == 'mv':
            samples = samples.float().to(device, non_blocking=True) * args.mv_norm_value
        else:
            samples = samples.float().to(device, non_blocking=True)
        
        targets = targets.to(device, non_blocking=True)
        if is_binary:
            targets = targets.float().unsqueeze(-1)
        else:
            targets = targets.int().long()

        # with torch.cuda.amp.autocast():
        loss, output = train_class_batch(
            model, samples, targets, criterion)

        lora_delta_loss, lora_delta_raw, lora_delta_terms = compute_lora_delta_control_loss(model, args)
        if lora_delta_terms > 0:
            loss = loss + lora_delta_loss

        l2sp_loss, l2sp_raw, l2sp_terms = compute_l2sp_control_loss(model, args)
        if l2sp_terms > 0:
            loss = loss + l2sp_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Warning: Loss is {}".format(loss_value))

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        
        if (data_iter_step + 1) % update_freq == 0:
            module_e_controller = _module_e_dynamic_pressure_controller(model)
            if module_e_controller is not None:
                module_e_controller.finish_step(global_step=it, epoch=epoch + 1)
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if args.task_mod == 'Classification':
            if is_binary:
                class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
            else:
                class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()
            metric_logger.update(class_acc=class_acc)
        
        metric_logger.update(loss=loss_value)
        if lora_delta_terms > 0:
            metric_logger.update(lora_delta_raw=lora_delta_raw)
            metric_logger.update(lora_delta_loss=float(lora_delta_loss.detach().cpu().item()))
        if l2sp_terms > 0:
            metric_logger.update(l2sp_raw=l2sp_raw)
            metric_logger.update(l2sp_loss=float(l2sp_loss.detach().cpu().item()))
        # metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        # metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if args.task_mod == 'Classification':
                log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(args, data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'],
             return_details=False, logit_bias=None):
    """
    增强版 evaluate：
    1. 保留原本全局 metrics 计算逻辑；
    2. 支持 return_details=True，用于返回 per-class recall / confusion matrix / worst-class recall；
    3. 默认 return_details=False，因此不破坏原本代码调用方式。
    """
    is_binary = (args.nb_classes == 1)

    if args.task_mod == 'Regression':
        criterion = torch.nn.MSELoss()
    elif is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()

    pred_list = []
    true_list = []

    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[-1]

        if args.norm_method == 'mv':
            EEG = EEG.float().to(device, non_blocking=True) * args.mv_norm_value
        else:
            EEG = EEG.float().to(device, non_blocking=True)

        target = target.to(device, non_blocking=True)

        if args.task_mod == 'Regression':
            target_for_loss = target.float()
        elif is_binary:
            target_for_loss = target.float().unsqueeze(-1)
        else:
            target_for_loss = target.int().long()

        output = model(EEG)
        loss = criterion(output, target_for_loss)

        metric_output = output
        if (
            args.task_mod == 'Classification'
            and (not is_binary)
            and logit_bias is not None
        ):
            metric_output = output + logit_bias.to(device=output.device, dtype=output.dtype).view(1, -1)

        if args.task_mod == 'Regression':
            output_for_metric = output.detach().cpu()
            target_for_metric = target_for_loss.detach().cpu()
        elif is_binary:
            output_for_metric = torch.sigmoid(output).detach().cpu()
            target_for_metric = target_for_loss.detach().cpu()
        else:
            output_for_metric = metric_output.detach().cpu()
            target_for_metric = target_for_loss.detach().cpu()

        pred_list.append(output_for_metric)
        true_list.append(target_for_metric)
        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()

    pred = torch.cat(pred_list, dim=0).numpy()
    true = torch.cat(true_list, dim=0).numpy()

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg

    details = {}
    if return_details and args.task_mod == 'Classification':
        details = _build_classification_details(
            pred=pred,
            true=true,
            nb_classes=args.nb_classes,
            is_binary=is_binary
        )

        ret['worst_class_recall'] = details['worst_class_recall']
        ret['recall_std'] = details['recall_std']
        ret['pred_entropy'] = details['pred_entropy']

    metric_text = "  ".join([
        f"{k}: {v:.4f}" if isinstance(v, (float, int)) and not math.isnan(v) else f"{k}: {v}"
        for k, v in ret.items()
    ])
    print(f"* Global {header} {metric_text}")

    if return_details:
        return ret, details
    return ret


def _build_classification_details(pred, true, nb_classes, is_binary):
    """
    从全数据集预测结果里计算：
    - confusion matrix
    - per-class recall
    - worst-class recall
    - recall std
    - prediction distribution entropy

    注意：
    AdaBrain-Bench 里二分类时 args.nb_classes 会被改成 1，
    但真实标签仍然是 0/1，所以这里二分类的类别数手动设成 2。
    """
    true = np.asarray(true).reshape(-1)

    if is_binary:
        score = np.asarray(pred).reshape(-1)
        pred_label = (score >= 0.5).astype(np.int64)
        n_classes = 2
    else:
        score = np.asarray(pred)
        pred_label = np.argmax(score, axis=1).astype(np.int64)
        n_classes = int(nb_classes)

    true_label = true.astype(np.int64)

    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true_label, pred_label):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            confusion[t, p] += 1

    per_class_recall = []
    for c in range(n_classes):
        denom = confusion[c, :].sum()
        if denom == 0:
            per_class_recall.append(np.nan)
        else:
            per_class_recall.append(confusion[c, c] / denom)

    per_class_recall = np.asarray(per_class_recall, dtype=np.float64)

    valid_recall = per_class_recall[~np.isnan(per_class_recall)]
    if len(valid_recall) == 0:
        worst_class_recall = float('nan')
        recall_std = float('nan')
    else:
        worst_class_recall = float(np.min(valid_recall))
        recall_std = float(np.std(valid_recall))

    pred_count = np.bincount(pred_label, minlength=n_classes).astype(np.float64)
    pred_prob = pred_count / max(pred_count.sum(), 1.0)
    pred_prob_nonzero = pred_prob[pred_prob > 0]
    pred_entropy = float(-(pred_prob_nonzero * np.log(pred_prob_nonzero + 1e-12)).sum())

    return {
        'confusion_matrix': confusion,
        'per_class_recall': per_class_recall,
        'worst_class_recall': worst_class_recall,
        'recall_std': recall_std,
        'pred_entropy': pred_entropy,
        'y_true': true_label,
        'y_pred': pred_label,
        'scores': score,
    }


def build_logit_adjust_bias(val_details, nb_classes, strength=1.0, clip=2.0, eps=1e-6):
    """Build a validation-based logit bias for lightweight decision calibration.

    The bias nudges the predicted class prior on validation toward the true
    validation class prior. It is intended as a diagnostic/lightweight
    calibration step for class-imbalanced few-shot experiments, not as a
    replacement for training.
    """
    if val_details is None:
        return None
    y_true = val_details.get('y_true', None)
    y_pred = val_details.get('y_pred', None)
    if y_true is None or y_pred is None:
        return None

    n_classes = int(nb_classes)
    if n_classes <= 1:
        return None

    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    true_counts = np.bincount(y_true[(y_true >= 0) & (y_true < n_classes)], minlength=n_classes).astype(np.float64)
    pred_counts = np.bincount(y_pred[(y_pred >= 0) & (y_pred < n_classes)], minlength=n_classes).astype(np.float64)

    true_prior = (true_counts + float(eps)) / (true_counts.sum() + float(eps) * n_classes)
    pred_prior = (pred_counts + float(eps)) / (pred_counts.sum() + float(eps) * n_classes)

    bias = np.log(true_prior) - np.log(pred_prior)
    bias = bias - bias.mean()
    if clip is not None and float(clip) > 0:
        bias = np.clip(bias, -float(clip), float(clip))
    bias = bias * float(strength)

    print(f"[LogitAdjust] true_counts={true_counts.astype(int).tolist()} pred_counts={pred_counts.astype(int).tolist()}")
    print(f"[LogitAdjust] bias={bias.tolist()} strength={strength} clip={clip}")
    return torch.tensor(bias, dtype=torch.float32)

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def get_model_output(model, samples, ch_names):
    outputs = model(samples)
    
    return outputs

def train_model(args, eeg_model, dataloader, optimizer, device, 
                img_features_all, config, loss_scaler, start_steps=None, 
                lr_schedule_values=None, wd_schedule_values=None, ch_names=None,
                num_training_steps_per_epoch=None, model_ema: Optional[ModelEma] = None):
    
    eeg_model.train()
    if args.finetune_mod == 'linear':
        eeg_model.main_model.eval()
    
    img_features_all = (img_features_all[::10]).to(device).float()

    if loss_scaler is None:
        eeg_model.zero_grad()
        eeg_model.micro_steps = 0
    else:
        optimizer.zero_grad()
    
    total_loss = 0
    correct = 0
    total = 0
    alpha=0.99
    features_list = []  # List to store features
    save_features= True
    for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):
        step = batch_idx
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        optimizer.zero_grad()
        eeg_data = eeg_data.float().to(device, non_blocking=True)
        
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        if loss_scaler is None:
            eeg_features = eeg_features.half()
            eeg_features = eeg_model(eeg_data).float()
        else:
            # with torch.cuda.amp.autocast():
            eeg_features = eeg_model(eeg_data).float()
        
        features_list.append(eeg_features)

        loss_scale = eeg_model.module.loss_scale if args.distributed else eeg_model.loss_scale
        img_loss = eeg_model.loss_func(eeg_features, img_features, loss_scale)
        loss = img_loss
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}.".format(loss_value))

        max_norm = args.clip_grad
        if loss_scaler is None:
            eeg_model.backward(loss)
            eeg_model.step()
            if model_ema is not None:
                model_ema.update(eeg_model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(eeg_model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=eeg_model.parameters(), create_graph=is_second_order,
                                    update_grad=True)
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(eeg_model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        total_loss += loss.item()
        logit_scale = loss_scale
        # Compute the corresponding logits
        logits_img = logit_scale * eeg_features @ img_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(logits_single, dim=1) # (n_batch, ) in {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features
    
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total

    return average_loss, accuracy, torch.cat(features_list, dim=0)

def evaluate_model(args, eeg_model, dataloader, device, img_features_all, k, config, ch_names):
    eeg_model.eval()

    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    # Get all unique classes
    all_labels = set(range(img_features_all.size(0)))
    top5_acc = 0
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, img_features) in enumerate(dataloader):
            eeg_data = eeg_data.float().to(device, non_blocking=True)
            
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = eeg_data.size(0)
            eeg_features = eeg_model(eeg_data)
        
            logit_scale = eeg_model.loss_scale
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            loss = img_loss
            
            total_loss += loss.item()
            
            for idx, label in enumerate(labels):
                # First, select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                if k==200:
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]

                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    
                    predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest =True)
                                                   
                    # Check if the true label is in the top-5 predictions
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
                        top5_correct_count+=1                                
                    total += 1
                elif k==2 or k==4 or k==10:
                    selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                    # Compute the corresponding logits
                    logits_img = logit_scale * eeg_features[idx] @ selected_img_features.T
                    logits_single = logits_img
                    # Get the predicted class
                    predicted_label = selected_classes[torch.argmax(logits_single).item()] # (n_batch, ) in {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features
    
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total

    return average_loss, accuracy, top5_acc

def main_train_loop(args, current_time, eeg_model, 
                    train_dataloader, test_dataloader, optimizer, 
                    device, img_features_train_all, img_features_test_all, 
                    config, loss_scaler, logger=None, lr_schedule_values=None, ch_names=None,
                    wd_schedule_values=None, num_training_steps_per_epoch=None, model_ema=None):
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger) 
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    for epoch in range(config.epochs):
        # Train the model
        start_steps=epoch * num_training_steps_per_epoch
        train_loss, train_accuracy, features_tensor = train_model(
            args, eeg_model, train_dataloader, optimizer, device, 
            img_features_train_all, config=config, loss_scaler=loss_scaler, 
            start_steps=start_steps, lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values, 
            ch_names=ch_names, num_training_steps_per_epoch=num_training_steps_per_epoch, model_ema=model_ema)
        if (epoch + 1) % args.save_ckpt_freq == 0:
        # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')
            save_dir = os.path.join(args.output_dir, 'saved_models')
            save_dir = f"{save_dir}/contrast/across/{config.model_name}_{current_time}"
            os.makedirs(save_dir, exist_ok=True)             
            file_path = f"{save_dir}/{epoch+1}.pth"
            torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=200, config=config, ch_names=ch_names)
        _, v2_acc, _ = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=2, config=config, ch_names=ch_names)
        _, v4_acc, _ = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=4, config=config, ch_names=ch_names)
        _, v10_acc, _ = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=10, config=config, ch_names=ch_names)
        _, v50_acc, v50_top5_acc = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=50, config=config, ch_names=ch_names)
        _, v100_acc, v100_top5_acc = evaluate_model(args, eeg_model, test_dataloader, device, img_features_test_all, k=100, config=config, ch_names=ch_names)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)
        
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "v2_acc": v2_acc,
        "v4_acc": v4_acc,
        "v10_acc": v10_acc,
        "top5_acc":top5_acc,
        "v50_acc": v50_acc,
        "v100_acc": v100_acc,
        "v50_top5_acc":v50_top5_acc,
        "v100_top5_acc": v100_top5_acc
        }

        results.append(epoch_results)
        # If the test accuracy of the current epoch is the best, save the model and related information
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # best_model_weights = model.state_dict().copy()
            
            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc":v2_acc,
                "v4_acc":v4_acc,
                "v10_acc":v10_acc
            }
        logger.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_accuracy,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "v2 Accuracy": v2_acc,
            "v4 Accuracy": v4_acc,
            "v10 Accuracy": v10_acc,
            "Epoch": epoch
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
        print(f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy:{v2_acc} - v4 Accuracy:{v4_acc} - v10 Accuracy:{v10_acc} - v50 Accuracy:{v50_acc} - v100 Accuracy:{v100_acc}")
  
    # # Load the best model weights
    # model.load_state_dict(best_model_weights)

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss curve
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(test_losses, label='Test Loss')
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy curve
    axs[0, 1].plot(train_accuracies, label='Train Accuracy')
    axs[0, 1].plot(test_accuracies, label='Test Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you've already calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label='2-class Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label='4-class Accuracy')
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label='10-class Accuracy')
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information for annotation
    info_text = (f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
                f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
                f"Train Accuracy: {best_epoch_info['train_accuracy']:.4f}\n"
                f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
                f"Test Accuracy: {best_epoch_info['test_accuracy']:.4f}\n"
                f"v2_acc:{best_epoch_info['v2_acc']:.4f}\n"
                f"v4_acc:{best_epoch_info['v4_acc']:.4f}\n"
                f"v10_acc:{best_epoch_info['v10_acc']:.4f}")

    axs[2, 1].axis('off')  
    axs[2, 1].text(0.5, 0.5, info_text, fontsize=10, ha='center', va='center', transform=axs[2, 1].transAxes)

    plt.tight_layout()

    # Add main title
    plt.suptitle('pos_img_text', fontsize=16, y=1.05)
    plt.savefig('pos_img_text')
    logger.finish()

    print(info_text)

    return results
