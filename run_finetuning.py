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

import argparse
import copy
import datetime
from pyexpat import model
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os
import re
import random
import models.modeling_finetune
from pathlib import Path
from collections import OrderedDict, Counter
from timm.models import create_model
from util.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from util.utils import NativeScalerWithGradNormCount as NativeScaler
import util.utils as utils
from util.eegdatasets import EEGDataset
from util.dataset_config import load_task_dataset_info
from util.lora import (
    freeze_all_parameters,
    unfreeze_module,
    apply_lora_to_eegfm,
    mark_lora_and_selected_modules_trainable,
    print_trainable_parameters,
    set_lora_runtime_scale,
)
from engine_for_finetuning import (
    train_one_epoch,
    evaluate,
    main_train_loop,
    build_logit_adjust_bias,
    apply_lora_lifecycle_controls,
    _build_classification_criterion,
)
from module_a_lifecycle import (
    add_lifecycle_window_args,
    capture_lifecycle_snapshot,
    run_lifecycle_window_search,
)

# Functional-block profiling framework (FB2). Kept outside model core and LoRA implementation.
_fb_import_error = None
try:
    from util.fb_policy import add_fb_args, resolve_functional_args, write_resolved_recipe
    from util.fb_probe import save_split_integrity, save_block_registry, save_block_delta_summary
    from util.fb_collect import collect_outputs_if_requested
    from util.module_e_structural_routing import (
        attach_module_e_dynamic_pressure_controller,
        is_module_e_target,
        module_e_branch_from_lora_param_name,
        module_e_mode_from_args,
        save_module_e_coverage_audit,
        save_module_e_lora_injection_audit,
        save_module_e_structural_pressure_proxy,
    )
    from util.module_d_semantic_refinement import module_d_eval_row_from_details, save_module_d_sbr_eval
    from util.module_c_preflight_policy import (
        _resolve_module_c_support_batch_limit,
        capture_module_c_rng_state,
        module_c_preflight_requested,
        restore_module_c_rng_state,
        run_module_c_preflight_selection,
    )
    from util.fb_runtime_hooks import run_signal_alignment_probe_after_training
except Exception as _caught_fb_import_error:
    _fb_import_error = _caught_fb_import_error
    def add_fb_args(parser): return parser
    def resolve_functional_args(args): return args
    def write_resolved_recipe(args, output_dir): return None
    def save_split_integrity(*args, **kwargs): return None
    def save_block_registry(*args, **kwargs): return None
    def save_block_delta_summary(*args, **kwargs): return None
    def collect_outputs_if_requested(*args, **kwargs): return None
    def save_module_e_coverage_audit(*args, **kwargs): return None
    def save_module_e_lora_injection_audit(*args, **kwargs): return None
    def save_module_e_structural_pressure_proxy(*args, **kwargs): return None
    def attach_module_e_dynamic_pressure_controller(*args, **kwargs): return None
    def is_module_e_target(*args, **kwargs): return False
    def module_e_branch_from_lora_param_name(*args, **kwargs): return None
    def module_e_mode_from_args(*args, **kwargs): return "dynamic_pressure_gate"
    def module_d_eval_row_from_details(*args, **kwargs): return {}
    def save_module_d_sbr_eval(*args, **kwargs): return None
    def module_c_preflight_requested(*args, **kwargs): return False
    def _resolve_module_c_support_batch_limit(*args, **kwargs): return 0
    def capture_module_c_rng_state(*args, **kwargs): return None
    def restore_module_c_rng_state(*args, **kwargs): return None
    def run_module_c_preflight_selection(*args, **kwargs): return None
    def run_signal_alignment_probe_after_training(*args, **kwargs): return False
    print(f"[FB2][WARN] functional-block framework unavailable: {_fb_import_error}")
import csv
from functools import partial
from models.cbramod import CBraMod
from models.EEGPT_mcae import EEGTransformer, Conv1dWithConstraint, LinearWithConstraint
from models.biot import BIOTClassifier
from models.csbrain import CSBrain
try:
    from models.gram_ada import GramAdaBackbone
except Exception:
    GramAdaBackbone = None
from models.EEGNet import EEGNet
from models.LMDA import LMDA
from models.EEGConformer import Conformer
from models.EEGTransformer import STTransformer
from models.loss import ClipLoss

from torch.utils.data import random_split, ConcatDataset

# -------------------------------The pre-trained weights of the foundation model---------------------------------------
finetune_list = {
    'LaBraM': './checkpoints/labram-base.pth',
    'CBraMod': './checkpoints/pretrained_weights.pth',
    'EEGPT': './checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt',
    'BIOT': "./checkpoints/EEG-six-datasets-18-channels.ckpt",
    'CSBrain': './checkpoints/CSBrain.pth',
    'Gram': './checkpoints/gram_base.pth',
}
# ---------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------Parameters------------------------------------------------------
def get_args(argv=None):
    parser = argparse.ArgumentParser()
    # Fine-tuning parameters
    parser.add_argument('--dataset', default='SEED-IV', type=str,
                        choices=['SEED', 'SEED-IV', 'BCI-IV-2A', 'SHU', 'SEED-VIG', 'EEGMAT',
                                 'Sleep-EDF', 'HMC', 'SHHS', 'TUAB', 'TUEV', 'Things-EEG', 'Siena'],
                        help=('Dataset name. Classification/regression require an entry in '
                              'dataset_config; some choices are retrieval or preprocessing-only.'))
    parser.add_argument('--model_name', default='LaBraM', type=str,
                        choices=['LaBraM', 'CBraMod', 'EEGPT', 'BIOT', 'CSBrain', 'Gram', 'EEGNet', 'LMDA', 'EEGConformer', 'ST-Transformer'])
    parser.add_argument('--csbrain_ckpt', default='./checkpoints/CSBrain.pth', type=str,
                        help='CSBrain pretrained checkpoint path.')
    parser.add_argument('--gram_ckpt', default='./checkpoints/gram_base.pth', type=str,
                        help='Gram pretrained checkpoint path. The official Gram repo should be under external/Gram or GRAM_ROOT.')
    parser.add_argument('--gram_vqgan_ckpt', default='./checkpoints/base_class_quantization.pth', type=str,
                        help='Gram base-class quantization / VQGAN checkpoint path required by official Gram fine-tune model.')
    parser.add_argument('--gram_root', default='external/Gram', type=str,
                        help='Path to official Gram repository root. Expected file: external/Gram/model/modeling_Gram_finetune.py')
    parser.add_argument('--gram_allow_scratch', action='store_true', default=False,
                        help='Allow Gram to run without gram_base.pth. Use only for interface debugging, not official FM baseline.')
    parser.add_argument('--csbrain_allow_scratch', action='store_true', default=False,
                        help='Allow CSBrain to run without CSBrain.pth. Use only for interface debugging, not official FM baseline.')
    parser.add_argument('--task_mod', default="Classification", type=str, choices=['Classification', 'Regression', 'Retrieval'],
                        help='type of task')
    parser.add_argument('--subject_mod', default="multi", type=str, choices=['multi', 'cross', 'fewshot', 'single'],
                        help='evaluation settings including cross-subject, multi-subject and few-shot settings (single-subject setting for retrieval task)')
    parser.add_argument('--finetune_mod', default='full', type=str, choices=['full', 'linear', 'lora', 'all'], help='model finetune mod')
    parser.add_argument('--disable_pretrained_loading', action='store_true', default=False,
                        help='Use the same model/adaptation protocol without loading EEGFM pretrained weights. This is for random/scratch control baselines.')
    parser.add_argument('--nb_classes', default=0, type=int, help='number of classes in the datasets (classification task)')
    parser.add_argument('--batch_size', default=64, type=int, help='training batch size')
    parser.add_argument('--epochs', default=50, type=int, help='epoches for training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--norm_method', default='z_score', type=str, choices=['z_score', '0.1mv', '95'],
                        help='normalization methods including z-score, 95-percentile, and unit rescale (0.1mv)')
    
    parser.add_argument('--max_subject', default=8, type=int, help='number of subjects used for spliting validation set')
    parser.add_argument('--sampling_rate', default=200, type=int, choices=[200, 256], help='BIOT, LaBraM and CBraMod is 200Hz; EEGPT is 256Hz')
    parser.add_argument('--k_shot', default=10, type=float, help='number of shots in the few_shot setting')
    parser.add_argument('--run_tag', default='', type=str,
                        help='Optional short suffix appended to output_dir to avoid auto-resume/output collisions.')
    parser.add_argument('--output_dir', default='', type=str,
                        help=('Optional base result directory. The generated run tag is created '
                              'under this directory; empty uses finetuning_results/<task>/<model>/<mode>.'))
    parser.add_argument('--short_output_tag_only', action='store_true', default=False,
                        help='Use only run_tag as output folder name under the model result root. Keeps Windows paths short.')
    

    parser.add_argument('--logger', type=bool, default=False, help='enable WandB logging for retrieval')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--save_ckpt', action='store_true', default=True)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--loader_prefetch_factor', default=2, type=int,
                        help='DataLoader prefetch_factor when num_workers > 0. Lower values reduce Windows shared-memory pressure.')
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='pin CPU memory in dataLoader for more efficient (sometimes) transfer to GPU')
    parser.add_argument('--mv_norm_value', default=0.01, type=float,
                        help='scale_value when using 0.1mv norm_method, default is 0.01.')
    parser.add_argument('--subject_id', type=int, default=1, help='subject id for single subject retrieval task')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--update_freq', default=1, type=int)

    # Loss / class-imbalance parameters
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['ce', 'balanced_ce', 'sqrt_balanced_ce',
                                 'soft_sqrt_balanced_ce', 'anneal_sqrt_balanced_ce'],
                        help=(
                            'Training loss for classification. ce = original CE; '
                            'balanced_ce = inverse-frequency class-weighted CE; '
                            'sqrt_balanced_ce = softer sqrt inverse-frequency weighted CE; '
                            'soft_sqrt_balanced_ce = CE mixed with lambda * sqrt-balanced CE; '
                            'anneal_sqrt_balanced_ce = soft sqrt-balanced CE with lambda annealed to 0.'
                        ))
    parser.add_argument('--class_weight_clip_max', default=5.0, type=float,
                        help='Clip maximum class weight for balanced losses to avoid over-correcting rare classes.')
    parser.add_argument('--class_weight_clip_min', default=0.2, type=float,
                        help='Clip minimum class weight for balanced losses.')
    parser.add_argument('--class_balance_lambda', default=1.0, type=float,
                        help='Lambda for soft/annealed class-aware CE. 0=plain CE, 1=full sqrt-balanced CE.')
    parser.add_argument('--class_balance_anneal_start_epoch', default=3, type=int,
                        help='For anneal_sqrt_balanced_ce: keep full lambda through this epoch.')
    parser.add_argument('--class_balance_anneal_end_epoch', default=8, type=int,
                        help='For anneal_sqrt_balanced_ce: lambda reaches floor at/after this epoch.')
    parser.add_argument('--class_balance_anneal_floor', default=0.0, type=float,
                        help='For anneal_sqrt_balanced_ce: final lambda after annealing. Use 0.25 to keep weak class-aware pressure.')

    # LR schedule / stabilization parameters
    parser.add_argument('--lr_schedule_type', default='cosine', type=str, choices=['cosine', 'constant', 'plateau'],
                        help='cosine = original schedule; constant = fixed lr; plateau = ReduceLROnPlateau-style epoch LR reduction.')
    parser.add_argument('--plateau_metric', default='balanced_accuracy', type=str,
                        help='Validation metric used by plateau LR. Usually balanced_accuracy for TUEV.')
    parser.add_argument('--plateau_patience', default=2, type=int,
                        help='Number of non-improving eval epochs before reducing LR.')
    parser.add_argument('--plateau_factor', default=0.5, type=float,
                        help='LR multiplier when plateau is triggered.')
    parser.add_argument('--plateau_min_delta', default=1e-4, type=float,
                        help='Minimum metric improvement required to reset plateau patience.')
    parser.add_argument('--plateau_min_lr', default=1e-6, type=float,
                        help='Lower bound for plateau LR.')
    parser.add_argument('--plateau_warmup_epochs', default=0, type=int,
                        help='Do not reduce LR before this many epochs are finished.')

    # Validation-based logit calibration. This is only used at evaluation time.
    parser.add_argument('--eval_logit_adjust', action='store_true', default=False,
                        help='Use validation predictions to build a class-prior logit bias and evaluate the test set with it.')
    parser.add_argument('--logit_adjust_strength', default=1.0, type=float,
                        help='Strength multiplier for validation-based logit adjustment.')
    parser.add_argument('--logit_adjust_clip', default=2.0, type=float,
                        help='Clip absolute logit adjustment bias to this value.')

    # Prototype / rapid-calibration diagnostic.
    # This keeps the normal model logits, but additionally evaluates a lightweight
    # prototype readout in logit space. It is intended for EEGPT few-shot diagnosis.
    parser.add_argument('--proto_eval', action='store_true', default=False,
                        help='Evaluate logit-space prototype fusion using support logits from train_eval or val.')
    parser.add_argument('--proto_source', default='train_eval', type=str, choices=['train_eval', 'val'],
                        help='Support set used to build prototypes for --proto_eval.')
    parser.add_argument('--proto_alpha', default=2.0, type=float,
                        help='Fusion strength: fused_logits = model_logits + alpha * prototype_similarity.')
    parser.add_argument('--proto_metric', default='cosine', type=str, choices=['cosine', 'neg_l2'],
                        help='Prototype similarity metric in logit space.')

    # CBraMod mechanism-diagnosis controls
    parser.add_argument('--cbra_train_patch_embed_when_frozen', action='store_true', default=False,
                        help='For CBraMod freeze-LoRA: additionally train main_model.patch_embedding parameters.')
    parser.add_argument('--cbra_freeze_patch_embed_in_full', action='store_true', default=False,
                        help='For CBraMod full-style LoRA: freeze main_model.patch_embedding parameters.')

    parser.add_argument('--cbra_train_norm_bias', action='store_true', default=False,
                        help='For CBraMod LoRA: additionally train LayerNorm/norm parameters and bias terms. Used to test whether Full FT gains come from internal scale/bias adaptation.')
    parser.add_argument('--cbra_train_norm_only', action='store_true', default=False,
                        help='For CBraMod LoRA: train norm parameters but not generic bias terms. Has effect only with --cbra_train_norm_bias.')
    parser.add_argument('--cbra_train_wrapped_base', action='store_true', default=False,
                        help='For CBraMod LoRA: additionally train the frozen base weights inside LoRA-wrapped modules. This tests targeted partial adaptation of the modules selected by lora_target.')
    parser.add_argument('--cbra_train_wrapped_base_last_n', default=-1, type=int,
                        help='If >0 with --cbra_train_wrapped_base, only unfreeze LoRA wrapped base params whose encoder layer index is in the last N layers.')

    parser.add_argument('--cbra_l2sp_lambda', default=0.0, type=float,
                        help='CBraMod only: L2-SP regularization coefficient for selected trainable base/front/norm params against their initialization. 0 disables it.')
    parser.add_argument('--cbra_l2sp_scope', default='wrapped_base_patch_norm', type=str,
                        choices=['wrapped_base', 'wrapped_base_patch', 'wrapped_base_patch_norm', 'all_trainable_non_lora_head'],
                        help='Which trainable params to constrain with --cbra_l2sp_lambda.')

    parser.add_argument('--cbra_stage_mode', default='none', type=str,
                        choices=['none', 'two_stage_lora_norm', 'head_refit'],
                        help='CBraMod only: stage-wise trainable mask. two_stage_lora_norm keeps LoRA+norm+bias+head after cbra_stage_epoch; head_refit keeps only head+norm/bias.')
    parser.add_argument('--cbra_stage_epoch', default=-1, type=int,
                        help='CBraMod only: after this epoch switch to the stage-2 trainable mask.')
    parser.add_argument('--cbra_grad_scale_wrapped_base', default=1.0, type=float,
                        help='CBraMod only: gradient multiplier for trainable LoRA-wrapped base parameters.')
    parser.add_argument('--cbra_grad_scale_patch', default=1.0, type=float,
                        help='CBraMod only: gradient multiplier for trainable patch/front/positional parameters.')
    parser.add_argument('--cbra_grad_scale_norm_bias', default=1.0, type=float,
                        help='CBraMod only: gradient multiplier for trainable norm/bias parameters.')

    # LoRA parameters
    parser.add_argument('--lora_target', default='qv', type=str,
                        choices=['none', 'qv', 'qkv', 'qkvo', 'module_c', 'ffn', 'mlp', 'sem', 'semantic', 'fb_sem',
                                 'sig', 'signal', 'signal_align', 'front_align',
                                 'str', 'struct', 'structural', 'struct_mix', 'mix',
                                 'ffn_late', 'ffn_tophalf', 'ffn_last2', 'ffn_last1',
                                 'qv_ffn', 'qkvo_ffn', 'attn_ffn', 'all_linear',
                                 'spatial_attn', 'temporal_attn', 'spatial_attn_ffn', 'temporal_attn_ffn', 'bridge', 'input_bridge', 'front', 'bridge_ffn', 'bridge_ffn_last2', 'bridge_last2ffn_pure', 'input_side', 'channel_adapter'],
                        help=(
                            'LoRA injection target. Main targets: ffn, ffn_late/ffn_last2, '
                            'signal_align for Module B input-front alignment, '
                            'temporal_attn/spatial_attn for structural routing, '
                            'bridge/input_bridge for EEGPT/BIOT input adapter.'
                            ' module_c automatically selects a nonempty B/D/E subset with matched one-pass support training and paired validation log-loss unless --module_c_selected is provided.'
                        ))
    parser.add_argument('--module_b_sites', default='both', type=str,
                        choices=['both', 'input', 'bridge'],
                        help=(
                            'Module B site switch for signal_align and module_c+B. '
                            'both = raw-input residual plus 1x1 channel-bridge LoRA (default); '
                            'input = raw-input residual only; bridge = existing channel-bridge LoRA only.'
                        ))
    parser.add_argument('--lora_base_update', default='freeze', type=str, choices=['freeze', 'full'],
                        help=(
                            'How to handle original backbone parameters when LoRA is injected. '
                            'freeze = standard Frozen-LoRA: freeze original W and train LoRA/head only. '
                            'full = Full FT + LoRA: keep original W trainable and also train LoRA. '
                            'Use this to separate the freeze variable from the LoRA variable.'
                        ))
    parser.add_argument('--lora_rank', default=4, type=int,
                        help='LoRA rank r.')
    parser.add_argument('--lora_alpha', default=8.0, type=float,
                        help='LoRA scaling alpha. Effective scaling = alpha / rank.')
    parser.add_argument('--lora_dropout', default=0.1, type=float,
                        help='Dropout used inside LoRA branch.')
    parser.add_argument('--module_e_mode', default='dynamic_pressure_gate', type=str,
                        choices=['dynamic_pressure_gate'],
                        help=(
                            'Module E execution mode. The formal method inserts structural LoRA and lets online '
                            'branch pressure gate the LoRA update strength.'
                        ))
    parser.add_argument('--module_e_pressure_beta', default=0.95, type=float,
                        help='EMA beta for dynamic Module E LoRA-B gradient pressure.')
    parser.add_argument('--module_e_gate_temperature', default=1.0, type=float,
                        help='Softmax temperature for dynamic Module E branch pressure gates.')
    parser.add_argument('--module_e_gate_floor', default=0.2, type=float,
                        help='Uniform-share floor for dynamic Module E gates; prevents hard branch collapse.')
    parser.add_argument('--module_e_scale_min', default=0.5, type=float,
                        help='Minimum branch LoRA gradient multiplier used by dynamic Module E.')
    parser.add_argument('--module_e_scale_max', default=1.5, type=float,
                        help='Maximum branch LoRA gradient multiplier used by dynamic Module E.')
    parser.add_argument('--module_e_warmup_steps', default=0, type=int,
                        help='Optimizer steps to collect Module E pressure while keeping branch LoRA scales at 1. Default 0 means immediate gating.')
    parser.add_argument('--module_e_diag_freq', default=1, type=int,
                        help='Write dynamic Module E pressure diagnostics every N optimizer steps.')
    parser.add_argument('--lora_train_head', action='store_true', default=True,
                        help='Train task_head together with LoRA parameters.')
    parser.add_argument('--lora_freeze_head', action='store_false', dest='lora_train_head',
                        help='Freeze task_head during LoRA fine-tuning.')
    parser.add_argument('--lora_train_chan_conv', action='store_true', default=False,
                        help='Also train EEGPT/BIOT channel adapter conv during LoRA fine-tuning.')
    parser.add_argument('--lora_grad_decay_after_epoch', default=-1, type=int,
                        help='If >0, multiply LoRA gradients by lora_grad_decay_factor after this epoch. Used as LoRA LR decay without rebuilding optimizer groups.')
    parser.add_argument('--lora_grad_decay_factor', default=1.0, type=float,
                        help='Gradient scale factor for LoRA params after lora_grad_decay_after_epoch.')
    parser.add_argument('--lora_delta_lambda', default=0.0, type=float,
                        help='If >0, add normalized LoRA delta-W penalty to reduce few-shot over-writing.')
    parser.add_argument('--lora_delta_mode', default='relative_l2', type=str, choices=['relative_l2', 'absolute_l2'],
                        help='relative_l2 penalizes ||delta_W||/||W_base||; absolute_l2 penalizes mean(delta_W^2).')
    parser.add_argument('--freeze_non_lora_after_epoch', default=-1, type=int,
                        help='If >0, after this epoch freeze non-LoRA backbone and keep LoRA/head selected small modules trainable.')
    parser.add_argument('--lora_head_grad_decay_after_epoch', default=-1, type=int,
                        help='If >0, after this epoch scale LoRA and task_head gradients by lora_head_grad_decay_factor.')
    parser.add_argument('--lora_head_grad_decay_factor', default=1.0, type=float,
                        help='Gradient scale for LoRA/head after lora_head_grad_decay_after_epoch, e.g. 0.1.')
    parser.add_argument('--staged_lora_mode', default='none', type=str, choices=['none', 'eegpt_bridge_then_ffn', 'eegpt_bridge_only'],
                        help='Stage-wise LoRA lifecycle. eegpt_bridge_then_ffn: first train bridge LoRA+head, then freeze bridge and train last2 FFN LoRA+head.')
    parser.add_argument('--staged_lora_bridge_epochs', default=5, type=int,
                        help='For staged_lora_mode=eegpt_bridge_then_ffn: number of early epochs used for bridge-LoRA alignment.')
    parser.add_argument('--staged_lora_bridge_grad_mult', default=1.0, type=float,
                        help='For staged bridge-first LoRA: multiply bridge-LoRA gradients during the bridge stage, e.g. 5.0.')

    # Boundary-anchor LoRA snapshot controls
    parser.add_argument('--boundary_anchor_eval', action='store_true', default=False,
                        help='Save and finally evaluate the best validation-selected boundary-aware LoRA snapshot.')
    parser.add_argument('--boundary_anchor_metric', default='selection_bacc_worst_std', type=str,
                        help='Validation metric used to select the boundary anchor snapshot. The generic default uses worst-class recall and recall stability.')
    parser.add_argument('--boundary_anchor_min_epoch', default=1, type=int,
                        help='Do not consider epochs before this value for boundary anchor selection.')
    parser.add_argument('--boundary_anchor_max_epoch', default=-1, type=int,
                        help='If > 0, do not consider epochs after this value for boundary anchor selection. Use for lifecycle-window anchors.')
    parser.add_argument('--boundary_anchor_tag', default='boundary_anchor', type=str,
                        help='Short monitor checkpoint tag for the selected boundary anchor state.')
    parser.add_argument('--boundary_anchor_strategy', default='best', type=str,
                        choices=['best', 'earliest_top', 'epoch_penalty', 'window_best', 'window_balanced'],
                        help='best: max validation score. earliest_top: earliest epoch within top ratio. epoch_penalty: score - penalty*(epoch-min_epoch). window_best: max score inside [min_epoch, max_epoch]. window_balanced: window_best minus val class0/class2 imbalance.')
    parser.add_argument('--boundary_anchor_top_ratio', default=0.85, type=float,
                        help='For earliest_top: select the earliest epoch with score >= best_score * this ratio.')
    parser.add_argument('--boundary_anchor_epoch_penalty', default=0.02, type=float,
                        help='For epoch_penalty: subtract this value per epoch after min_epoch from boundary score.')
    parser.add_argument('--boundary_anchor_balance_lambda', default=0.25, type=float,
                        help='For window_balanced: subtract lambda * abs(val_class0 - val_class2) from boundary score.')

    # Validation checkpoint selection controls
    parser.add_argument('--selection_worst_alpha', default=0.25, type=float,
                        help='Weight for worst-class recall in selection_* metrics.')
    parser.add_argument('--selection_min02_alpha', default=0.25, type=float,
                        help='Weight for min(class0, class2) recall in selection_* metrics.')
    parser.add_argument('--selection_std_gamma', default=0.10, type=float,
                        help='Penalty weight for recall_std in selection_*_std metrics.')
    parser.add_argument('--selection_hardmix_worst_alpha', default=0.30, type=float,
                        help='Weight for global worst-class recall in selection_bacc_hardmix_std.')
    parser.add_argument('--selection_hardmix_min02_alpha', default=0.35, type=float,
                        help='Weight for min(class0, class2) in selection_bacc_hardmix_std.')
    parser.add_argument('--selection_hardmix_std_gamma', default=0.18, type=float,
                        help='Penalty weight for recall_std in selection_bacc_hardmix_std.')
    parser.add_argument('--selection_hardmix_imbalance_gamma', default=0.10, type=float,
                        help='Penalty for abs(class0 - class2) in selection_bacc_hardmix_std.')
    parser.add_argument('--selection_hardmix_floor', default=0.08, type=float,
                        help='Hard-class floor for min(class0, class2) in selection_bacc_hardmix_std.')
    parser.add_argument('--selection_hardmix_floor_gamma', default=0.25, type=float,
                        help='Penalty strength when min(class0, class2) is below selection_hardmix_floor.')


    # Monitoring / diagnostics parameters
    parser.add_argument('--monitor_dynamics', action='store_true', default=False,
                        help='Enable epoch-level dynamics monitoring for few-shot fine-tuning.')
    parser.add_argument('--eval_train_set', action='store_true', default=False,
                        help='Evaluate the training set again in eval mode after each epoch.')
    parser.add_argument('--diag_freq', default=5, type=int,
                        help='Run spectral diagnostics every N epochs when monitor_dynamics is enabled.')
    parser.add_argument('--save_epoch_ckpt_freq', default=5, type=int,
                        help='Save extra epoch checkpoints every N epochs when monitor_dynamics is enabled.')
    parser.add_argument('--spectral_diag', action='store_true', default=False,
                        help='Enable weight delta spectral diagnostics.')
    parser.add_argument('--max_svd_layers', default=200, type=int,
                        help='Maximum number of layers to run SVD on per diagnostic epoch.')
    parser.add_argument('--max_svd_numel', default=3000000, type=int,
                        help='Skip SVD for tensors with more elements than this value.')
    parser.add_argument('--svd_topk', default=20, type=int,
                        help='Number of top singular values to save.')
    parser.add_argument('--diag_trainable_only', action='store_true', default=False,
                        help='Only run weight delta diagnostics on trainable parameters.')
    parser.add_argument('--best_metric', default='balanced_accuracy', type=str,
                        help='Metric used for saving best checkpoint. Usually balanced_accuracy for EEG classification.')
    parser.add_argument('--module_d_sbr_eval', action='store_true', default=False,
                        help='After a Module D run, write diagnostics/module_d_sbr_eval.csv from a supplied reference CSV.')
    parser.add_argument('--module_d_reference_csv', default='', type=str,
                        help='Reference adaptive_swa_eval-style CSV for Module D SBR, usually a no-D or previous-best baseline.')
    parser.add_argument('--module_d_reference_name', default='', type=str,
                        help='Optional row filter inside --module_d_reference_csv, e.g. qv_ref or previous_best.')
    parser.add_argument('--module_d_hard_k', default=2, type=int,
                        help='Number of hard classes selected from reference validation recall for SBR.')

    # Snapshot-stable evaluation / top-k logit ensemble
    parser.add_argument('--snapshot_eval', action='store_true', default=False,
                        help='After training, load saved epoch checkpoints and evaluate top-k validation-selected logit ensemble.')
    parser.add_argument('--snapshot_topk', default=3, type=int,
                        help='Number of validation-selected epoch snapshots used for logit ensemble.')
    parser.add_argument('--snapshot_select_metric', default='', type=str,
                        help='Metric used to rank snapshots. Empty means args.best_metric. Use names without val_ prefix, e.g. selection_bacc_worst_std.')
    parser.add_argument('--snapshot_include_top1', action='store_true', default=True,
                        help='Also save a top1 snapshot row besides top-k ensemble.')
    parser.add_argument('--snapshot_no_top1', action='store_false', dest='snapshot_include_top1',
                        help='Do not save the top1 snapshot row.')


    # Offline adapter/front-end strength calibration. This is evaluation-only;
    # it reuses saved monitor_checkpoints from previous runs and does not retrain.
    parser.add_argument('--adapter_calib_eval', action='store_true', default=False,
                        help='Evaluation-only mode: sweep LoRA runtime scale / CBra front-end beta on saved epoch checkpoints.')
    parser.add_argument('--adapter_eval_tag', default='', type=str,
                        help='Find an existing output directory whose name contains this tag, e.g. ssv2_bi_s0.')
    parser.add_argument('--adapter_eval_input_dir', default='', type=str,
                        help='Explicit existing output directory to evaluate. If empty, search by adapter_eval_tag.')
    parser.add_argument('--adapter_eval_collect_dir', default='', type=str,
                        help='Directory to save adapter strength calibration CSV results.')
    parser.add_argument('--adapter_eval_exp_name', default='', type=str,
                        help='Short experiment name used in output CSV filenames.')
    parser.add_argument('--adapter_eval_epoch_min', default=1, type=int,
                        help='Minimum epoch checkpoint considered during adapter calibration.')
    parser.add_argument('--adapter_eval_epoch_max', default=-1, type=int,
                        help='Maximum epoch checkpoint considered; <=0 means args.epochs.')
    parser.add_argument('--adapter_eval_topn', default=8, type=int,
                        help='Evaluate only top-N epoch candidates ranked by val metric from epoch_metrics.csv.')
    parser.add_argument('--adapter_eval_metric', default='balanced_accuracy', type=str,
                        help='Validation metric used to rank candidate epochs. Use names without val_ prefix.')
    parser.add_argument('--adapter_lora_scales', default='0.25,0.5,0.75,1.0,1.25', type=str,
                        help='Comma-separated LoRA runtime scales to sweep.')
    parser.add_argument('--adapter_front_betas', default='1.0', type=str,
                        help='Comma-separated CBraMod patch/front-end interpolation betas. Non-CBraMod usually use 1.0.')
    parser.add_argument('--cbra_eval_front_beta', default=1.0, type=float,
                        help='CBraMod only. During standard epoch evaluation, interpolate patch/spectral front-end with init state: W=init+beta*(trained-init). Training and checkpoints remain unchanged.')
    parser.add_argument('--adapter_swa', action='store_true', default=False,
                        help='Maintain a simple running average of trainable floating parameters after adapter_swa_start_epoch.')
    parser.add_argument('--adapter_swa_start_epoch', default=8, type=int,
                        help='First epoch included in adapter_swa running average.')
    parser.add_argument('--adapter_swa_end_epoch', default=-1, type=int,
                        help='Last epoch included in adapter_swa running average. -1 means keep averaging until the final epoch.')
    parser.add_argument('--adapter_swa_eval', action='store_true', default=False,
                        help='Use the running averaged trainable parameters during standard epoch evaluation, then restore current training weights.')
    parser.add_argument('--adapter_swa_trainable_only', action='store_true', default=True,
                        help='Average only parameters that were trainable at monitor init time. Default True.')
    parser.add_argument('--adapter_swa_filter_rank', default=-1, type=int,
                        help='If >0, apply rank-filtering to averaged LoRA A/B pairs before adapter-SWA evaluation. This keeps the shared low-rank adapter directions and removes noisy directions.')


    add_lifecycle_window_args(parser)
    parser.add_argument('--save_eval_state_ckpt', action='store_true', default=False,
                        help='When temporary eval state is used (adapter SWA or CBra front beta), overwrite monitor epoch checkpoint with that eval state so offline FBD/rank reads the same state as official eval.')
    parser.add_argument('--adapter_calib_strengths', default='', type=str,
                        help='Optional comma-separated logit-adjust strengths to sweep. Empty uses current logit_adjust_strength or no calibration.')

    # Functional-block profiling framework parameters.
    add_fb_args(parser)

    # Distributed training parameters
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--dist_eval', action='store_true', default=False, help='enabling distributed evaluation')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)
    parser.add_argument('--auto_resume', action='store_true', default=True)
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume',
                        help='Disable auto-resume. Use this for new short-tag experiments to avoid resuming an old run.')

    known_args, _ = parser.parse_known_args(argv)

    if known_args.enable_deepspeed:
        try:
            try:
                import deepspeed
            except ImportError:
                deepspeed = None
            # from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please install the optional dependencies from requirements-optional.txt")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(argv), ds_init
# -------------------------------------------------------------------------------------------------------------

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, flatten=0, dropout=0, patch_mean=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        self.flatten = flatten
        self.patch_mean = patch_mean
        self.drop_out = nn.Dropout(p=dropout) if dropout else None
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )

    def forward(self, x):
        if self.flatten:
            x = x.flatten(self.flatten)
        elif self.patch_mean:
            x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1)
        if self.drop_out is not None:
            x = self.drop_out(x)
        # if self.doWeightNorm:
        #     self.weight.data = torch.renorm(
        #         self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        #     )
        return super().forward(x)

class RegressionLayers(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, output_dim, flatten=0, patch_mean=False, remove_cls=False):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        self.flatten = flatten
        self.patch_mean = patch_mean
        self.remove_cls = remove_cls
    def forward(self, x):
        if self.remove_cls:
            x = x[..., 1:, :]
        if self.flatten:
            x = x.flatten(self.flatten)
        elif self.patch_mean:
            x = x.reshape(x.shape[0], -1, x.shape[-1]).mean(1)
        out = self.clshead(x)
        return out

# -------------------------------------------------------------------------------------------------------------


def _set_trainable_by_name_keyword(model, keywords, trainable=True, verbose=True):
    """Set requires_grad for parameters whose names contain any keyword."""
    changed = []
    for name, param in model.named_parameters():
        if any(k in name for k in keywords):
            param.requires_grad = bool(trainable)
            changed.append(name)
    if verbose:
        action = "unfreeze" if trainable else "freeze"
        print(f"[CBraDiag] {action} {len(changed)} params by keywords={keywords}")
        for n in changed[:30]:
            print(f"  [CBraDiag] {action}: {n}")
        if len(changed) > 30:
            print(f"  [CBraDiag] ... {len(changed)-30} more")
    return changed


def _cbra_extract_encoder_layer_idx(param_name):
    """Parse CBraMod names like main_model.encoder.layers.10.linear1.base.weight."""
    m = re.search(r'encoder\.layers\.(\d+)\.', str(param_name))
    if m:
        return int(m.group(1))
    return None


def _cbra_max_encoder_layer_idx(model):
    idxs = []
    for name, _ in model.named_parameters():
        idx = _cbra_extract_encoder_layer_idx(name)
        if idx is not None:
            idxs.append(idx)
    return max(idxs) if idxs else None


def _set_cbra_norm_bias_trainable(model, train_bias=True, verbose=True):
    """
    Unfreeze CBraMod normalization / scale-shift parameters and optionally bias terms.

    This is designed for the clean CBraMod diagnosis:
    Full FT showed non-trivial norm/bias delta, while old C2/CFI kept them frozen.
    """
    changed = []
    for name, param in model.named_parameters():
        lower = name.lower()
        is_norm = ("norm" in lower) or ("layernorm" in lower) or (".ln" in lower)
        is_bias = lower.endswith(".bias")
        if is_norm or (train_bias and is_bias):
            param.requires_grad = True
            changed.append(name)
    if verbose:
        print(f"[CBraNormBitFit] unfreeze {len(changed)} norm/bias params (train_bias={train_bias})")
        for n in changed[:60]:
            print(f"  [CBraNormBitFit] {n}")
        if len(changed) > 60:
            print(f"  [CBraNormBitFit] ... {len(changed)-60} more")
    return changed


def _set_lora_wrapped_base_trainable(model, last_n=-1, verbose=True):
    """
    Unfreeze base parameters inside LoRA wrappers.

    LoRALinear / LoRAMultiheadAttention freeze their original base by design.
    For CBraMod, module-delta diagnosis showed clean Full FT relies heavily on
    original FFN and attention mixing updates. This flag lets us test targeted
    partial adaptation without unfreezing the whole backbone.

    If last_n > 0, only unfreeze wrapped base params from the last N encoder layers.
    """
    max_idx = _cbra_max_encoder_layer_idx(model)
    cutoff = None
    if last_n is not None and int(last_n) > 0 and max_idx is not None:
        cutoff = max_idx - int(last_n) + 1

    changed = []
    skipped = []
    for name, param in model.named_parameters():
        if ".base." not in name:
            continue
        idx = _cbra_extract_encoder_layer_idx(name)
        if cutoff is not None:
            if idx is None or idx < cutoff:
                skipped.append(name)
                continue
        param.requires_grad = True
        changed.append(name)

    if verbose:
        print(f"[CBraWrappedBase] unfreeze {len(changed)} LoRA-wrapped base params (last_n={last_n}, max_idx={max_idx}, cutoff={cutoff})")
        for n in changed[:80]:
            print(f"  [CBraWrappedBase] {n}")
        if len(changed) > 80:
            print(f"  [CBraWrappedBase] ... {len(changed)-80} more")
        if skipped:
            print(f"[CBraWrappedBase] skipped {len(skipped)} wrapped base params outside selected layers")
    return changed


def _cbra_l2sp_should_track(name, scope):
    """Whether a trainable CBraMod parameter should be constrained to its init value."""
    lower = str(name).lower()
    if "lora_" in lower:
        return False
    if "task_head" in lower or "classifier" in lower:
        return False

    is_wrapped_base = ".base." in name
    is_patch_front = (
        "patch_embedding" in lower
        or "spectral" in lower
        or "positional" in lower
        or "position" in lower
    )
    is_norm_bias = (
        "norm" in lower
        or "layernorm" in lower
        or ".ln" in lower
        or lower.endswith(".bias")
    )

    if scope == "wrapped_base":
        return is_wrapped_base
    if scope == "wrapped_base_patch":
        return is_wrapped_base or is_patch_front
    if scope == "wrapped_base_patch_norm":
        return is_wrapped_base or is_patch_front or is_norm_bias
    if scope == "all_trainable_non_lora_head":
        return True
    return is_wrapped_base or is_patch_front or is_norm_bias


def _register_cbra_l2sp_reference(model, args, verbose=True):
    """
    Store initialization references for CBraMod L2-SP.

    This is used for the CBraMod D3 conservative partial-adaptation test:
    allow patch/front + late FFN base to adapt, but penalize drifting too far
    from the starting pretrained/task-init parameters.
    """
    lam = float(getattr(args, "cbra_l2sp_lambda", 0.0))
    if lam <= 0.0:
        return 0
    if getattr(args, "model_name", None) != "CBraMod":
        return 0

    scope = str(getattr(args, "cbra_l2sp_scope", "wrapped_base_patch_norm"))
    refs = {}
    total_numel = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not _cbra_l2sp_should_track(name, scope):
            continue
        refs[name] = param.detach().clone().cpu()
        total_numel += int(param.numel())

    model._l2sp_ref = refs
    model._l2sp_lambda = lam
    model._l2sp_scope = scope

    if verbose:
        print(f"[CBraL2SP] enabled lambda={lam:g}, scope={scope}, tensors={len(refs)}, numel={total_numel}")
        for n in list(refs.keys())[:80]:
            print(f"  [CBraL2SP] track: {n}")
        if len(refs) > 80:
            print(f"  [CBraL2SP] ... {len(refs)-80} more")
    return len(refs)

# -----------------------------------------Custom Classes for models---------------------------------------------


class Ada_LaBraM(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        # Load LaBraM model
        model = create_model(
            'labram_base_patch200_200',
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=True,
            init_scale=0.001,
            use_rel_pos_bias=True,
            use_abs_pos_emb=True,
            init_values=0.1,
            qkv_bias=True,
            num_ch=len(ch_names),
            num_t=num_t 
        )
        

        # load the pre-trained weights.
        if from_pretrain:
            if finetune_list[args.model_name].startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    finetune_list[args.model_name], map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(finetune_list[args.model_name], map_location='cpu', weights_only=False)

            print("Load ckpt from %s" % finetune_list[args.model_name])
            checkpoint_model = None
            args.model_key = 'model|module'
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            if (checkpoint_model is not None):
                all_keys = list(checkpoint_model.keys())
                new_dict = OrderedDict()
                for key in all_keys:
                    if key.startswith('student.'):
                        new_dict[key[8:]] = checkpoint_model[key]
                    else:
                        pass
                checkpoint_model = new_dict

            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

            utils.load_state_dict(model, checkpoint_model)
        
        
        
        model.head = nn.Identity()
        self.main_model = model
        self.ch_names = ch_names

        self.task_head=nn.Identity()
        
    
    def forward(self, x):
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        input_chans = utils.get_input_chans(self.ch_names)
        output = self.main_model(x, input_chans, return_all_tokens=True)
        output = self.task_head(output)
        return output

class Ada_CBraMod(nn.Module):
    def __init__(self, args, from_pretrain=False):
        super().__init__()
        model = CBraMod()
        
        if from_pretrain:
            print("Load ckpt from %s" % finetune_list[args.model_name])
            model.load_state_dict(torch.load(finetune_list[args.model_name], map_location=torch.device('cpu')))
        
        model.proj_out = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()
        
    
    def forward(self, x):
        b, n, t = x.shape
        x = x.reshape(b, n, -1, 200)
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_EEGPT(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()

        with open("./util/eegpt_use_channels_names.json", "r") as f:
            model_channels = json.load(f)
        use_channels_names = model_channels.get(args.dataset, None)
        use_channels_names = use_channels_names.split(", ") if use_channels_names is not None else ch_names
        chans_num = len(use_channels_names)

        # init model
        model = EEGTransformer(
            img_size=[chans_num, 256 * num_t],
            patch_size=32 * 2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.chans_id = model.prepare_chan_ids(use_channels_names)

        if from_pretrain:
            print(f"Load ckpt from {finetune_list[args.model_name]}")
            checkpoint_path = finetune_list[args.model_name]
            pretrain_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            target_encoder_stat = {}
            for k, v in pretrain_ckpt['state_dict'].items():
                if k.startswith("target_encoder."):
                    target_encoder_stat[k[15:]] = v
            model.load_state_dict(target_encoder_stat)
        
        self.main_model = model
        self.chan_conv = Conv1dWithConstraint(len(ch_names), chans_num, 1, max_norm=1)
        self.task_head = nn.Identity()

    def forward(self, x):
        x = self.chan_conv(x)
        output = self.main_model(x, self.chans_id.to(x))
        output = self.task_head(output)
        return output

class Ada_BIOT(nn.Module):
    def __init__(self, args, ch_names, from_pretrain=False):
        super().__init__()
        in_channels = 18

        model = BIOTClassifier(n_classes=args.nb_classes, n_channels=in_channels, n_fft=200, hop_length=100)
        if from_pretrain:
            model.biot.load_state_dict(torch.load(finetune_list[args.model_name]))
            print(f"load pretrain model from {finetune_list[args.model_name]}")
        
        model.classifier = nn.Identity()
        self.main_model = model
        self.chan_conv = Conv1dWithConstraint(len(ch_names), in_channels, 1, max_norm=1)

        self.task_head=nn.Identity()

    def forward(self, x):
        x = self.chan_conv(x)
        output = self.main_model(x)
        output = self.task_head(output)
        return output


class Ada_CSBrain(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        # TUEV official CSBrain uses 16 bipolar signal channels and fixed brain-region grouping.
        target_channels = 16
        tuev_brain_regions = [
            0, 0, 2, 2, 0, 0, 2, 2,
            0, 0, 4, 1, 0, 0, 4, 1,
        ]
        tuev_signal_electrodes = [
            "FP1", "F7", "T3", "T5", "FP2", "F8", "T4", "T6",
            "FP1", "F3", "C3", "P3", "FP2", "F4", "C4", "P4",
        ]
        tuev_topology = {
            0: ["FP1", "F3", "F7", "FZ", "F4", "F8", "FP2"],
            4: ["C3", "CZ", "C4"],
            1: ["P3", "PZ", "P4"],
            2: ["T3", "T5", "T6", "T4"],
            3: ["O1", "O2"],
        }
        region_groups = {}
        for i, region in enumerate(tuev_brain_regions):
            region_groups.setdefault(region, []).append((i, tuev_signal_electrodes[i]))
        sorted_indices = []
        for region in sorted(region_groups.keys()):
            sorted_electrodes = sorted(region_groups[region], key=lambda x: tuev_topology[region].index(x[1]))
            sorted_indices.extend([e[0] for e in sorted_electrodes])

        self.chan_conv = nn.Identity()
        if len(ch_names) != target_channels:
            print(f"[CSBrain-Ada] channel adapter enabled: Ada channels={len(ch_names)} -> CSBrain channels={target_channels}")
            self.chan_conv = Conv1dWithConstraint(len(ch_names), target_channels, 1, max_norm=1)

        model = CSBrain(
            in_dim=200,
            out_dim=200,
            d_model=200,
            dim_feedforward=800,
            seq_len=int(num_t),
            n_layer=12,
            nhead=8,
            brain_regions=tuev_brain_regions,
            sorted_indices=sorted_indices,
        )

        if from_pretrain:
            ckpt_path = getattr(args, 'csbrain_ckpt', '') or finetune_list.get('CSBrain', './checkpoints/CSBrain.pth')
            if os.path.exists(ckpt_path):
                print(f"Load ckpt from {ckpt_path}")
                state = torch.load(ckpt_path, map_location='cpu')
                state = state.get('model', state.get('state_dict', state)) if isinstance(state, dict) else state
                state = {str(k).replace('module.', ''): v for k, v in state.items()}
                cur = model.state_dict()
                matched = {k: v for k, v in state.items() if k in cur and hasattr(v, 'shape') and cur[k].shape == v.shape}
                cur.update(matched)
                model.load_state_dict(cur, strict=False)
                print(f"[CSBrain-Ada] loaded matched tensors={len(matched)}")
                if len(matched) == 0 and not bool(getattr(args, 'csbrain_allow_scratch', False)):
                    raise RuntimeError(
                        "[CSBrain-Ada] matched tensor count is 0. This would be a random/scratch baseline; "
                        "check CSBrain.pth or add --csbrain_allow_scratch only for interface debugging."
                    )
            else:
                msg = f"[CSBrain-Ada] checkpoint not found: {ckpt_path}"
                if bool(getattr(args, 'csbrain_allow_scratch', False)):
                    print(msg + " ; --csbrain_allow_scratch is set, so this is only a scratch/debug run.")
                else:
                    raise FileNotFoundError(
                        msg + "\nFor a real foundation-model baseline, put CSBrain.pth at this path or pass --csbrain_ckpt. "
                        "Use --csbrain_allow_scratch only for interface debugging."
                    )

        model.proj_out = nn.Identity()
        self.main_model = model
        self.task_head = nn.Identity()

    def forward(self, x):
        b, n, t = x.shape
        x = self.chan_conv(x)
        if t % 200 != 0:
            raise ValueError(f"CSBrain expects time length divisible by 200, got {t}")
        x = x.reshape(b, x.shape[1], -1, 200)
        output = self.main_model(x)
        output = self.task_head(output)
        return output


class Ada_Gram(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        if GramAdaBackbone is None:
            raise ImportError('models.gram_ada could not be imported. Put gram_ada.py under models/.')
        self.main_model = GramAdaBackbone(args, ch_names, num_t, from_pretrain=from_pretrain)
        self.task_head = nn.Identity()

    def forward(self, x):
        return self.task_head(self.main_model(x))




class Ada_EEGNet(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = EEGNet(chans=len(ch_names), classes=args.nb_classes, time_points=num_t * 200)
        model.fc = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()

    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_LMDA(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = LMDA(num_classes=args.nb_classes, chans=len(ch_names), samples=num_t * 200, channel_depth1=24, channel_depth2=7)
        model.classifier = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()

    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_EEGConformer(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = Conformer(C=len(ch_names), time_points=num_t * 200, n_classes=args.nb_classes)
        model.classification_head = nn.Identity()
        self.main_model = model

        self.task_head=nn.Identity()

    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

class Ada_STTransformer(nn.Module):
    def __init__(self, args, ch_names, num_t):
        super().__init__()
        model = STTransformer(n_classes=args.nb_classes, channel_legnth=num_t * 200, n_channels=len(ch_names))
        self.main_model = model
        self.task_head=nn.Identity()
        
    def forward(self, x):
        output = self.main_model(x)
        output = self.task_head(output)
        return output

# --------------------------------------------------------------------------------------------------

# -----------------------------Load the models based on args.model_name------------------------------
def _register_lora_gradient_scaler(model):
    """Register gradient hooks so LoRA params can be softly slowed later.

    This avoids rebuilding optimizer parameter groups. The mutable holder is
    attached to model and updated at the beginning of each epoch.
    """
    holder = {"scale": 1.0}
    hook_count = 0

    def _make_hook(h):
        def _hook(grad):
            return grad * float(h["scale"])
        return _hook

    for name, p in model.named_parameters():
        if "lora_" in name and p.requires_grad:
            p.register_hook(_make_hook(holder))
            hook_count += 1
    model._lora_grad_scale_holder = holder
    print(f"[LoRA] registered gradient scaler hooks on {hook_count} LoRA tensors")
    return holder


def _set_lora_gradient_scale(model, args, epoch_id: int):
    holder = getattr(model, "_lora_grad_scale_holder", None)
    if holder is None:
        return 1.0
    after = int(getattr(args, "lora_grad_decay_after_epoch", -1))
    factor = float(getattr(args, "lora_grad_decay_factor", 1.0))
    scale = factor if (after > 0 and epoch_id > after) else 1.0
    holder["scale"] = scale
    if scale != 1.0:
        print(f"[LoRA] epoch {epoch_id}: LoRA gradient scale = {scale}")
    return scale


def _parse_semicolon_names(value):
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(v).strip() for v in value if str(v).strip())
    text = str(value or "").replace("\n", ";").replace(",", ";")
    return tuple(part.strip() for part in text.split(";") if part.strip())


def _is_module_c_execution_target(target):
    return str(target or "").lower() in ("module_c", "module_c_auto", "c_auto")


def _module_c_selection_contains_e(args):
    selected = getattr(args, "module_c_resolved_selected", getattr(args, "module_c_selected", ""))
    tokens = set()
    for token in re.split(r"[,\s;|]+", str(selected or "")):
        token = token.strip().upper()
        if token:
            tokens.add(token)
    return "E" in tokens


def _module_e_lora_requested(args):
    target = str(getattr(args, "lora_target", "") or "").lower()
    if target in ("str", "struct", "structural", "struct_mix", "mix", "spatial_attn", "temporal_attn") or is_module_e_target(target):
        return True
    if _is_module_c_execution_target(target) and _module_c_selection_contains_e(args):
        return True
    return False


def _module_e_mode(args):
    return module_e_mode_from_args(args)


def _module_e_dynamic_pressure_requested(args):
    return (
        str(getattr(args, "finetune_mod", "")).lower() == "lora"
        and str(getattr(args, "lora_target", "")).lower() != "none"
        and _module_e_mode(args) == "dynamic_pressure_gate"
        and _module_e_lora_requested(args)
    )


def _attach_requested_module_e_controller(args, model):
    if not _module_e_dynamic_pressure_requested(args):
        return None
    if bool(getattr(args, "enable_deepspeed", False)):
        raise RuntimeError("Dynamic Module E with DeepSpeed is unsupported in this patch.")
    try:
        controller = attach_module_e_dynamic_pressure_controller(args, model)
    except Exception as exc:
        raise RuntimeError(f"Requested Module E attachment failed: {exc}") from exc
    if controller is None:
        detail = f": {_fb_import_error}" if _fb_import_error is not None else ""
        raise RuntimeError(f"Requested Module E attachment returned no controller{detail}")
    return controller


def _create_formal_optimizer(args, model, skip_weight_decay_list, assigner=None, controller=None):
    try:
        optimizer = create_optimizer(
            args,
            model,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
            get_param_group_tag=(controller.optimizer_group_tag if controller is not None else None),
        )
        if controller is not None:
            controller.bind_optimizer(optimizer)
        return optimizer
    except Exception as exc:
        if controller is not None:
            raise RuntimeError(f"Requested Module E optimizer bind failed: {exc}") from exc
        raise


def _apply_lora_training_setup(model, args):
    if args.task_mod == 'Retrieval':
        raise ValueError('LoRA mode is currently implemented for Classification/Regression, not Retrieval.')
    if args.model_name not in ['LaBraM', 'CBraMod', 'EEGPT', 'BIOT', 'CSBrain', 'Gram']:
        raise ValueError(f'LoRA mode currently supports EEGFMs only, got {args.model_name}.')

    # Important experimental control:
    #   lora_base_update=freeze -> standard Frozen-LoRA. Original W is frozen; LoRA/head are trainable.
    #   lora_base_update=full   -> Full FT + LoRA. Original W remains trainable; LoRA is an extra branch.
    # This separates the freeze variable from the LoRA variable.
    pre_lora_trainability = {
        id(parameter): bool(parameter.requires_grad) for parameter in model.parameters()
    }
    module_c_selected_text = getattr(args, 'module_c_resolved_selected', getattr(args, 'module_c_selected', ''))
    module_c_selected_tokens = [
        token.strip().upper()
        for token in re.split(r"[,;\s|]+", str(module_c_selected_text or ""))
        if token.strip()
    ]
    explicit_target_none = str(args.lora_target).lower() == 'none'
    target_module_c_empty = _is_module_c_execution_target(args.lora_target) and len(module_c_selected_tokens) == 0
    if target_module_c_empty:
        raise RuntimeError(
            'Module C requires a nonempty B/D/E selection before LoRA injection. '
            'Keep automatic preflight enabled or provide --module_c_selected.'
        )
    target_none = explicit_target_none

    if args.lora_base_update == 'freeze':
        freeze_all_parameters(model)

    if target_none:
        if explicit_target_none and args.lora_base_update != 'freeze':
            raise ValueError('lora_target=none is intended for frozen-backbone diagnosis only; use --lora_base_update freeze.')
        print('[LoRA] lora_target=none: no LoRA modules will be injected. Selected modules will be unfrozen below.')
        if args.lora_base_update == 'freeze':
            if args.lora_train_head and hasattr(model, 'task_head'):
                unfreeze_module(model.task_head)
            if args.lora_train_chan_conv and hasattr(model, 'chan_conv'):
                unfreeze_module(model.chan_conv)
        elif args.lora_base_update == 'full':
            if not args.lora_train_head and hasattr(model, 'task_head'):
                for p in model.task_head.parameters():
                    p.requires_grad = False
        else:
            raise ValueError(f'Unknown lora_base_update={args.lora_base_update}')
        replaced = []
    else:
        replaced = apply_lora_to_eegfm(
            model=model,
            model_name=args.model_name,
            lora_target=args.lora_target,
            module_c_selected=getattr(args, 'module_c_resolved_selected', getattr(args, 'module_c_selected', '')),
            module_b_sites=getattr(args, 'module_b_sites', 'both'),
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            module_c_seed=int(getattr(args, 'seed', 0)),
            verbose=True,
        )
        if len(replaced) == 0:
            raise RuntimeError(
                f'No LoRA module was injected for model={args.model_name}, target={args.lora_target}. '
                'Please check module names or Module E pressure-selected targets.'
            )

        if args.lora_base_update == 'freeze':
            mark_lora_and_selected_modules_trainable(
                model,
                train_task_head=args.lora_train_head,
                train_chan_conv=args.lora_train_chan_conv,
            )
        elif args.lora_base_update == 'full':
            # LoRA wrappers freeze their nested base modules when constructed.
            # Restore every pre-existing parameter to its pre-injection state so
            # full means the same Full FT base for every selected action set.
            for parameter in model.parameters():
                parameter_id = id(parameter)
                if parameter_id not in pre_lora_trainability:
                    continue
                restore = pre_lora_trainability[parameter_id]
                if restore and not (parameter.is_floating_point() or parameter.is_complex()):
                    raise RuntimeError('A non-differentiable pre-LoRA parameter was unexpectedly trainable.')
                parameter.requires_grad_(restore)
            if not args.lora_train_head and hasattr(model, 'task_head'):
                for p in model.task_head.parameters():
                    p.requires_grad = False
        else:
            raise ValueError(f'Unknown lora_base_update={args.lora_base_update}')

    if _module_e_lora_requested(args):
        e_replaced = [
            name for name in replaced
            if module_e_branch_from_lora_param_name(args.model_name, name) is not None
        ]
        args.module_e_injected_names = ";".join(e_replaced)
    else:
        args.module_e_injected_names = ";".join(replaced)

    # CBraMod mechanism diagnosis controls.
    if args.model_name == 'CBraMod' and getattr(args, 'cbra_train_patch_embed_when_frozen', False):
        _set_trainable_by_name_keyword(
            model,
            keywords=['main_model.patch_embedding'],
            trainable=True,
            verbose=True,
        )
    if args.model_name == 'CBraMod' and getattr(args, 'cbra_freeze_patch_embed_in_full', False):
        _set_trainable_by_name_keyword(
            model,
            keywords=['main_model.patch_embedding'],
            trainable=False,
            verbose=True,
        )

    if args.model_name == 'CBraMod' and getattr(args, 'cbra_train_wrapped_base', False):
        _set_lora_wrapped_base_trainable(
            model,
            last_n=int(getattr(args, 'cbra_train_wrapped_base_last_n', -1)),
            verbose=True,
        )

    if args.model_name == 'CBraMod' and getattr(args, 'cbra_train_norm_bias', False):
        _set_cbra_norm_bias_trainable(
            model,
            train_bias=not bool(getattr(args, 'cbra_train_norm_only', False)),
            verbose=True,
        )

    print(f'[LoRA] base_update={args.lora_base_update}')
    print_trainable_parameters(model)
    if not target_none:
        _register_lora_gradient_scaler(model)
    return replaced


def _build_module_e_pressure_criterion(args, device):
    if args.task_mod == 'Regression':
        return torch.nn.MSELoss()
    return _build_classification_criterion(
        args,
        device,
        is_binary=(int(getattr(args, "nb_classes", 0)) == 1),
        epoch_id=1,
    )


def _register_cbra_grad_scale_hooks(model, args, verbose=True):
    """
    CBraMod group-LR approximation via gradient multipliers.

    We keep a single optimizer for minimal code disruption, but scale gradients
    for sensitive parameter groups:
      - wrapped base FFN parameters
      - patch/front/positional parameters
      - norm/bias parameters

    This is used because CBraMod-D1/D3 showed that patch/front + late FFN base
    can open class0/class2, but overly strong updates hurt class1/class3.
    """
    if getattr(args, "model_name", None) != "CBraMod":
        return 0
    if bool(getattr(model, "_cbra_grad_scale_hooks_registered", False)):
        return 0

    base_scale = float(getattr(args, "cbra_grad_scale_wrapped_base", 1.0))
    patch_scale = float(getattr(args, "cbra_grad_scale_patch", 1.0))
    norm_scale = float(getattr(args, "cbra_grad_scale_norm_bias", 1.0))

    if base_scale == 1.0 and patch_scale == 1.0 and norm_scale == 1.0:
        return 0

    def make_hook(scale):
        def _hook(grad):
            return grad * float(scale)
        return _hook

    hooked = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        scale = 1.0
        group = None

        if ".base." in name:
            scale = base_scale
            group = "wrapped_base"
        elif (
            "patch_embedding" in lower
            or "spectral" in lower
            or "positional" in lower
            or "position" in lower
        ):
            scale = patch_scale
            group = "patch_front"
        elif (
            "norm" in lower
            or "layernorm" in lower
            or ".ln" in lower
            or lower.endswith(".bias")
        ):
            scale = norm_scale
            group = "norm_bias"

        if group is not None and float(scale) != 1.0:
            param.register_hook(make_hook(scale))
            hooked.append((name, group, scale))

    model._cbra_grad_scale_hooks_registered = True
    if verbose:
        print(
            f"[CBraGradScale] hooks={len(hooked)} "
            f"base={base_scale:g}, patch={patch_scale:g}, norm_bias={norm_scale:g}"
        )
        for n, g, s in hooked[:80]:
            print(f"  [CBraGradScale] {g} scale={s:g}: {n}")
        if len(hooked) > 80:
            print(f"  [CBraGradScale] ... {len(hooked)-80} more")
    return len(hooked)


def _build_module_c_preflight_probe_model(args, ch_names, num_t):
    """Build the disposable pretrained model used by Module C search."""
    probe_args = copy.copy(args)
    probe_args.finetune_mod = "full"
    probe_args.lora_target = "none"
    probe_args.module_c_selected = ""
    probe_args.module_c_resolved_selected = ""
    probe_args.module_e_lora_deferred = False
    return get_models(probe_args, ch_names, num_t)


def get_models(args, ch_names, num_t):
    from_pretrain = False
    if (
        not getattr(args, 'disable_pretrained_loading', False)
        and args.model_name in ['LaBraM', 'CBraMod', 'EEGPT', 'BIOT', 'CSBrain', 'Gram']
    ):
        if args.finetune_mod in ['full', 'linear', 'lora']:
            from_pretrain=True
    elif getattr(args, 'disable_pretrained_loading', False):
        print('[Pretrain] disabled by --disable_pretrained_loading; this run is a random/scratch control.')
 
    # init models
    if args.model_name == 'LaBraM':
        model = Ada_LaBraM(args, ch_names, num_t, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint((len(ch_names) * num_t + 1) * 200, args.nb_classes, max_norm=1, flatten=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=200, hidden_dim=200, output_dim=1, patch_mean=True, remove_cls=True)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint((len(ch_names) * num_t + 1) * 200, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'CBraMod':
        model = Ada_CBraMod(args, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(len(ch_names) * num_t * 200, args.nb_classes, max_norm=1, flatten=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=(len(ch_names) * num_t) * 200, hidden_dim=200, output_dim=1, flatten=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(len(ch_names) * num_t * 200, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'EEGPT':
        model = Ada_EEGPT(args, ch_names, num_t, from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = nn.Sequential(
                LinearWithConstraint(2048, 16, max_norm=1, flatten=2, dropout=0.5),
                LinearWithConstraint(4 * num_t * 16, args.nb_classes, max_norm=0.25, flatten=1)
            )
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=512, hidden_dim=256, output_dim=1, patch_mean=True)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(4 * 2048, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'BIOT':
        model = Ada_BIOT(args, ch_names, from_pretrain=from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(256, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=256, hidden_dim=256, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(256, 1024, max_norm=1)
    elif args.model_name == 'CSBrain':
        model = Ada_CSBrain(args, ch_names, num_t, from_pretrain=from_pretrain)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(16 * num_t * 200, args.nb_classes, max_norm=1, flatten=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=16 * num_t * 200, hidden_dim=200, output_dim=1, flatten=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(16 * num_t * 200, 1024, max_norm=1, flatten=1)
    elif args.model_name == 'Gram':
        model = Ada_Gram(args, ch_names, num_t, from_pretrain=from_pretrain)
        # Official Gram fine-tune model already returns task logits.
        model.task_head = nn.Identity()
    elif args.model_name == 'EEGNet':
        model = Ada_EEGNet(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.linear_size, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=model.linear_size, hidden_dim=200, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.linear_size, 1024, max_norm=1)
    elif args.model_name == 'LMDA':
        model = Ada_LMDA(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.linear_size, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=model.linear_size, hidden_dim=200, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.linear_size, 1024, max_norm=1)
    elif args.model_name == 'EEGConformer':
        model = Ada_EEGConformer(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(model.time_points * 40, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=model.time_points * 40, hidden_dim=40, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(model.time_points * 40, 1024, max_norm=1)
    elif args.model_name == 'ST-Transformer':
        model = Ada_STTransformer(args, ch_names, num_t)
        if args.task_mod == 'Classification':
            model.task_head = LinearWithConstraint(256, args.nb_classes, max_norm=1)
        elif args.task_mod == 'Regression':
            model.task_head = RegressionLayers(input_dim=256, hidden_dim=256, output_dim=1)
        elif args.task_mod == 'Retrieval':
            model.task_head = LinearWithConstraint(256, 1024, max_norm=1)
    else:
        print("Unknown model name!")
        exit(0)
    
    # Expose raw input channel count for universal input-side signal-alignment LoRA.
    # This is diagnostic metadata only; it does not change model forward/protocol.
    try:
        model.input_channels = len(ch_names)
    except Exception:
        pass

    # check task head
    if model.task_head is None:
        print("Task head is None, please check your args or code.")
        exit(0)
    
    if args.finetune_mod == 'linear':
        for p in model.main_model.parameters():
            p.requires_grad = False

    if args.finetune_mod == 'lora':
        args.module_e_lora_deferred = False
        _apply_lora_training_setup(model, args)
    
    # add modules for retrieval
    if args.task_mod == 'Retrieval':
        model.loss_scale = nn.Parameter(torch.tensor(1.0))
        model.loss_func = ClipLoss()

    return model
# ----------------------------------------------------------------------------------------------------------------

# ------------------------------------------Load the dataset-------------------------------------------------------
def get_datasets(args, dataset_info):
    root = dataset_info['root'][args.subject_mod]
    if args.subject_mod == 'fewshot':
        dataset_train = utils.FewShotDataLoader(root + '/train.json', args.sampling_rate, args.norm_method, k_shot=args.k_shot)
        dataset_val = utils.CustomDataLoader(root + '/val.json', args.sampling_rate, args.norm_method)
    else:
        if os.path.exists(root + '/val.json'):
            dataset_train = utils.CustomDataLoader(root + '/train.json', args.sampling_rate, args.norm_method)
            dataset_val = utils.CustomDataLoader(root + '/val.json', args.sampling_rate, args.norm_method)
        else:
            dataset_train = None
            dataset_val = None
            for i in range(args.max_subject):
                subject_dataset = utils.CustomDataLoader(root + '/train.json', args.sampling_rate, args.norm_method, cross=True, subject_id=i)
                train_size = int(0.8 * len(subject_dataset))
                valid_size = len(subject_dataset) - train_size
                train_dataset, valid_dataset = random_split(subject_dataset, [train_size, valid_size])
                if dataset_train is None:
                    dataset_train = train_dataset
                    dataset_val = valid_dataset
                else:
                    dataset_train = ConcatDataset([dataset_train, train_dataset])
                    dataset_val = ConcatDataset([dataset_val, valid_dataset])
    
    dataset_test = utils.CustomDataLoader(root + '/test.json', args.sampling_rate, args.norm_method)
    ch_names = dataset_test.get_ch_names()
    ch_names = [ch.upper() for ch in ch_names]
    args.nb_classes = dataset_info['num_classes']
    if args.nb_classes == 2:
        args.nb_classes = 1
    return dataset_train, dataset_test, dataset_val, ch_names
# -------------------------------------------------------------------------------------------------------------


# ------------------------------------------Monitoring utilities------------------------------------------------

def _to_builtin_scalar(x):
    """
    把 numpy / torch scalar 转成 Python 原生类型，方便写 CSV / JSON。
    """
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return str(tuple(x.shape))
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    try:
        return float(x)
    except Exception:
        return str(x)


def _append_csv_row(csv_path, row):
    """
    追加写入一行 CSV。
    如果文件不存在，则自动写 header。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    clean_row = {k: _to_builtin_scalar(v) for k, v in row.items()}

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)


def _append_per_class_recall(csv_path, epoch, split, per_class_recall):
    """
    保存每个 epoch、每个 split 的 per-class recall。
    格式：epoch, split, class_0, class_1, ...
    """
    if per_class_recall is None:
        return

    row = {
        "epoch": epoch,
        "split": split,
    }

    for i, value in enumerate(per_class_recall):
        if np.isnan(value):
            row[f"class_{i}"] = ""
        else:
            row[f"class_{i}"] = float(value)

    _append_csv_row(csv_path, row)


def _get_model_cpu_state(model):
    """
    保存当前模型 state_dict 到 CPU。
    这里只保存 tensor，避免乱七八糟对象进来。
    """
    state = {}
    for k, v in model.state_dict().items():
        if torch.is_tensor(v):
            state[k] = v.detach().cpu().clone()
    return state


def _get_trainable_param_names(model):
    """
    记录哪些参数是 requires_grad=True。
    后面做谱分析时可以标记 trainable / frozen。
    """
    return {name for name, p in model.named_parameters() if p.requires_grad}


def _save_monitor_checkpoint(args, model, epoch, tag):
    """
    额外保存 epoch checkpoint。
    不替代原本 utils.save_model，只用于后面诊断训练动态。
    """
    ckpt_dir = os.path.join(args.output_dir, "monitor_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pth")
    torch.save({
        "epoch": epoch,
        "model": _get_model_cpu_state(model),
        "args": vars(args),
    }, ckpt_path)

    print(f"[Monitor] checkpoint saved to: {ckpt_path}")


def _effective_rank_from_singular_values(s):
    """
    effective rank = exp(entropy(normalized_energy))
    """
    if s.numel() == 0:
        return float("nan")

    energy = s.float() ** 2
    total = energy.sum()

    if total.item() <= 0:
        return 0.0

    prob = energy / total
    entropy = -(prob * torch.log(prob + 1e-12)).sum()
    return float(torch.exp(entropy).item())


def _top_energy(s, topk):
    """
    前 topk 个 singular values 的能量占比。
    """
    if s.numel() == 0:
        return float("nan")

    energy = s.float() ** 2
    total = energy.sum()

    if total.item() <= 0:
        return 0.0

    k = min(topk, s.numel())
    return float(energy[:k].sum().item() / total.item())


def _run_weight_delta_diagnostics(args, model, init_state, trainable_names, epoch):
    """
    对每一层计算：
    - ||delta W||
    - ||W0||
    - ||delta W|| / ||W0||
    - top-1 / top-4 / top-8 singular energy
    - effective rank
    - top singular values

    注意：这里分析的是 delta W = W_epoch - W_init，不是直接分析原始 W。
    """
    if init_state is None:
        return

    diag_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    csv_path = os.path.join(diag_dir, f"weight_delta_epoch_{epoch:03d}.csv")
    current_state = _get_model_cpu_state(model)

    rows = []
    svd_count = 0

    for name, current_tensor in current_state.items():
        if name not in init_state:
            continue

        base_tensor = init_state[name]

        if current_tensor.shape != base_tensor.shape:
            continue

        if not torch.is_floating_point(current_tensor):
            continue

        is_trainable = name in trainable_names
        if args.diag_trainable_only and not is_trainable:
            continue

        current_float = current_tensor.float()
        base_float = base_tensor.float()
        delta = current_float - base_float

        delta_norm = float(torch.norm(delta).item())
        base_norm = float(torch.norm(base_float).item())
        relative_delta_norm = delta_norm / (base_norm + 1e-12)

        row = {
            "epoch": epoch,
            "name": name,
            "shape": str(tuple(current_tensor.shape)),
            "numel": int(current_tensor.numel()),
            "is_trainable": int(is_trainable),
            "delta_norm": delta_norm,
            "base_norm": base_norm,
            "relative_delta_norm": relative_delta_norm,
            "top1_energy": "",
            "top4_energy": "",
            "top8_energy": "",
            "effective_rank": "",
            "singular_values_topk": "",
        }

        can_svd = (
            args.spectral_diag
            and delta.ndim >= 2
            and delta.numel() <= args.max_svd_numel
            and svd_count < args.max_svd_layers
            and delta_norm > 0
        )

        if can_svd:
            try:
                # Conv / Linear 都统一压成 [out_dim, in_dim-like]
                mat = delta.reshape(delta.shape[0], -1)

                if min(mat.shape) > 0:
                    s = torch.linalg.svdvals(mat)
                    s = s.detach().cpu()

                    row["top1_energy"] = _top_energy(s, 1)
                    row["top4_energy"] = _top_energy(s, 4)
                    row["top8_energy"] = _top_energy(s, 8)
                    row["effective_rank"] = _effective_rank_from_singular_values(s)

                    topk = min(args.svd_topk, s.numel())
                    row["singular_values_topk"] = ";".join([f"{x:.8e}" for x in s[:topk].tolist()])

                    svd_count += 1

            except RuntimeError as e:
                row["singular_values_topk"] = f"SVD_FAILED: {str(e)[:120]}"

        rows.append(row)

    if len(rows) == 0:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Monitor] weight delta diagnostics saved to: {csv_path}")


def _data_loader_kwargs(args, drop_last):
    kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_mem),
        "drop_last": bool(drop_last),
    }
    if int(args.num_workers) > 0:
        prefetch = int(getattr(args, "loader_prefetch_factor", 2))
        if prefetch > 0:
            kwargs["prefetch_factor"] = prefetch
    return kwargs


def _make_train_eval_loader(args, dataset_train):
    """
    训练集 eval loader：和训练 loader 分开，避免 RandomSampler / drop_last 影响诊断结果。
    """
    sampler_train_eval = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train_eval = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train_eval,
        batch_size=int(args.batch_size),
        **_data_loader_kwargs(args, drop_last=False),
    )
    return data_loader_train_eval


def _make_formal_train_loader(args, dataset_train, sampler_train):
    return torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        **_data_loader_kwargs(args, drop_last=True),
    )


def _make_module_c_preflight_loaders(args, dataset_train, dataset_val):
    if dataset_val is None:
        raise RuntimeError("Automatic Module C requires a validation dataset.")
    support_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=torch.utils.data.SequentialSampler(dataset_train),
        batch_size=int(args.batch_size),
        **_data_loader_kwargs(args, drop_last=False),
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=int(args.batch_size),
        **_data_loader_kwargs(args, drop_last=False),
    )
    return support_loader, validation_loader


def _ensure_module_c_preflight_has_no_resume(args):
    resume_checkpoint = utils.resolve_resume_checkpoint(args)
    if resume_checkpoint is not None:
        raise RuntimeError(
            "Automatic Module C topology recovery from a resume checkpoint is unsupported "
            f"({resume_checkpoint}). Use a new output directory and no resume checkpoint."
        )


def _ensure_module_c_preflight_is_single_process():
    world_size = int(utils.get_world_size())
    if world_size != 1:
        raise RuntimeError(
            "Automatic Module C topology selection is single-process only (world_size=1); "
            "distributed rank synchronization is not implemented."
        )


def _build_training_schedules(args, num_training_steps_per_epoch):
    if args.lr_schedule_type == 'cosine':
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
    elif args.lr_schedule_type in ['constant', 'plateau']:
        lr_schedule_values = None
        print(f"[LR] Using {args.lr_schedule_type} LR schedule. Initial optimizer LR is kept from create_optimizer().")
    else:
        raise ValueError(f"Unknown lr_schedule_type={args.lr_schedule_type}")
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    return lr_schedule_values, wd_schedule_values


def _select_metric(stats, metric_name):
    """
    优先取 args.best_metric；如果没有，就回退到 accuracy。
    """
    if stats is None:
        return None
    if metric_name in stats:
        return stats[metric_name]
    if "accuracy" in stats:
        return stats["accuracy"]
    return None


def _add_selection_metrics(stats, details, args):
    """Add composite validation-selection scores into stats in-place.

    The generic scores combine balanced accuracy with dynamically computed
    worst-class recall and recall stability. Fixed class-0/2 scores remain
    available as explicit legacy selectors; ordinary metrics are unchanged.
    """
    if stats is None:
        return stats
    bacc = stats.get("balanced_accuracy", None)
    if bacc is None:
        return stats
    worst = stats.get("worst_class_recall", None)
    recall_std = stats.get("recall_std", 0.0)
    per_class = None
    if details is not None:
        per_class = details.get("per_class_recall", None)
    c0 = c2 = None
    if per_class is not None and len(per_class) > 0:
        try:
            c0 = float(per_class[0]) if not np.isnan(per_class[0]) else 0.0
        except Exception:
            c0 = 0.0
        try:
            c2 = float(per_class[2]) if len(per_class) > 2 and not np.isnan(per_class[2]) else 0.0
        except Exception:
            c2 = 0.0
    if worst is None:
        worst = 0.0
    try:
        recall_std = float(recall_std)
    except Exception:
        recall_std = 0.0
    min02 = min(c0 if c0 is not None else 0.0, c2 if c2 is not None else 0.0)
    a_worst = float(getattr(args, "selection_worst_alpha", 0.25))
    a_min02 = float(getattr(args, "selection_min02_alpha", 0.25))
    g_std = float(getattr(args, "selection_std_gamma", 0.10))
    hardmix_worst = float(getattr(args, "selection_hardmix_worst_alpha", 0.30))
    hardmix_min02 = float(getattr(args, "selection_hardmix_min02_alpha", 0.35))
    hardmix_std = float(getattr(args, "selection_hardmix_std_gamma", 0.18))
    hardmix_imbalance = float(getattr(args, "selection_hardmix_imbalance_gamma", 0.10))
    hardmix_floor = float(getattr(args, "selection_hardmix_floor", 0.08))
    hardmix_floor_gamma = float(getattr(args, "selection_hardmix_floor_gamma", 0.25))
    hard_imbalance = abs(float(c0 if c0 is not None else 0.0) - float(c2 if c2 is not None else 0.0))
    hard_floor_penalty = hardmix_floor_gamma * max(0.0, hardmix_floor - float(min02))

    stats["selection_bacc_worst"] = float(bacc) + a_worst * float(worst)
    stats["selection_bacc_min02"] = float(bacc) + a_min02 * float(min02)
    stats["selection_bacc_worst_std"] = float(bacc) + a_worst * float(worst) - g_std * recall_std
    stats["selection_bacc_min02_std"] = float(bacc) + a_min02 * float(min02) - g_std * recall_std
    stats["selection_bacc_hardmix_std"] = (
        float(bacc)
        + hardmix_worst * float(worst)
        + hardmix_min02 * float(min02)
        - hardmix_std * float(recall_std)
        - hardmix_imbalance * float(hard_imbalance)
        - float(hard_floor_penalty)
    )
    stats["selection_class0"] = float(c0 if c0 is not None else 0.0)
    stats["selection_class2"] = float(c2 if c2 is not None else 0.0)
    stats["selection_min02"] = float(min02)
    stats["selection_imbalance02"] = float(hard_imbalance)
    stats["selection_hardmix_floor_penalty"] = float(hard_floor_penalty)
    return stats


def _corresponding_test_metric_for_selection(test_stats, best_metric):
    """When selecting by a composite val score, report corresponding test BAcc."""
    if test_stats is None:
        return None
    if str(best_metric).startswith("selection_"):
        return test_stats.get("balanced_accuracy", None)
    return _select_metric(test_stats, best_metric)



def _safe_float(x):
    try:
        if x is None or x == "":
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _snapshot_metric_column(metric_name: str) -> str:
    metric_name = str(metric_name or '').strip()
    if metric_name.startswith('val_') or metric_name.startswith('test_') or metric_name.startswith('train_eval_'):
        return metric_name
    return f'val_{metric_name}'


def _read_snapshot_candidates(metrics_csv, select_metric):
    """Read epoch_metrics.csv and rank epochs by a validation metric."""
    col = _snapshot_metric_column(select_metric)
    rows = []
    if not os.path.exists(metrics_csv):
        print(f"[Snapshot] metrics CSV not found: {metrics_csv}")
        return rows, col
    with open(metrics_csv, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            score = _safe_float(r.get(col, None))
            epoch = r.get('epoch', r.get('epoch_id', None))
            try:
                epoch = int(float(epoch))
            except Exception:
                continue
            if score is None:
                continue
            rows.append({'epoch': epoch, 'score': score, 'row': dict(r)})
    rows.sort(key=lambda x: x['score'], reverse=True)
    return rows, col


def _classification_details_from_logits(logits_np, true_np, nb_classes):
    true_label = np.asarray(true_np).reshape(-1).astype(np.int64)
    pred_label = np.argmax(np.asarray(logits_np), axis=1).astype(np.int64)
    n_classes = int(nb_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true_label, pred_label):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            confusion[t, p] += 1
    per_class_recall = []
    for c in range(n_classes):
        denom = confusion[c, :].sum()
        per_class_recall.append(float('nan') if denom == 0 else float(confusion[c, c] / denom))
    per_class_recall = np.asarray(per_class_recall, dtype=np.float64)
    valid = per_class_recall[~np.isnan(per_class_recall)]
    worst = float(np.min(valid)) if len(valid) else float('nan')
    std = float(np.std(valid)) if len(valid) else float('nan')
    pred_count = np.bincount(pred_label, minlength=n_classes).astype(np.float64)
    pred_prob = pred_count / max(pred_count.sum(), 1.0)
    nz = pred_prob[pred_prob > 0]
    entropy = float(-(nz * np.log(nz + 1e-12)).sum())
    return {
        'confusion_matrix': confusion,
        'per_class_recall': per_class_recall,
        'worst_class_recall': worst,
        'recall_std': std,
        'pred_entropy': entropy,
        'y_true': true_label,
        'y_pred': pred_label,
    }


@torch.no_grad()
def _collect_logits_and_targets(args, data_loader, model, device, logit_bias=None):
    """Collect dataset-level logits for snapshot logit ensemble."""
    model.eval()
    logits_list, target_list = [], []
    for batch in data_loader:
        EEG = batch[0]
        target = batch[-1]
        if args.norm_method == 'mv':
            EEG = EEG.float().to(device, non_blocking=True) * args.mv_norm_value
        else:
            EEG = EEG.float().to(device, non_blocking=True)
        output = model(EEG)
        if logit_bias is not None and args.task_mod == 'Classification' and args.nb_classes > 1:
            output = output + logit_bias.to(device=output.device, dtype=output.dtype).view(1, -1)
        logits_list.append(output.detach().cpu())
        target_list.append(target.detach().cpu())
    return torch.cat(logits_list, dim=0), torch.cat(target_list, dim=0)


def _merge_prefixed_scalar_stats(dst, prefix, src):
    """Merge scalar stats into dst with a prefix, avoiding arrays/dicts."""
    if dst is None or src is None:
        return dst
    for k, v in src.items():
        if isinstance(v, (list, tuple, dict)):
            continue
        if hasattr(v, "shape"):
            continue
        try:
            if isinstance(v, (float, int, np.floating, np.integer, str, bool)) or v is None:
                dst[f"{prefix}_{k}"] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        except Exception:
            pass
    return dst


@torch.no_grad()
def _evaluate_logit_prototype_fusion(args, support_loader, query_loader, model, device, metrics, header='Proto-Test:'):
    """
    Lightweight rapid-calibration diagnostic in logit space.

    We do not assume access to penultimate features across all EEGFM wrappers.
    Instead, we use the model's own class-logit vector as a compact representation,
    build one prototype per class from the support set, then fuse prototype similarity
    back into the model logits.

    This is deliberately diagnostic: it answers whether a few-shot support readout
    can recover class-balanced decisions beyond standard logits/logit calibration.
    """
    if support_loader is None or query_loader is None:
        return None, None

    support_logits, support_y = _collect_logits_and_targets(args, support_loader, model, device, logit_bias=None)
    query_logits, query_y = _collect_logits_and_targets(args, query_loader, model, device, logit_bias=None)

    n_classes = int(args.nb_classes)
    support_y = support_y.long()
    query_y = query_y.long()

    support_emb = support_logits.float()
    query_emb = query_logits.float()

    metric = str(getattr(args, 'proto_metric', 'cosine')).lower()
    if metric == 'cosine':
        support_emb = torch.nn.functional.normalize(support_emb, dim=1, eps=1e-8)
        query_emb = torch.nn.functional.normalize(query_emb, dim=1, eps=1e-8)

    proto = torch.zeros(n_classes, support_emb.shape[1], dtype=support_emb.dtype)
    counts = torch.zeros(n_classes, dtype=support_emb.dtype)
    valid = (support_y >= 0) & (support_y < n_classes)
    if valid.any():
        proto.index_add_(0, support_y[valid], support_emb[valid])
        counts.index_add_(0, support_y[valid], torch.ones_like(support_y[valid], dtype=support_emb.dtype))
    proto = proto / counts.clamp_min(1.0).view(-1, 1)

    if metric == 'cosine':
        proto = torch.nn.functional.normalize(proto, dim=1, eps=1e-8)
        sim = torch.matmul(query_emb, proto.t())
    else:
        # negative squared Euclidean distance
        q2 = (query_emb ** 2).sum(dim=1, keepdim=True)
        p2 = (proto ** 2).sum(dim=1).view(1, -1)
        sim = -(q2 + p2 - 2.0 * torch.matmul(query_emb, proto.t()))

    missing = counts <= 0
    if missing.any():
        sim[:, missing] = -1e4

    alpha = float(getattr(args, 'proto_alpha', 2.0))
    fused_logits = query_logits.float() + alpha * sim.float()

    fused_np = fused_logits.numpy()
    true_np = query_y.numpy()
    stats = utils.get_metrics(fused_np, true_np, metrics, False, 0.5)
    details = _classification_details_from_logits(fused_np, true_np, n_classes)

    stats['worst_class_recall'] = details['worst_class_recall']
    stats['recall_std'] = details['recall_std']
    stats['pred_entropy'] = details['pred_entropy']
    stats['proto_alpha'] = alpha
    stats['proto_metric'] = metric
    stats['proto_support_size'] = int(support_y.numel())
    stats['proto_missing_classes'] = int(missing.sum().item())

    per_class = details.get('per_class_recall', None)
    if per_class is not None:
        for i, v in enumerate(per_class):
            stats[f'class{i}_recall'] = float(v) if not np.isnan(v) else float('nan')

    print(
        f"[ProtoEval] {header} source={getattr(args, 'proto_source', 'train_eval')} "
        f"metric={metric} alpha={alpha} "
        f"BAcc={stats.get('balanced_accuracy', float('nan')) * 100:.2f}% "
        f"Acc={stats.get('accuracy', float('nan')) * 100:.2f}% "
        f"missing_classes={stats['proto_missing_classes']}"
    )
    if per_class is not None:
        print(f"[ProtoEval] per_class_recall={per_class.tolist()}")

    return stats, details


def _load_epoch_checkpoint_into_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"[Snapshot] load {os.path.basename(ckpt_path)} missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[Snapshot] load {os.path.basename(ckpt_path)} unexpected keys: {len(unexpected)}")


def _write_snapshot_candidates(diag_dir, ranked_rows, metric_col):
    out = os.path.join(diag_dir, 'snapshot_candidates.csv')
    os.makedirs(diag_dir, exist_ok=True)
    fieldnames = ['rank', 'epoch', 'select_metric_col', 'select_score',
                  'val_balanced_accuracy', 'val_selection_bacc_worst_std',
                  'val_selection_bacc_min02_std',
                  'val_worst_class_recall', 'val_selection_min02', 'val_recall_std',
                  'test_balanced_accuracy', 'test_accuracy']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(ranked_rows, start=1):
            r = item['row']
            writer.writerow({
                'rank': i,
                'epoch': item['epoch'],
                'select_metric_col': metric_col,
                'select_score': item['score'],
                'val_balanced_accuracy': r.get('val_balanced_accuracy', ''),
                'val_selection_bacc_worst_std': r.get('val_selection_bacc_worst_std', ''),
                'val_selection_bacc_min02_std': r.get('val_selection_bacc_min02_std', ''),
                'val_worst_class_recall': r.get('val_worst_class_recall', ''),
                'val_selection_min02': r.get('val_selection_min02', ''),
                'val_recall_std': r.get('val_recall_std', ''),
                'test_balanced_accuracy': r.get('test_balanced_accuracy', ''),
                'test_accuracy': r.get('test_accuracy', ''),
            })
    print(f"[Snapshot] candidates saved to: {out}")


def _run_snapshot_ensemble_report(args, model, data_loader_val, data_loader_test, device, metrics):
    """Post-training top-k validation-selected snapshot logit ensemble.

    To reduce few-shot checkpoint instability, rank saved epoch snapshots by a
    validation/composite metric and average their test logits.
    """
    if args.task_mod != 'Classification' or int(args.nb_classes) <= 1:
        print('[Snapshot] skip: only multiclass classification is supported for now.')
        return
    if data_loader_val is None or data_loader_test is None:
        print('[Snapshot] skip: val/test loader unavailable.')
        return

    diag_dir = os.path.join(args.output_dir, 'diagnostics')
    ckpt_dir = os.path.join(args.output_dir, 'monitor_checkpoints')
    metrics_csv = os.path.join(diag_dir, 'epoch_metrics.csv')
    select_metric = getattr(args, 'snapshot_select_metric', '') or getattr(args, 'best_metric', 'balanced_accuracy')
    ranked, metric_col = _read_snapshot_candidates(metrics_csv, select_metric)
    if not ranked:
        print(f"[Snapshot] no ranked candidates found for metric={metric_col}")
        return
    _write_snapshot_candidates(diag_dir, ranked, metric_col)

    topk = max(1, int(getattr(args, 'snapshot_topk', 3)))
    selected = []
    for item in ranked:
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{item['epoch']:03d}.pth")
        if os.path.exists(ckpt_path):
            selected.append({**item, 'ckpt_path': ckpt_path})
        else:
            print(f"[Snapshot] missing checkpoint for epoch {item['epoch']}: {ckpt_path}")
        if len(selected) >= topk:
            break
    if not selected:
        print('[Snapshot] no selected checkpoints exist. Use --snapshot_eval or --save_epoch_ckpt_freq 1.')
        return

    # Save current final state so later operations are not surprised.
    final_state = _get_model_cpu_state(model)

    rows = []
    for use_k in ([1, len(selected)] if getattr(args, 'snapshot_include_top1', True) and len(selected) > 1 else [len(selected)]):
        use_items = selected[:use_k]
        logits_accum = None
        true_ref = None
        used_epochs = []
        used_scores = []
        for item in use_items:
            _load_epoch_checkpoint_into_model(model, item['ckpt_path'])
            logit_bias = None
            if getattr(args, 'eval_logit_adjust', False):
                val_stats_tmp, val_details_tmp = evaluate(
                    args, data_loader_val, model, device,
                    header=f"Snapshot-Val-E{item['epoch']}:",
                    metrics=metrics, return_details=True
                )
                _add_selection_metrics(val_stats_tmp, val_details_tmp, args)
                logit_bias = build_logit_adjust_bias(
                    val_details_tmp, nb_classes=args.nb_classes,
                    strength=args.logit_adjust_strength,
                    clip=args.logit_adjust_clip,
                )
            logits, true = _collect_logits_and_targets(args, data_loader_test, model, device, logit_bias=logit_bias)
            logits_accum = logits if logits_accum is None else logits_accum + logits
            if true_ref is None:
                true_ref = true
            used_epochs.append(str(item['epoch']))
            used_scores.append(f"{item['score']:.8f}")
        avg_logits = (logits_accum / float(use_k)).numpy()
        true_np = true_ref.numpy()
        stats = utils.get_metrics(avg_logits, true_np, metrics, False, 0.5)
        details = _classification_details_from_logits(avg_logits, true_np, args.nb_classes)
        stats['worst_class_recall'] = details['worst_class_recall']
        stats['recall_std'] = details['recall_std']
        stats['pred_entropy'] = details['pred_entropy']

        row = {
            'mode': f'top{use_k}_logit_ensemble' if use_k > 1 else 'top1_snapshot',
            'select_metric_col': metric_col,
            'selected_epochs': ';'.join(used_epochs),
            'selected_scores': ';'.join(used_scores),
        }
        for k, v in stats.items():
            if isinstance(v, (float, int, np.generic)):
                row[k] = _to_builtin_scalar(v)
        for i, v in enumerate(details['per_class_recall']):
            row[f'class_{i}'] = '' if np.isnan(v) else float(v)
        rows.append(row)

        # Windows has a practical path-length limit in many Python/Numpy calls.
        # The output_dir for these experiments is already long, so using verbose
        # filenames such as snapshot_confusion_top3_logit_ensemble.npy can fail
        # with FileNotFoundError even when diagnostics/ exists. Keep this file
        # name deliberately short.
        os.makedirs(diag_dir, exist_ok=True)
        short_mode = 't1' if use_k == 1 else f't{use_k}'
        cm_path = os.path.join(diag_dir, f"cm_{short_mode}.npy")
        try:
            np.save(cm_path, details['confusion_matrix'])
        except OSError as e:
            print(f"[Snapshot] warning: failed to save confusion matrix to {cm_path}: {e}")
        print(f"[Snapshot] {row['mode']} epochs={row['selected_epochs']} BAcc={row.get('balanced_accuracy', float('nan')) * 100:.2f}%")

    out_csv = os.path.join(diag_dir, 'snapshot_ensemble.csv')
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Snapshot] ensemble report saved to: {out_csv}")

    model.load_state_dict(final_state, strict=False)



def _details_class_recall(details, idx):
    if details is None:
        return ''
    pcr = details.get('per_class_recall', None)
    if pcr is None or len(pcr) <= idx:
        return ''
    try:
        v = float(pcr[idx])
        return '' if np.isnan(v) else v
    except Exception:
        return ''


def _write_boundary_anchor_row(args, row):
    diag_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    csv_path = os.path.join(diag_dir, "boundary_anchor.csv")
    _append_csv_row(csv_path, row)



def _read_boundary_trace(args):
    trace_csv = os.path.join(args.output_dir, "diagnostics", "boundary_anchor_trace.csv")
    rows = []
    if not os.path.exists(trace_csv):
        return rows
    with open(trace_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _float_or_none(x):
    try:
        if x is None or x == '':
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def _choose_boundary_anchor_from_trace(args):
    """Choose an anchor epoch from validation-only trace rows.

    The previous max-score anchor was too easy to overfit to TUEV validation.
    V2 supports anti-late-overfit strategies:
      - earliest_top: choose the earliest epoch that reaches a fixed fraction of
        the best validation boundary score.
      - epoch_penalty: choose max(score - lambda * late_epoch_penalty).
      - window_best: choose max(score) only inside [min_epoch, max_epoch].
      - window_balanced: choose max(score - lambda * abs(val_class0 - val_class2)) inside the window.
    """
    rows = _read_boundary_trace(args)
    cand = []
    min_ep = int(getattr(args, 'boundary_anchor_min_epoch', 1))
    max_ep = int(getattr(args, 'boundary_anchor_max_epoch', -1))
    for r in rows:
        ep = _float_or_none(r.get('epoch'))
        sc = _float_or_none(r.get('score'))
        if ep is None or sc is None:
            continue
        ep = int(ep)
        if ep < min_ep:
            continue
        if max_ep > 0 and ep > max_ep:
            continue
        rr = dict(r)
        rr['_epoch'] = ep
        rr['_score'] = float(sc)
        cand.append(rr)
    if not cand:
        return None, 'no_trace_candidates', None

    strategy = str(getattr(args, 'boundary_anchor_strategy', 'best') or 'best')
    if strategy == 'earliest_top':
        best = max(c['_score'] for c in cand)
        ratio = float(getattr(args, 'boundary_anchor_top_ratio', 0.85))
        threshold = best * ratio
        eligible = [c for c in cand if c['_score'] >= threshold]
        chosen = min(eligible, key=lambda c: c['_epoch'])
        reason = f"earliest_top: best={best:.6f}, ratio={ratio:.3f}, threshold={threshold:.6f}"
        return chosen['_epoch'], reason, chosen

    if strategy == 'epoch_penalty':
        penalty = float(getattr(args, 'boundary_anchor_epoch_penalty', 0.02))
        for c in cand:
            c['_adjusted_score'] = c['_score'] - penalty * max(0, c['_epoch'] - min_ep)
        chosen = max(cand, key=lambda c: c['_adjusted_score'])
        reason = (
            f"epoch_penalty: raw={chosen['_score']:.6f}, penalty={penalty:.6f}, "
            f"adjusted={chosen['_adjusted_score']:.6f}, min_epoch={min_ep}"
        )
        return chosen['_epoch'], reason, chosen

    if strategy == 'window_balanced':
        lam = float(getattr(args, 'boundary_anchor_balance_lambda', 0.25))
        for c in cand:
            c0 = _float_or_none(c.get('val_class0'))
            c2 = _float_or_none(c.get('val_class2'))
            if c0 is None or c2 is None:
                imb = 0.0
            else:
                imb = abs(float(c0) - float(c2))
            c['_imbalance02'] = imb
            c['_adjusted_score'] = c['_score'] - lam * imb
        chosen = max(cand, key=lambda c: c['_adjusted_score'])
        max_ep = int(getattr(args, 'boundary_anchor_max_epoch', -1))
        reason = (
            f"window_balanced: raw={chosen['_score']:.6f}, "
            f"imbalance02={chosen.get('_imbalance02', 0.0):.6f}, "
            f"lambda={lam:.6f}, adjusted={chosen['_adjusted_score']:.6f}, "
            f"min_epoch={min_ep}, max_epoch={max_ep}"
        )
        return chosen['_epoch'], reason, chosen

    if strategy == 'window_best':
        chosen = max(cand, key=lambda c: c['_score'])
        max_ep = int(getattr(args, 'boundary_anchor_max_epoch', -1))
        reason = (
            f"window_best: raw={chosen['_score']:.6f}, "
            f"min_epoch={min_ep}, max_epoch={max_ep}"
        )
        return chosen['_epoch'], reason, chosen

    chosen = max(cand, key=lambda c: c['_score'])
    max_ep = int(getattr(args, 'boundary_anchor_max_epoch', -1))
    if max_ep > 0:
        return chosen['_epoch'], f'best_validation_score_in_window: min_epoch={min_ep}, max_epoch={max_ep}', chosen
    return chosen['_epoch'], 'best_validation_score', chosen


def _maybe_update_boundary_anchor(args, model, epoch_id, val_stats, val_details, test_stats, test_details,
                                  current_best_score, current_best_epoch):
    """Log every eligible boundary score and optionally save max-score anchor.

    V1 only saved the max validation score. That failed for BIOT because the
    highest validation-boundary checkpoint could be badly overfitted to val.
    V2 always writes a full trace so final selection can use earliest_top or
    epoch_penalty without looking at test.
    """
    if not getattr(args, 'boundary_anchor_eval', False):
        return current_best_score, current_best_epoch
    if val_stats is None or int(epoch_id) < int(getattr(args, 'boundary_anchor_min_epoch', 1)):
        return current_best_score, current_best_epoch

    metric_name = str(getattr(args, 'boundary_anchor_metric', 'selection_bacc_worst_std') or 'selection_bacc_worst_std')
    score = _select_metric(val_stats, metric_name)
    if score is None:
        print(f"[BoundaryAnchor] metric not found: {metric_name}")
        return current_best_score, current_best_epoch

    score = float(score)
    row = {
        'event': 'trace',
        'epoch': int(epoch_id),
        'metric': metric_name,
        'score': score,
        'strategy': getattr(args, 'boundary_anchor_strategy', 'best'),
        'candidate_min_epoch': getattr(args, 'boundary_anchor_min_epoch', ''),
        'candidate_max_epoch': getattr(args, 'boundary_anchor_max_epoch', ''),
        'balance_lambda': getattr(args, 'boundary_anchor_balance_lambda', ''),
        'val_bacc': val_stats.get('balanced_accuracy', ''),
        'val_acc': val_stats.get('accuracy', ''),
        'val_worst': val_stats.get('worst_class_recall', ''),
        'val_recall_std': val_stats.get('recall_std', ''),
        'val_class0': _details_class_recall(val_details, 0),
        'val_class2': _details_class_recall(val_details, 2),
        'val_class5': _details_class_recall(val_details, 5),
        # test values are diagnostic only and are never used for anchor selection
        'test_bacc_diag': '' if test_stats is None else test_stats.get('balanced_accuracy', ''),
        'test_acc_diag': '' if test_stats is None else test_stats.get('accuracy', ''),
        'test_class0_diag': _details_class_recall(test_details, 0),
        'test_class2_diag': _details_class_recall(test_details, 2),
        'test_class5_diag': _details_class_recall(test_details, 5),
    }
    diag_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    _append_csv_row(os.path.join(diag_dir, "boundary_anchor_trace.csv"), row)

    # Keep the old max-score checkpoint too, for comparison/backward compatibility.
    if score > float(current_best_score):
        tag = str(getattr(args, 'boundary_anchor_tag', 'boundary_anchor') or 'boundary_anchor')
        _save_monitor_checkpoint(args=args, model=model, epoch=int(epoch_id), tag=tag)
        update_row = dict(row)
        update_row['event'] = 'max_update'
        update_row['previous_best'] = '' if current_best_epoch is None else float(current_best_score)
        update_row['previous_epoch'] = '' if current_best_epoch is None else int(current_best_epoch)
        update_row['checkpoint_tag'] = tag
        _write_boundary_anchor_row(args, update_row)
        print(
            f"[BoundaryAnchor] max-update epoch={epoch_id}, metric={metric_name}, "
            f"score={score:.6f}, diag_test_bacc={row['test_bacc_diag']}"
        )
        return score, int(epoch_id)

    return current_best_score, current_best_epoch


def _load_anchor_epoch_state(args, epoch):
    ckpt_dir = os.path.join(args.output_dir, "monitor_checkpoints")
    epoch_path = os.path.join(ckpt_dir, f"epoch_{int(epoch):03d}.pth")
    if os.path.exists(epoch_path):
        obj = torch.load(epoch_path, map_location='cpu')
        return obj, epoch_path
    # fallback to max-update tag if needed
    tag = str(getattr(args, 'boundary_anchor_tag', 'boundary_anchor') or 'boundary_anchor')
    tag_path = os.path.join(ckpt_dir, f"{tag}.pth")
    if os.path.exists(tag_path):
        obj = torch.load(tag_path, map_location='cpu')
        return obj, tag_path
    raise FileNotFoundError(f"No anchor checkpoint found for epoch={epoch}; tried {epoch_path} and {tag_path}")


def _run_boundary_anchor_final_eval(args, model, data_loader_val, data_loader_test, device, metrics):
    """Select an anchor from validation trace and evaluate it as a formal candidate."""
    if not getattr(args, 'boundary_anchor_eval', False):
        return
    if data_loader_val is None or data_loader_test is None:
        return

    chosen_epoch, reason, chosen_row = _choose_boundary_anchor_from_trace(args)
    if chosen_epoch is None:
        print(f"[BoundaryAnchor] no anchor selected: {reason}")
        return

    final_state = _get_model_cpu_state(model)
    obj, ckpt_path = _load_anchor_epoch_state(args, chosen_epoch)
    state = obj.get('model', obj)
    epoch = obj.get('epoch', chosen_epoch)
    model.load_state_dict(state, strict=False)

    val_stats, val_details = evaluate(
        args, data_loader_val, model, device,
        header=f"BoundaryAnchor-Val-E{epoch}:",
        metrics=metrics, return_details=True
    )
    test_stats, test_details = evaluate(
        args, data_loader_test, model, device,
        header=f"BoundaryAnchor-Test-E{epoch}:",
        metrics=metrics, return_details=True
    )
    _add_selection_metrics(val_stats, val_details, args)
    _add_selection_metrics(test_stats, test_details, args)

    # Save the actually selected anchor under a stable name.
    tag = str(getattr(args, 'boundary_anchor_tag', 'boundary_anchor') or 'boundary_anchor')
    selected_tag = f"{tag}_selected"
    _save_monitor_checkpoint(args=args, model=model, epoch=int(epoch), tag=selected_tag)

    diag_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    row = {
        'mode': 'boundary_anchor_v2',
        'anchor_epoch': epoch,
        'anchor_metric': getattr(args, 'boundary_anchor_metric', 'selection_bacc_worst_std') or 'selection_bacc_worst_std',
        'anchor_strategy': getattr(args, 'boundary_anchor_strategy', 'best'),
        'anchor_reason': reason,
        'anchor_source_ckpt': ckpt_path,
        'anchor_trace_score': '' if chosen_row is None else chosen_row.get('score', ''),
        'selected_checkpoint_tag': selected_tag,
    }
    _flatten_eval_row('val', val_stats, row)
    _flatten_eval_row('test', test_stats, row)
    _add_per_class_to_row('val', val_details, row)
    _add_per_class_to_row('test', test_details, row)
    out_csv = os.path.join(diag_dir, 'boundary_anchor_eval.csv')
    fieldnames = list(row.keys())
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(_json_safe_dict(row))
    print(f"[BoundaryAnchor] final eval saved to: {out_csv}")
    print(
        f"[BoundaryAnchor] strategy={row['anchor_strategy']}, epoch={epoch}, "
        f"test BAcc={row.get('test_balanced_accuracy', float('nan'))}"
    )

    model.load_state_dict(final_state, strict=False)



def _parse_float_list(text, default_values):
    if text is None or str(text).strip() == '':
        return list(default_values)
    vals = []
    for part in str(text).replace(';', ',').split(','):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals if vals else list(default_values)


def _find_existing_output_dir_for_adapter_eval(args):
    if getattr(args, 'adapter_eval_input_dir', ''):
        p = os.path.abspath(args.adapter_eval_input_dir)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"adapter_eval_input_dir does not exist: {p}")
        return p

    tag = str(getattr(args, 'adapter_eval_tag', '')).strip()
    if not tag:
        raise ValueError('adapter_calib_eval requires --adapter_eval_tag or --adapter_eval_input_dir')

    root = os.path.join(
        'finetuning_results', args.task_mod,
        f'{args.model_name}_results', f'finetune_{args.finetune_mod}'
    )
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Cannot find result root: {root}")
    candidates = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and tag in name:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No output directory under {root} contains tag={tag}")
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    print(f"[AdapterCalib] using existing output_dir: {candidates[0]}")
    return candidates[0]


def _metric_col_for_adapter_eval(metric_name):
    metric = str(metric_name or 'balanced_accuracy')
    return metric if metric.startswith('val_') else f'val_{metric}'


def _rank_adapter_candidate_epochs(metrics_csv, metric_name, epoch_min, epoch_max, topn):
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Missing epoch_metrics.csv: {metrics_csv}")
    metric_col = _metric_col_for_adapter_eval(metric_name)
    rows = []
    with open(metrics_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(float(row.get('epoch', row.get('epoch_id', '0'))))
            except Exception:
                continue
            if ep < int(epoch_min):
                continue
            if int(epoch_max) > 0 and ep > int(epoch_max):
                continue
            val = row.get(metric_col, '')
            if val in ('', None):
                continue
            try:
                score = float(val)
            except Exception:
                continue
            rows.append({'epoch': ep, 'score': score, 'row': row})
    rows.sort(key=lambda x: x['score'], reverse=True)
    if topn and int(topn) > 0:
        rows = rows[:int(topn)]
    print(f"[AdapterCalib] ranked {len(rows)} candidate epochs by {metric_col}")
    return rows, metric_col




def _should_apply_cbra_eval_front_beta(args, monitor_init_state):
    return (
        str(getattr(args, 'model_name', '')).lower() == 'cbramod'
        and monitor_init_state is not None
        and abs(float(getattr(args, 'cbra_eval_front_beta', 1.0)) - 1.0) > 1e-12
    )


def _apply_cbra_front_beta_state(model, init_state, beta):
    if init_state is None:
        return
    beta = float(beta)
    cur_state = _get_model_cpu_state(model)
    mixed = {}
    for k, v in cur_state.items():
        if (
            isinstance(v, torch.Tensor)
            and 'main_model.patch_embedding' in k
            and k in init_state
            and isinstance(init_state[k], torch.Tensor)
            and init_state[k].shape == v.shape
            and torch.is_floating_point(v)
        ):
            mv = init_state[k].float() + beta * (v.float() - init_state[k].float())
            mixed[k] = mv.to(dtype=v.dtype)
        else:
            mixed[k] = v
    model.load_state_dict(mixed, strict=False)
    print(f"[CBra-CFI] standard evaluation uses front beta={beta:g}")


def _adapter_swa_update(model, swa_state, swa_count, trainable_names, trainable_only=True):
    state = model.state_dict()
    if swa_state is None:
        swa_state = {}
        swa_count = 0
    new_count = int(swa_count) + 1
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or not torch.is_floating_point(v):
            continue
        if trainable_only and k not in trainable_names:
            continue
        val = v.detach().cpu().float()
        if k not in swa_state:
            swa_state[k] = val.clone()
        else:
            swa_state[k].mul_(float(swa_count) / float(new_count)).add_(val / float(new_count))
    return swa_state, new_count


def _apply_partial_float_state(model, partial_state):
    if not partial_state:
        return
    cur = _get_model_cpu_state(model)
    for k, v in partial_state.items():
        if k in cur and isinstance(cur[k], torch.Tensor) and cur[k].shape == v.shape:
            cur[k] = v.to(dtype=cur[k].dtype)
    model.load_state_dict(cur, strict=False)


def _rank_filter_lora_state(partial_state, filter_rank):
    """Return a copy of partial_state with LoRA effective deltas rank-filtered.

    For a LoRA pair A/B, forward uses B @ A. We compute SVD(B@A), keep
    top-k directions, and refactor the filtered delta back into A/B with the
    original tensor shapes. This is intentionally used only for evaluation of
    averaged/SWA LoRA states, not during training.
    """
    k = int(filter_rank)
    if k <= 0 or not partial_state:
        return partial_state

    out = {name: (v.clone() if isinstance(v, torch.Tensor) else v) for name, v in partial_state.items()}
    done = 0
    for akey, aval in list(partial_state.items()):
        if not isinstance(aval, torch.Tensor) or 'lora_A' not in akey:
            continue
        bkey = akey.replace('lora_A', 'lora_B')
        if bkey not in partial_state or not isinstance(partial_state[bkey], torch.Tensor):
            continue
        A0 = partial_state[akey].detach().float().cpu()
        B0 = partial_state[bkey].detach().float().cpu()
        orig_a_shape = tuple(A0.shape)
        orig_b_shape = tuple(B0.shape)
        conv_1x1 = False
        if A0.ndim == 3 and B0.ndim == 3 and A0.shape[-1] == 1 and B0.shape[-1] == 1:
            conv_1x1 = True
            A = A0.squeeze(-1)
            B = B0.squeeze(-1)
        elif A0.ndim == 2 and B0.ndim == 2:
            A = A0
            B = B0
        else:
            continue
        if A.shape[0] != B.shape[1]:
            continue
        r = int(A.shape[0])
        kk = max(1, min(k, r, A.shape[1], B.shape[0]))
        try:
            delta = B @ A
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        except Exception as e:
            print(f'[Rank-SWA] skip {akey}: {e}')
            continue
        sqrt_s = torch.sqrt(S[:kk].clamp_min(0))
        B_new = torch.zeros_like(B)
        A_new = torch.zeros_like(A)
        B_new[:, :kk] = U[:, :kk] * sqrt_s.reshape(1, -1)
        A_new[:kk, :] = sqrt_s.reshape(-1, 1) * Vh[:kk, :]
        if conv_1x1:
            B_new = B_new.reshape(orig_b_shape)
            A_new = A_new.reshape(orig_a_shape)
        out[akey] = A_new.to(dtype=partial_state[akey].dtype)
        out[bkey] = B_new.to(dtype=partial_state[bkey].dtype)
        done += 1
    print(f'[Rank-SWA] applied rank-filter rank={k} to {done} LoRA pairs')
    return out



def _load_epoch_state_with_front_beta(model, ckpt_path, init_state_path=None, front_beta=1.0):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    trained_state = ckpt.get('model', ckpt)
    if init_state_path and os.path.exists(init_state_path) and abs(float(front_beta) - 1.0) > 1e-12:
        init_obj = torch.load(init_state_path, map_location='cpu')
        init_state = init_obj.get('model', init_obj)
        beta = float(front_beta)
        mixed_state = {}
        for k, v in trained_state.items():
            if (
                isinstance(v, torch.Tensor)
                and 'main_model.patch_embedding' in k
                and k in init_state
                and isinstance(init_state[k], torch.Tensor)
                and init_state[k].shape == v.shape
                and torch.is_floating_point(v)
            ):
                mixed_state[k] = init_state[k].float() + beta * (v.float() - init_state[k].float())
                mixed_state[k] = mixed_state[k].to(dtype=v.dtype)
            else:
                mixed_state[k] = v
        state = mixed_state
    else:
        state = trained_state
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        print(f"[AdapterCalib] load {os.path.basename(ckpt_path)} missing keys: {len(missing)}")
    if len(unexpected) > 0:
        print(f"[AdapterCalib] load {os.path.basename(ckpt_path)} unexpected keys: {len(unexpected)}")


def _flatten_eval_row(prefix, stats, row):
    if stats is None:
        return
    for k, v in stats.items():
        if isinstance(v, (float, int, np.generic)):
            row[f'{prefix}_{k}'] = _to_builtin_scalar(v)


def _add_per_class_to_row(prefix, details, row):
    if details is None:
        return
    pcr = details.get('per_class_recall', None)
    if pcr is not None:
        for i, v in enumerate(pcr):
            row[f'{prefix}_class_{i}'] = '' if np.isnan(v) else float(v)


def _run_adapter_strength_calibration(args, model, data_loader_val, data_loader_test, device, metrics):
    """Offline adapter strength calibration on saved epoch checkpoints.

    It searches an existing output_dir, loads monitor_checkpoints/epoch_XXX.pth,
    sweeps LoRA runtime scale and optional CBraMod patch/front-end interpolation
    beta, chooses by validation BAcc, and reports the corresponding test result.
    """
    if args.task_mod != 'Classification' or int(args.nb_classes) <= 1:
        raise RuntimeError('adapter_calib_eval currently supports multiclass classification only.')
    if data_loader_val is None or data_loader_test is None:
        raise RuntimeError('adapter_calib_eval requires validation and test loaders.')

    old_output_dir = _find_existing_output_dir_for_adapter_eval(args)
    diag_dir = os.path.join(old_output_dir, 'diagnostics')
    ckpt_dir = os.path.join(old_output_dir, 'monitor_checkpoints')
    metrics_csv = os.path.join(diag_dir, 'epoch_metrics.csv')
    init_state_path = os.path.join(diag_dir, 'init_model_state.pth')

    epoch_max = int(getattr(args, 'adapter_eval_epoch_max', -1))
    if epoch_max <= 0:
        epoch_max = int(args.epochs)
    candidates, metric_col = _rank_adapter_candidate_epochs(
        metrics_csv=metrics_csv,
        metric_name=getattr(args, 'adapter_eval_metric', 'balanced_accuracy'),
        epoch_min=int(getattr(args, 'adapter_eval_epoch_min', 1)),
        epoch_max=epoch_max,
        topn=int(getattr(args, 'adapter_eval_topn', 8)),
    )

    lora_scales = _parse_float_list(getattr(args, 'adapter_lora_scales', ''), [1.0])
    front_betas = _parse_float_list(getattr(args, 'adapter_front_betas', ''), [1.0])
    if args.model_name != 'CBraMod':
        front_betas = [1.0]

    if getattr(args, 'adapter_calib_strengths', ''):
        calib_strengths = _parse_float_list(args.adapter_calib_strengths, [float(getattr(args, 'logit_adjust_strength', 1.0))])
    elif getattr(args, 'eval_logit_adjust', False):
        calib_strengths = [float(getattr(args, 'logit_adjust_strength', 1.0))]
    else:
        calib_strengths = [None]

    collect_dir = str(getattr(args, 'adapter_eval_collect_dir', '')).strip()
    if not collect_dir:
        collect_dir = os.path.join(old_output_dir, 'adapter_strength_calibration')
    os.makedirs(collect_dir, exist_ok=True)
    exp = str(getattr(args, 'adapter_eval_exp_name', '') or getattr(args, 'adapter_eval_tag', '') or args.model_name)
    safe_exp = re.sub(r'[^A-Za-z0-9_.-]+', '_', exp).strip('._-') or args.model_name

    rows = []
    best_item = None
    final_state = _get_model_cpu_state(model)

    for cand in candidates:
        ep = int(cand['epoch'])
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{ep:03d}.pth')
        if not os.path.exists(ckpt_path):
            print(f"[AdapterCalib] missing checkpoint epoch {ep}: {ckpt_path}")
            continue
        for beta in front_betas:
            _load_epoch_state_with_front_beta(model, ckpt_path, init_state_path=init_state_path, front_beta=beta)
            for scale in lora_scales:
                set_lora_runtime_scale(model, scale, verbose=False)
                val_stats_raw, val_details_raw = evaluate(
                    args, data_loader_val, model, device,
                    header=f'Adapter-Val-E{ep}-s{scale:g}-b{beta:g}:',
                    metrics=metrics, return_details=True
                )
                _add_selection_metrics(val_stats_raw, val_details_raw, args)

                for strength in calib_strengths:
                    logit_bias = None
                    val_stats_for_select = val_stats_raw
                    val_details_for_select = val_details_raw
                    test_header = f'Adapter-Test-E{ep}-s{scale:g}-b{beta:g}'
                    if strength is not None:
                        logit_bias = build_logit_adjust_bias(
                            val_details_raw, nb_classes=args.nb_classes,
                            strength=float(strength), clip=args.logit_adjust_clip,
                        )
                        test_header += f'-cal{float(strength):g}'

                    test_stats, test_details = evaluate(
                        args, data_loader_test, model, device,
                        header=test_header + ':', metrics=metrics,
                        return_details=True, logit_bias=logit_bias,
                    )
                    _add_selection_metrics(test_stats, test_details, args)

                    row = {
                        'epoch': ep,
                        'epoch_rank_metric_col': metric_col,
                        'epoch_rank_score': cand['score'],
                        'lora_scale': float(scale),
                        'front_beta': float(beta),
                        'calib_strength': '' if strength is None else float(strength),
                        'source_output_dir': old_output_dir,
                    }
                    _flatten_eval_row('val', val_stats_raw, row)
                    _flatten_eval_row('test', test_stats, row)
                    _add_per_class_to_row('val', val_details_raw, row)
                    _add_per_class_to_row('test', test_details, row)
                    rows.append(row)

                    val_score = float(val_stats_raw.get('balanced_accuracy', val_stats_raw.get('accuracy', 0.0)))
                    if best_item is None or val_score > best_item['select_score']:
                        best_item = {'select_score': val_score, 'row': row}

    if not rows:
        raise RuntimeError('adapter_calib_eval produced no rows; check checkpoints and tags.')

    # Write all rows.
    out_csv = os.path.join(collect_dir, f'{safe_exp}_adapter_scale_sweep.csv')
    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write selected row only.
    selected_csv = os.path.join(collect_dir, f'{safe_exp}_adapter_scale_selected.csv')
    with open(selected_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(best_item['row'])

    br = best_item['row']
    print(f"[AdapterCalib] saved sweep: {out_csv}")
    print(f"[AdapterCalib] saved selected: {selected_csv}")
    print(
        f"[AdapterCalib] selected epoch={br.get('epoch')} scale={br.get('lora_scale')} "
        f"beta={br.get('front_beta')} cal={br.get('calib_strength')} "
        f"val_BAcc={float(br.get('val_balanced_accuracy', float('nan'))) * 100:.2f}% "
        f"test_BAcc={float(br.get('test_balanced_accuracy', float('nan'))) * 100:.2f}%"
    )

    model.load_state_dict(final_state, strict=False)


def _add_prefixed_scalars(row, prefix, stats):
    """
    把 stats 中的标量加到 row 里，自动加 prefix。
    """
    if stats is None:
        return

    for k, v in stats.items():
        if isinstance(v, (list, tuple, dict, np.ndarray)):
            continue
        row[f"{prefix}_{k}"] = _to_builtin_scalar(v)


def _save_split_details(args, epoch, split, details):
    """
    保存 per-class recall 和 confusion matrix。
    """
    if details is None or len(details) == 0:
        return

    diag_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    per_class_path = os.path.join(diag_dir, "per_class_recall.csv")
    _append_per_class_recall(
        csv_path=per_class_path,
        epoch=epoch,
        split=split,
        per_class_recall=details.get("per_class_recall", None)
    )

    confusion = details.get("confusion_matrix", None)
    if confusion is not None:
        cm_path = os.path.join(diag_dir, f"cm_{split}_e{epoch:03d}.npy")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        np.save(cm_path, confusion)


def _json_safe_dict(d):
    return {k: _to_builtin_scalar(v) for k, v in d.items()}




def _extract_label_from_item(item):
    """Best-effort label extraction from one dataset item."""
    if isinstance(item, dict):
        for key in ["label", "labels", "target", "targets", "y"]:
            if key in item:
                item = item[key]
                break
    elif isinstance(item, (tuple, list)):
        item = item[-1]

    if torch.is_tensor(item):
        item = item.detach().cpu()
        if item.numel() == 1:
            return int(item.item())
        return int(item.view(-1)[0].item())
    if isinstance(item, np.ndarray):
        if item.size == 1:
            return int(item.item())
        return int(item.reshape(-1)[0])
    return int(item)


def _labels_from_dataset_fast(dataset):
    """Try common label attributes before falling back to iterating the dataset."""
    # torch.utils.data.Subset support
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_labels = _labels_from_dataset_fast(dataset.dataset)
        if base_labels is not None:
            return [int(base_labels[int(i)]) for i in list(dataset.indices)]

    for attr in ["targets", "labels", "y", "label", "data_labels", "all_labels"]:
        if hasattr(dataset, attr):
            labels = getattr(dataset, attr)
            try:
                if torch.is_tensor(labels):
                    labels = labels.detach().cpu().numpy()
                labels = np.asarray(labels).reshape(-1).tolist()
                if len(labels) == len(dataset):
                    return [int(x) for x in labels]
            except Exception:
                pass
    return None


def _compute_class_weights_from_dataset(dataset, nb_classes, loss_type, clip_min=0.2, clip_max=5.0):
    """Compute normalized inverse-frequency weights from the current training split.

    We normalize weights so the mean over present classes is approximately 1.0.
    sqrt_balanced_ce is intentionally softer than balanced_ce and should be tried first.
    """
    labels = _labels_from_dataset_fast(dataset)
    if labels is None:
        labels = []
        for i in range(len(dataset)):
            labels.append(_extract_label_from_item(dataset[i]))

    labels = [int(x) for x in labels if int(x) >= 0]
    counts = np.zeros(int(nb_classes), dtype=np.float64)
    for y in labels:
        if 0 <= y < nb_classes:
            counts[y] += 1.0

    safe_counts = counts.copy()
    safe_counts[safe_counts <= 0] = 1.0

    if loss_type == 'balanced_ce':
        weights = safe_counts.sum() / (float(nb_classes) * safe_counts)
    elif loss_type in ['sqrt_balanced_ce', 'soft_sqrt_balanced_ce', 'anneal_sqrt_balanced_ce']:
        weights = np.sqrt(safe_counts.sum() / (float(nb_classes) * safe_counts))
    else:
        weights = np.ones(int(nb_classes), dtype=np.float64)

    # absent classes should not explode; keep them at 0 if they truly do not appear in the train split
    weights[counts <= 0] = 0.0

    # normalize present weights to mean 1, then clip for safety
    present = counts > 0
    if present.any():
        weights[present] = weights[present] / max(weights[present].mean(), 1e-12)
    weights = np.clip(weights, float(clip_min), float(clip_max))
    weights[counts <= 0] = 0.0

    print(f"[Loss] train class counts: {counts.astype(int).tolist()}")
    print(f"[Loss] {loss_type} weights: {weights.tolist()}")
    return weights.astype(np.float32).tolist(), counts.astype(int).tolist()


def _get_optimizer_lr_stats(optimizer):
    lrs = [float(g.get('lr', 0.0)) for g in optimizer.param_groups]
    if not lrs:
        return 0.0, 0.0
    return min(lrs), max(lrs)


def _reduce_optimizer_lr_on_plateau(optimizer, factor, min_lr):
    old_min, old_max = _get_optimizer_lr_stats(optimizer)
    for group in optimizer.param_groups:
        old_lr = float(group.get('lr', 0.0))
        group['lr'] = max(old_lr * float(factor), float(min_lr))
    new_min, new_max = _get_optimizer_lr_stats(optimizer)
    print(f"[PlateauLR] reduce lr: min {old_min:.8g} -> {new_min:.8g}, max {old_max:.8g} -> {new_max:.8g}")
    return old_min, old_max, new_min, new_max


# -------------------------------------------------------------------------------------------------------------

# -------------------------------Main function for fine-tuning-------------------------------------------------
def resolve_output_root(args):
    """Resolve the base result directory before the unique run tag is appended."""

    configured_root = str(getattr(args, 'output_dir', '') or '').strip()
    if configured_root:
        return os.path.normpath(os.path.expanduser(configured_root))
    return os.path.join(
        "finetuning_results",
        args.task_mod,
        f"{args.model_name}_results",
        f"finetune_{args.finetune_mod}",
    )


def _configure_cudnn_runtime():
    """Use stable algorithms for variable-shape EEG loaders and preflight."""
    cudnn.benchmark = False


def main(args, ds_init):

    if ds_init is not None:
        utils.create_ds_config(args)

    args.save_ckpt_freq = args.epochs

    # FB2: resolve framework switches before output tag/model/optimizer are built.
    args = resolve_functional_args(args)
    args.module_e_mode = module_e_mode_from_args(args)

    # Keep result folder names short enough for Windows path length limits.
    # The previous verbose naming could exceed MAX_PATH once diagnostic .npy files were saved.
    output_root = resolve_output_root(args)

    short_tag = (
        f"{args.dataset}_e{args.epochs}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr:g}"
        f"_{args.norm_method}"
        f"_s{args.seed}"
    )

    if args.finetune_mod == 'lora':
        base_short = {
            'full': 'bf',
            'freeze': 'bz',
            'tiny': 'bt',
        }.get(args.lora_base_update, str(args.lora_base_update))

        target_short = {
            'none': 'none',
            'qv': 'qv',
            'qkv': 'qkv',
            'qkvo': 'qkvo',
            'ffn': 'ffn',
            'mlp': 'mlp',
            'qv_ffn': 'qvffn',
            'qkvo_ffn': 'qkvoffn',
            'attn_ffn': 'attnffn',
            'all_linear': 'all',
            'ffn_late': 'ffnlate',
            'ffn_last2': 'ffnlast2',
            'spatial_attn': 'spattn',
            'temporal_attn': 'tmpattn',
            'bridge': 'bridge',
            'input_bridge': 'bridge',
            'front': 'front',
            'bridge_ffn': 'bridgeffn',
        }.get(args.lora_target, str(args.lora_target))

        short_tag += (
            f"_{base_short}_{target_short}"
            f"_r{args.lora_rank}"
            f"_a{args.lora_alpha:g}"
            f"_d{args.lora_dropout:g}"
        )
        if args.lora_train_chan_conv:
            short_tag += "_cc"
        if getattr(args, 'module_b_sites', 'both') != 'both':
            short_tag += f"_bsite{getattr(args, 'module_b_sites')}"
        if getattr(args, 'cbra_train_patch_embed_when_frozen', False):
            short_tag += "_cbpatchtrain"
        if getattr(args, 'cbra_freeze_patch_embed_in_full', False):
            short_tag += "_cbpatchfreeze"

    loss_short = {
        'ce': 'ce',
        'balanced_ce': 'bce',
        'sqrt_balanced_ce': 'sqbce',
        'soft_sqrt_balanced_ce': 'softsq',
        'anneal_sqrt_balanced_ce': 'ansq',
    }.get(args.loss_type, str(args.loss_type))

    if args.loss_type != 'ce':
        short_tag += f"_{loss_short}"
        if args.loss_type in ['soft_sqrt_balanced_ce', 'anneal_sqrt_balanced_ce']:
            short_tag += f"_l{args.class_balance_lambda:g}"
        if args.loss_type == 'anneal_sqrt_balanced_ce':
            short_tag += (
                f"_a{args.class_balance_anneal_start_epoch}"
                f"-{args.class_balance_anneal_end_epoch}"
            )
            if float(args.class_balance_anneal_floor) != 0.0:
                short_tag += f"_f{args.class_balance_anneal_floor:g}"

    if args.finetune_mod == 'lora' and args.lora_grad_decay_after_epoch > 0 and args.lora_grad_decay_factor != 1.0:
        short_tag += (
            f"_ld{args.lora_grad_decay_after_epoch}"
            f"x{args.lora_grad_decay_factor:g}"
        )

    if args.finetune_mod == 'lora' and float(getattr(args, 'lora_delta_lambda', 0.0)) > 0.0:
        short_tag += f"_loraCtrl{args.lora_delta_lambda:g}"

    if str(getattr(args, 'best_metric', 'balanced_accuracy')).startswith('selection_'):
        bm_safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(args.best_metric)).strip('._-')
        short_tag += f"_sel-{bm_safe}"

    if getattr(args, 'eval_logit_adjust', False):
        short_tag += f"_cal{args.logit_adjust_strength:g}"

    if getattr(args, 'adaptive_swa_eval', False):
        short_tag += "_aswa"

    if getattr(args, 'fb_enable', False) and not getattr(args, 'short_output_tag_only', False):
        fb_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(getattr(args, 'fb_recipe', 'profile'))).strip('._-').lower()
        if fb_tag and fb_tag != 'none':
            short_tag += f"_fb{fb_tag}"

    if args.lr_schedule_type != 'cosine':
        short_tag += f"_sch{args.lr_schedule_type}"
        if args.lr_schedule_type == 'plateau':
            short_tag += (
                f"_p{args.plateau_patience}"
                f"_f{args.plateau_factor:g}"
            )

    safe_run_tag = ''
    if getattr(args, 'run_tag', ''):
        safe_run_tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(args.run_tag)).strip('._-')
        if safe_run_tag:
            short_tag += f"_{safe_run_tag}"

    if getattr(args, 'short_output_tag_only', False):
        if not safe_run_tag:
            raise ValueError('--short_output_tag_only requires a non-empty --run_tag')
        short_tag = safe_run_tag

    args.output_dir = os.path.join(output_root, short_tag)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    _configure_cudnn_runtime()

    # Classification/regression use the validated task registry. Retrieval has
    # its own EEGDataset registry and therefore does not require Retrieval.json.
    dataset_info = load_task_dataset_info(args.task_mod, args.dataset)
    if args.task_mod == 'Retrieval':
        os.environ["WANDB_API_KEY"] = ""
        os.environ["WANDB_MODE"] = 'offline'
        dataset_train = EEGDataset(args.dataset, train=True, subject_mod=args.subject_mod, subject_id=args.subject_id, sampling_rate=args.sampling_rate, norm_method=args.norm_method)
        dataset_test = EEGDataset(args.dataset, train=False, subject_mod=args.subject_mod, subject_id=args.subject_id, sampling_rate=args.sampling_rate, norm_method=args.norm_method)
        dataset_val = None
        ch_names = dataset_train.get_ch_names()
    else:
        dataset_train, dataset_test, dataset_val, ch_names = get_datasets(args, dataset_info)

    save_split_integrity(
        args=args,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        dataset_info=dataset_info,
    )

    # ----------------------------Loss weights for imbalanced classification.--------------------------------
    args.class_weights = None
    args.class_counts = None
    if args.task_mod == 'Classification' and args.nb_classes > 1 and args.loss_type in [
        'balanced_ce', 'sqrt_balanced_ce', 'soft_sqrt_balanced_ce', 'anneal_sqrt_balanced_ce'
    ]:
        args.class_weights, args.class_counts = _compute_class_weights_from_dataset(
            dataset_train,
            nb_classes=args.nb_classes,
            loss_type=args.loss_type,
            clip_min=args.class_weight_clip_min,
            clip_max=args.class_weight_clip_max,
        )

    # ----------------------------Get dataloaders.--------------------------------
    if args.disable_eval_during_finetuning:
        dataset_val = None
        dataset_test = None

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    lr_schedule_values, wd_schedule_values = _build_training_schedules(
        args, num_training_steps_per_epoch
    )

    # if True:  # args.distributed:
    global_rank = 0
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            if type(dataset_test) == list:
                sampler_test = [torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True) for dataset in dataset_test]
            else:
                sampler_test = torch.utils.data.DistributedSampler(
                    dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val is not None else None
            sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)



    data_loader_train = _make_formal_train_loader(
        args, dataset_train, sampler_train
    )

    if args.monitor_dynamics and args.eval_train_set and args.task_mod != 'Retrieval':
        data_loader_train_eval = _make_train_eval_loader(args, dataset_train)
    else:
        data_loader_train_eval = None

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(args.batch_size),
            **_data_loader_kwargs(args, drop_last=False),
        )
    else:
        data_loader_val = None
    
    if dataset_test is not None:
        if type(dataset_test) == list:
            data_loader_test = [torch.utils.data.DataLoader(
                dataset, sampler=sampler,
                batch_size=int(args.batch_size),
                **_data_loader_kwargs(args, drop_last=False),
            ) for dataset, sampler in zip(dataset_test, sampler_test)]
        else:
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(args.batch_size),
                **_data_loader_kwargs(args, drop_last=False),
            )
    else:
        data_loader_test = None
    # ------------------------------------------------------------------------------------------

    module_c_preflight_ran = False
    if module_c_preflight_requested(args):
        _ensure_module_c_preflight_is_single_process()
        _ensure_module_c_preflight_has_no_resume(args)
        print("[ModuleC] running exhaustive matched one-pass search before formal LoRA training.")
        module_c_support_loader, module_c_validation_loader = _make_module_c_preflight_loaders(
            args, dataset_train, dataset_val
        )
        preflight_rng_state = capture_module_c_rng_state(
            (module_c_support_loader, module_c_validation_loader)
        )
        probe_model = None
        try:
            probe_model = _build_module_c_preflight_probe_model(args, ch_names, dataset_info["num_t"])
            probe_model.to(device)
            run_module_c_preflight_selection(
                args=args,
                model=probe_model,
                data_loader_train=module_c_support_loader,
                data_loader_val=module_c_validation_loader,
                device=device,
                criterion_builder=_build_module_e_pressure_criterion,
                is_main_process=utils.is_main_process(),
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
            )
            module_c_preflight_ran = True
        finally:
            if probe_model is not None:
                del probe_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            restore_module_c_rng_state(preflight_rng_state)

    write_resolved_recipe(args, args.output_dir)
    print(args)

    if bool(getattr(args, "module_c_preflight_only", False)):
        if not module_c_preflight_ran:
            raise RuntimeError("--module_c_preflight_only requested, but automatic Module C search did not run.")
        print("[ModuleC] preflight-only verification completed; formal training was not started.")
        return

    # load the model
    model = get_models(args, ch_names, dataset_info['num_t'])
    model.to(device)
    # model_ema = None
    model_without_ddp = model

    module_e_controller = _attach_requested_module_e_controller(args, model_without_ddp)

    if getattr(args, 'model_name', None) == 'CBraMod' and float(getattr(args, 'cbra_l2sp_lambda', 0.0)) > 0.0:
        _register_cbra_l2sp_reference(model_without_ddp, args, verbose=True)

    if getattr(args, 'model_name', None) == 'CBraMod':
        _register_cbra_grad_scale_hooks(model_without_ddp, args, verbose=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    save_block_registry(args, model_without_ddp)
    save_module_e_coverage_audit(args, model_without_ddp)
    save_module_e_lora_injection_audit(args, model_without_ddp)

    if getattr(args, 'adapter_calib_eval', False):
        if args.task_mod == 'Regression':
            adapter_metrics = ["Pearson_Correlation", 'R2_Score', 'RMSE']
        elif args.nb_classes > 1:
            adapter_metrics = ["accuracy", 'balanced_accuracy', 'f1_weighted', 'cohen_kappa']
        else:
            adapter_metrics = ["accuracy", 'balanced_accuracy', 'pr_auc', 'roc_auc']
        _run_adapter_strength_calibration(
            args=args,
            model=model_without_ddp,
            data_loader_val=data_loader_val,
            data_loader_test=data_loader_test,
            device=device,
            metrics=adapter_metrics,
        )
        return

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    # layer_decay for labram and cbramod
    # if args.layer_decay < 1.0:
    if args.model_name in ['LaBraM', 'CBraMod']:
        if args.model_name == 'LaBraM':
            num_layers = model_without_ddp.main_model.get_num_layers()
        elif args.model_name == 'CBraMod':
            num_layers = len(model_without_ddp.main_model.encoder.layers)
        else:
            print("Layer_decay is not supported by the model. ")
            exit(0)
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = []

    # get optimizer, lr_scheduler...
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        optimizer = _create_formal_optimizer(
            args,
            model_without_ddp,
            skip_weight_decay_list=skip_weight_decay_list,
            assigner=assigner,
            controller=module_e_controller,
        )
        loss_scaler = NativeScaler()

    # load checkpoint for resume
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    # ---------------------------monitor init state--------------------------------
    monitor_init_state = None
    monitor_trainable_names = set()

    if (args.monitor_dynamics or getattr(args, 'adaptive_swa_eval', False)) and args.task_mod != 'Retrieval' and utils.is_main_process():
        monitor_init_state = _get_model_cpu_state(model_without_ddp)
        monitor_trainable_names = _get_trainable_param_names(model_without_ddp)

        monitor_dir = os.path.join(args.output_dir, "diagnostics")
        os.makedirs(monitor_dir, exist_ok=True)

        init_path = os.path.join(monitor_dir, "init_model_state.pth")
        torch.save({
            "model": monitor_init_state,
            "trainable_names": sorted(list(monitor_trainable_names)),
            "args": vars(args),
        }, init_path)

        print(f"[Monitor] init model state saved to: {init_path}")
        print(f"[Monitor] trainable params: {len(monitor_trainable_names)}")
    # --------------------------------------------------------------------------------

    adapter_swa_state = None
    adapter_swa_count = 0
    adaptive_swa_snapshots = []

    # start finetuning
    print(f"Start training for {args.epochs} epochs")

    if args.task_mod == 'Retrieval':
        current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
        img_features_train_all = dataset_train.img_features
        img_features_test_all = dataset_test.img_features
        results = main_train_loop(
            args, current_time, model, data_loader_train, data_loader_test, optimizer, device, 
            img_features_train_all, img_features_test_all, config=args, loss_scaler=loss_scaler, 
            logger=args.logger, lr_schedule_values=lr_schedule_values, ch_names=ch_names,
            wd_schedule_values=wd_schedule_values, num_training_steps_per_epoch=num_training_steps_per_epoch)
        
        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, current_time)
        os.makedirs(results_dir, exist_ok=True)

        results_file = f"{results_dir}/results.csv"
        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')
    else:
        start_time = time.time()
        max_accuracy = 0.0
        max_accuracy_test = 0.0
        max_r2 = 0.0
        max_r2_test = 0.0

        plateau_best = -float('inf')
        plateau_bad_epochs = 0
        plateau_reductions = 0

        boundary_anchor_best = -float('inf')
        boundary_anchor_epoch = None

        # metrics: list of strings, the metrics you want to use. We utilize PyHealth to implement it.
        if args.task_mod == 'Regression':
            metrics = ["Pearson_Correlation", 'R2_Score', 'RMSE']
        elif args.nb_classes > 1:
            metrics = ["accuracy", 'balanced_accuracy', 'f1_weighted', 'cohen_kappa']
        else:
            metrics = ["accuracy", 'balanced_accuracy', 'pr_auc', 'roc_auc']

        module_d_latest_eval_row = None

        for epoch in range(args.start_epoch, args.epochs):
            epoch_id = epoch + 1

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            # -------------------------------train one epoch--------------------------------
            if args.finetune_mod == 'lora':
                _set_lora_gradient_scale(model_without_ddp, args, epoch_id)
                apply_lora_lifecycle_controls(model_without_ddp, args, epoch_id)
            train_stats = train_one_epoch(
                args, model, data_loader_train, optimizer,
                device, epoch, loss_scaler,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                ch_names=ch_names
            )

            # -------------------------------original checkpoint--------------------------------
            if args.output_dir and args.save_ckpt:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    save_ckpt_freq=args.save_ckpt_freq
                )

            # -------------------------------periodic monitor checkpoint--------------------------------
            if (
                args.monitor_dynamics
                and utils.is_main_process()
                and args.save_epoch_ckpt_freq > 0
                and (
                    epoch_id == 1
                    or getattr(args, 'snapshot_eval', False)
                    or epoch_id % args.save_epoch_ckpt_freq == 0
                    or epoch_id == args.epochs
                )
            ):
                _save_monitor_checkpoint(
                    args=args,
                    model=model_without_ddp,
                    epoch=epoch_id,
                    tag=f"epoch_{epoch_id:03d}"
                )

            # -------------------------------adaptive SWA snapshot capture--------------------------------
            if (
                getattr(args, 'adaptive_swa_eval', False)
                and utils.is_main_process()
                and monitor_trainable_names is not None
            ):
                snap = capture_lifecycle_snapshot(
                    model_without_ddp,
                    epoch_id,
                    monitor_trainable_names,
                    trainable_only=bool(getattr(args, 'adaptive_swa_trainable_only', True)),
                )
                adaptive_swa_snapshots.append(snap)
                print(f"[Adaptive-SWA] captured trainable snapshot epoch {epoch_id}, tensors={len(snap['state'])}")

            # -------------------------------adapter SWA update--------------------------------
            if (
                getattr(args, 'adapter_swa', False)
                and epoch_id >= int(getattr(args, 'adapter_swa_start_epoch', 8))
                and (int(getattr(args, 'adapter_swa_end_epoch', -1)) <= 0 or epoch_id <= int(getattr(args, 'adapter_swa_end_epoch', -1)))
                and monitor_trainable_names is not None
            ):
                adapter_swa_state, adapter_swa_count = _adapter_swa_update(
                    model_without_ddp,
                    adapter_swa_state,
                    adapter_swa_count,
                    monitor_trainable_names,
                    trainable_only=bool(getattr(args, 'adapter_swa_trainable_only', True)),
                )
                print(f"[Adapter-SWA] epoch {epoch_id}: update count={adapter_swa_count}")

            # -------------------------------evaluation--------------------------------
            train_eval_stats, train_eval_details = None, None
            val_stats, val_details = None, None
            test_stats, test_details = None, None

            eval_restore_state = None
            eval_uses_temp_state = False
            if (
                getattr(args, 'adapter_swa_eval', False)
                and adapter_swa_state is not None
                and adapter_swa_count > 0
            ) or _should_apply_cbra_eval_front_beta(args, monitor_init_state):
                eval_restore_state = _get_model_cpu_state(model_without_ddp)
                if getattr(args, 'adapter_swa_eval', False) and adapter_swa_state is not None and adapter_swa_count > 0:
                    eval_swa_state = adapter_swa_state
                    if int(getattr(args, 'adapter_swa_filter_rank', -1)) > 0:
                        eval_swa_state = _rank_filter_lora_state(adapter_swa_state, int(getattr(args, 'adapter_swa_filter_rank', -1)))
                    _apply_partial_float_state(model_without_ddp, eval_swa_state)
                    print(f"[Adapter-SWA] standard evaluation uses averaged trainable params, count={adapter_swa_count}, filter_rank={getattr(args, 'adapter_swa_filter_rank', -1)}")
                if _should_apply_cbra_eval_front_beta(args, monitor_init_state):
                    _apply_cbra_front_beta_state(model_without_ddp, monitor_init_state, getattr(args, 'cbra_eval_front_beta', 1.0))
                eval_uses_temp_state = True

            if args.monitor_dynamics and args.eval_train_set and data_loader_train_eval is not None:
                train_eval_stats, train_eval_details = evaluate(
                    args,
                    data_loader_train_eval,
                    model,
                    device,
                    header='Train-Eval:',
                    ch_names=ch_names,
                    metrics=metrics,
                    return_details=True
                )

            if data_loader_val is not None:
                if args.monitor_dynamics or getattr(args, 'module_d_sbr_eval', False):
                    val_stats, val_details = evaluate(
                        args,
                        data_loader_val,
                        model,
                        device,
                        header='Val:',
                        ch_names=ch_names,
                        metrics=metrics,
                        return_details=True
                    )
                    test_logit_bias = None
                    test_header = 'Test:'
                    if args.eval_logit_adjust and args.task_mod == 'Classification' and val_details is not None:
                        test_logit_bias = build_logit_adjust_bias(
                            val_details=val_details,
                            nb_classes=args.nb_classes,
                            strength=args.logit_adjust_strength,
                            clip=args.logit_adjust_clip,
                        )
                        if test_logit_bias is not None:
                            test_header = 'Test-Calib:'

                    test_stats, test_details = evaluate(
                        args,
                        data_loader_test,
                        model,
                        device,
                        header=test_header,
                        ch_names=ch_names,
                        metrics=metrics,
                        return_details=True,
                        logit_bias=test_logit_bias
                    )
                    _add_selection_metrics(train_eval_stats, train_eval_details, args)
                    _add_selection_metrics(val_stats, val_details, args)
                    _add_selection_metrics(test_stats, test_details, args)

                    if (
                        getattr(args, 'proto_eval', False)
                        and args.task_mod == 'Classification'
                        and args.nb_classes > 1
                    ):
                        proto_source = str(getattr(args, 'proto_source', 'train_eval')).lower()
                        support_loader = data_loader_train_eval if proto_source == 'train_eval' else data_loader_val
                        proto_stats, proto_details = _evaluate_logit_prototype_fusion(
                            args=args,
                            support_loader=support_loader,
                            query_loader=data_loader_test,
                            model=model,
                            device=device,
                            metrics=metrics,
                            header='Proto-Test:',
                        )
                        _add_selection_metrics(proto_stats, proto_details, args)
                        _merge_prefixed_scalar_stats(test_stats, 'proto', proto_stats)

                    if args.task_mod == 'Classification':
                        module_d_latest_eval_row = module_d_eval_row_from_details(
                            val_stats=val_stats,
                            val_details=val_details,
                            test_stats=test_stats,
                            test_details=test_details,
                            source=f"epoch_{epoch_id:03d}",
                        )
                else:
                    val_stats = evaluate(
                        args,
                        data_loader_val,
                        model,
                        device,
                        header='Val:',
                        ch_names=ch_names,
                        metrics=metrics
                    )
                    test_stats = evaluate(
                        args,
                        data_loader_test,
                        model,
                        device,
                        header='Test:',
                        ch_names=ch_names,
                        metrics=metrics
                    )

                if eval_uses_temp_state and eval_restore_state is not None:
                    if getattr(args, 'save_eval_state_ckpt', False) and args.monitor_dynamics:
                        _save_monitor_checkpoint(
                            args=args,
                            model=model_without_ddp,
                            epoch=epoch_id,
                            tag=f"epoch_{epoch_id:03d}"
                        )
                        print(f"[EvalState] saved temporary eval state to monitor checkpoint epoch_{epoch_id:03d}.pth")
                    model_without_ddp.load_state_dict(eval_restore_state, strict=False)
                    print('[EvalState] restored current training weights after temporary eval state')

                # -------------------------------print main metrics--------------------------------
                if args.task_mod == 'Classification':
                    val_acc = val_stats.get("accuracy", float("nan"))
                    test_acc = test_stats.get("accuracy", float("nan"))
                    val_bacc = val_stats.get("balanced_accuracy", float("nan"))
                    test_bacc = test_stats.get("balanced_accuracy", float("nan"))

                    print(f"Accuracy on the val set: {val_acc * 100:.2f}%")
                    print(f"Accuracy on the test set: {test_acc * 100:.2f}%")
                    print(f"BAcc on the val set: {val_bacc * 100:.2f}%")
                    print(f"BAcc on the test set: {test_bacc * 100:.2f}%")
                else:
                    print(f"R2_Score on the val set: {val_stats['R2_Score']:.2f}")
                    print(f"R2_Score on the test set: {test_stats['R2_Score']:.2f}")

                # -------------------------------save best checkpoint--------------------------------
                if args.task_mod == 'Classification':
                    current_val_metric = _select_metric(val_stats, args.best_metric)
                    current_test_metric = _corresponding_test_metric_for_selection(test_stats, args.best_metric)

                    if current_val_metric is not None and max_accuracy < current_val_metric:
                        max_accuracy = current_val_metric

                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args,
                                model=model,
                                model_without_ddp=model_without_ddp,
                                optimizer=optimizer,
                                loss_scaler=loss_scaler,
                                epoch="best"
                            )

                        max_accuracy_test = current_test_metric

                    print(
                        f"Best metric [{args.best_metric}] val: {max_accuracy * 100:.2f} %, "
                        f"corresponding test: {max_accuracy_test * 100:.2f} %"
                    )

                    boundary_anchor_best, boundary_anchor_epoch = _maybe_update_boundary_anchor(
                        args=args,
                        model=model_without_ddp,
                        epoch_id=epoch_id,
                        val_stats=val_stats,
                        val_details=val_details,
                        test_stats=test_stats,
                        test_details=test_details,
                        current_best_score=boundary_anchor_best,
                        current_best_epoch=boundary_anchor_epoch,
                    )

                else:
                    if max_r2 < val_stats["R2_Score"]:
                        max_r2 = val_stats["R2_Score"]

                        if args.output_dir and args.save_ckpt:
                            utils.save_model(
                                args=args,
                                model=model,
                                model_without_ddp=model_without_ddp,
                                optimizer=optimizer,
                                loss_scaler=loss_scaler,
                                epoch="best"
                            )

                        max_r2_test = test_stats["R2_Score"]

                    print(f'Max R2_Score val: {max_r2:.2f}, max R2_Score test: {max_r2_test:.2f}')

                # -------------------------------plateau-aware LR--------------------------------
                if args.lr_schedule_type == 'plateau' and args.task_mod == 'Classification':
                    plateau_value = _select_metric(val_stats, args.plateau_metric)
                    if plateau_value is not None:
                        if epoch_id <= args.plateau_warmup_epochs:
                            print(f"[PlateauLR] warmup epoch {epoch_id}; skip plateau check.")
                        elif plateau_value > plateau_best + args.plateau_min_delta:
                            plateau_best = plateau_value
                            plateau_bad_epochs = 0
                            print(f"[PlateauLR] metric {args.plateau_metric} improved to {plateau_best * 100:.2f}%")
                        else:
                            plateau_bad_epochs += 1
                            print(
                                f"[PlateauLR] no improvement on {args.plateau_metric}: "
                                f"current={plateau_value * 100:.2f}%, best={plateau_best * 100:.2f}%, "
                                f"bad_epochs={plateau_bad_epochs}/{args.plateau_patience}"
                            )
                            if plateau_bad_epochs >= args.plateau_patience:
                                _reduce_optimizer_lr_on_plateau(
                                    optimizer,
                                    factor=args.plateau_factor,
                                    min_lr=args.plateau_min_lr,
                                )
                                plateau_reductions += 1
                                plateau_bad_epochs = 0

                # -------------------------------normal log_stats--------------------------------
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'val_{k}': v for k, v in val_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'epoch_id': epoch_id,
                    'n_parameters': n_parameters
                }

                if train_eval_stats is not None:
                    log_stats.update({f'train_eval_{k}': v for k, v in train_eval_stats.items()})

            else:
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch,
                    'epoch_id': epoch_id,
                    'n_parameters': n_parameters
                }

            # -------------------------------monitor scalar CSV--------------------------------
            if args.monitor_dynamics and utils.is_main_process():
                metrics_csv = os.path.join(args.output_dir, "diagnostics", "epoch_metrics.csv")

                lr_min_now, lr_max_now = _get_optimizer_lr_stats(optimizer)
                epoch_row = {
                    "epoch": epoch_id,
                    "n_parameters": n_parameters,
                    "best_metric": args.best_metric,
                    "loss_type": args.loss_type,
                    "lr_schedule_type": args.lr_schedule_type,
                    "lr_min_now": lr_min_now,
                    "lr_max_now": lr_max_now,
                    "plateau_bad_epochs": plateau_bad_epochs,
                    "plateau_reductions": plateau_reductions,
                    "class_balance_anneal_floor": getattr(args, "class_balance_anneal_floor", 0.0),
                    "eval_logit_adjust": int(bool(getattr(args, "eval_logit_adjust", False))),
                    "logit_adjust_strength": getattr(args, "logit_adjust_strength", 1.0),
                    "lora_delta_lambda": getattr(args, "lora_delta_lambda", 0.0),
                    "lora_delta_mode": getattr(args, "lora_delta_mode", "relative_l2"),
                    "selection_worst_alpha": getattr(args, "selection_worst_alpha", 0.25),
                    "selection_min02_alpha": getattr(args, "selection_min02_alpha", 0.25),
                    "selection_std_gamma": getattr(args, "selection_std_gamma", 0.10),
                    "selection_hardmix_worst_alpha": getattr(args, "selection_hardmix_worst_alpha", 0.30),
                    "selection_hardmix_min02_alpha": getattr(args, "selection_hardmix_min02_alpha", 0.35),
                    "selection_hardmix_std_gamma": getattr(args, "selection_hardmix_std_gamma", 0.18),
                    "selection_hardmix_imbalance_gamma": getattr(args, "selection_hardmix_imbalance_gamma", 0.10),
                    "selection_hardmix_floor": getattr(args, "selection_hardmix_floor", 0.08),
                    "selection_hardmix_floor_gamma": getattr(args, "selection_hardmix_floor_gamma", 0.25),
                }

                _add_prefixed_scalars(epoch_row, "train_loop", train_stats)
                _add_prefixed_scalars(epoch_row, "train_eval", train_eval_stats)
                _add_prefixed_scalars(epoch_row, "val", val_stats)
                _add_prefixed_scalars(epoch_row, "test", test_stats)

                # 过拟合 gap：优先看 balanced_accuracy，没有就退回 accuracy
                if train_eval_stats is not None and val_stats is not None:
                    train_eval_main = _select_metric(train_eval_stats, args.best_metric)
                    val_main = _select_metric(val_stats, args.best_metric)
                    if train_eval_main is not None and val_main is not None:
                        epoch_row[f"train_val_gap_{args.best_metric}"] = train_eval_main - val_main

                if train_eval_stats is not None and test_stats is not None:
                    train_eval_main = _select_metric(train_eval_stats, args.best_metric)
                    test_main = _select_metric(test_stats, args.best_metric)
                    if train_eval_main is not None and test_main is not None:
                        epoch_row[f"train_test_gap_{args.best_metric}"] = train_eval_main - test_main

                _append_csv_row(metrics_csv, epoch_row)

                # 保存 per-class recall / confusion matrix
                if (
                    epoch_id == 1
                    or epoch_id % args.diag_freq == 0
                    or epoch_id == args.epochs
                ):
                    _save_split_details(args, epoch_id, "train_eval", train_eval_details)
                    _save_split_details(args, epoch_id, "val", val_details)
                    _save_split_details(args, epoch_id, "test", test_details)

                # 保存权重 delta 谱分析
                if (
                    monitor_init_state is not None
                    and (
                        epoch_id == 1
                        or epoch_id % args.diag_freq == 0
                        or epoch_id == args.epochs
                    )
                ):
                    _run_weight_delta_diagnostics(
                        args=args,
                        model=model_without_ddp,
                        init_state=monitor_init_state,
                        trainable_names=monitor_trainable_names,
                        epoch=epoch_id
                    )
                    save_block_delta_summary(
                        args=args,
                        model=model_without_ddp,
                        init_state=monitor_init_state,
                        trainable_names=monitor_trainable_names,
                        epoch=epoch_id,
                        metrics_row=epoch_row,
                    )
                    save_module_e_structural_pressure_proxy(args=args)

            # -------------------------------original log.txt--------------------------------
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(_json_safe_dict(log_stats)) + "\n")

        if (
            getattr(args, 'snapshot_eval', False)
            and args.task_mod == 'Classification'
            and data_loader_val is not None
            and data_loader_test is not None
            and utils.is_main_process()
        ):
            _run_snapshot_ensemble_report(
                args=args,
                model=model_without_ddp,
                data_loader_val=data_loader_val,
                data_loader_test=data_loader_test,
                device=device,
                metrics=metrics,
            )

        if (
            getattr(args, 'boundary_anchor_eval', False)
            and args.task_mod == 'Classification'
            and data_loader_val is not None
            and data_loader_test is not None
            and utils.is_main_process()
        ):
            _run_boundary_anchor_final_eval(
                args=args,
                model=model_without_ddp,
                data_loader_val=data_loader_val,
                data_loader_test=data_loader_test,
                device=device,
                metrics=metrics,
            )

        if (
            getattr(args, 'adaptive_swa_eval', False)
            and args.task_mod == 'Classification'
            and data_loader_val is not None
            and data_loader_test is not None
            and utils.is_main_process()
        ):
            def _apply_module_a_eval_state_adjust(eval_model):
                if _should_apply_cbra_eval_front_beta(args, monitor_init_state):
                    _apply_cbra_front_beta_state(
                        eval_model,
                        monitor_init_state,
                        getattr(args, 'cbra_eval_front_beta', 1.0),
                    )

            lifecycle_selection_row = run_lifecycle_window_search(
                args=args,
                model=model_without_ddp,
                data_loader_val=data_loader_val,
                data_loader_test=data_loader_test,
                device=device,
                metrics=metrics,
                snapshots=adaptive_swa_snapshots,
                evaluate_fn=evaluate,
                build_logit_adjust_bias_fn=build_logit_adjust_bias,
                eval_state_adjust_fn=_apply_module_a_eval_state_adjust,
            )
        else:
            lifecycle_selection_row = None

        module_d_adapted_row = lifecycle_selection_row if lifecycle_selection_row is not None else module_d_latest_eval_row
        if args.task_mod == 'Classification' and utils.is_main_process():
            save_module_d_sbr_eval(args, adapted_row=module_d_adapted_row)

        run_signal_alignment_probe_after_training(
            args=args,
            model=model_without_ddp,
            data_loader_val=data_loader_val,
            device=device,
            lifecycle_selection_row=lifecycle_selection_row,
            is_main_process=utils.is_main_process(),
        )

        collect_outputs_if_requested(args)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    opts, ds_init = get_args()
    main(opts, ds_init)
