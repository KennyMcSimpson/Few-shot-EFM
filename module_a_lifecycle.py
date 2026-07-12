"""Module A: validation-guided lifecycle window selection.

The training script supplies the model, loaders, and evaluate function. This file
keeps the trajectory-window logic together: arguments, epoch snapshots, window
construction, validation-only selection, final test evaluation, and diagnostics.
"""

import csv
import os

import numpy as np
import torch


__all__ = [
    'add_lifecycle_window_args',
    'capture_lifecycle_snapshot',
    'run_lifecycle_window_search',
]


def add_lifecycle_window_args(parser):
    # Validation-based adaptive SWA window selection.
    # It stores trainable snapshots each epoch, searches candidate windows using validation only,
    # then evaluates the selected averaged adapter once on test.
    parser.add_argument('--adaptive_swa_eval', action='store_true', default=False,
                        help='After training, search SWA windows using validation only and evaluate the selected averaged state on test.')
    parser.add_argument('--adaptive_swa_epoch_min', default=1, type=int,
                        help='Minimum epoch allowed in adaptive SWA window search.')
    parser.add_argument('--adaptive_swa_epoch_max', default=-1, type=int,
                        help='Maximum epoch allowed in adaptive SWA window search. -1 means args.epochs.')
    parser.add_argument('--adaptive_swa_min_len', default=3, type=int,
                        help='Minimum continuous window length for adaptive SWA search.')
    parser.add_argument('--adaptive_swa_max_len', default=8, type=int,
                        help='Maximum continuous window length for adaptive SWA search.')
    parser.add_argument('--adaptive_swa_stride', default=1, type=int,
                        help='Stride for adaptive SWA candidate start/end epochs.')
    parser.add_argument('--adaptive_swa_select_metric', default='selection_bacc_worst_std', type=str,
                        help='Validation metric used to choose the SWA window. The generic default combines balanced accuracy, worst-class recall, and recall stability; use names without val_ prefix. Test is never used.')
    parser.add_argument('--adaptive_swa_balance_lambda', default=0.0, type=float,
                        help='Extra validation-only penalty on the recall spread among configured hard classes.')
    parser.add_argument('--adaptive_swa_hard_classes', default='', type=str,
                        help='Optional comma-separated hard-class ids used by hard-class penalties and diagnostics. Empty disables hard-class behavior.')
    parser.add_argument('--adaptive_swa_tie_eps', default=0.002, type=float,
                        help='If window scores are within this value, prefer shorter and earlier windows.')
    parser.add_argument('--adaptive_swa_trainable_only', action='store_true', default=True,
                        help='Average only trainable floating parameters captured at training init. Default True.')
    parser.add_argument('--adaptive_swa_filter_rank', default=-1, type=int,
                        help='Optional rank filter applied to each candidate averaged LoRA state before validation evaluation.')
    parser.add_argument('--adaptive_swa_save_selected_ckpt', action='store_true', default=True,
                        help='Save the validation-selected adaptive SWA state as monitor_checkpoints/adaptive_swa_selected.pth.')
    parser.add_argument('--adaptive_swa_no_save_selected_ckpt', action='store_false', dest='adaptive_swa_save_selected_ckpt',
                        help='Do not save the selected adaptive SWA state checkpoint.')

    parser.add_argument('--adaptive_swa_profile', default='generic', type=str,
                        choices=['generic', 'biot_life', 'labram_early', 'cbra_hard'],
                        help='Validation-only lifecycle window selector profile. generic is the default; other names are kept as legacy compatibility aliases. Use explicit prior args for ablation presets.')
    parser.add_argument('--adaptive_swa_life_center', default=-1.0, type=float,
                        help='Preferred window center epoch for lifecycle prior. -1 uses profile default.')
    parser.add_argument('--adaptive_swa_life_center_weight', default=-1.0, type=float,
                        help='Weight for lifecycle center prior. -1 uses profile default.')
    parser.add_argument('--adaptive_swa_early_weight', default=-1.0, type=float,
                        help='Weight for earlier-window prior. -1 uses profile default.')
    parser.add_argument('--adaptive_swa_len_weight', default=-1.0, type=float,
                        help='Weight for longer/shorter window prior. Positive prefers longer; negative prefers shorter. -1 uses profile default.')
    parser.add_argument('--adaptive_swa_std_lambda', default=-1.0, type=float,
                        help='Validation recall-std penalty for adaptive SWA. -1 uses profile default.')
    parser.add_argument('--adaptive_swa_hard_floor', default=-1.0, type=float,
                        help='Optional validation hard-class floor. It is ignored when no hard classes are configured; -1 disables it.')
    parser.add_argument('--adaptive_swa_hard_floor_lambda', default=-1.0, type=float,
                        help='Penalty strength for hard-class floor. -1 uses profile default.')
    parser.add_argument('--adaptive_swa_late_end', default=-1.0, type=float,
                        help='Optional late-window end epoch; windows ending after it are penalized. -1 disables unless profile default enables it.')
    parser.add_argument('--adaptive_swa_late_lambda', default=-1.0, type=float,
                        help='Penalty strength for late-window end. -1 uses profile default.')
    parser.add_argument('--adaptive_swa_tie_mode', default='auto', type=str,
                        choices=['auto', 'short_early', 'early_long', 'long_mid', 'hard_stable'],
                        help='Tie break for near-equal validation scores. auto uses profile-specific behavior.')


def _to_builtin_scalar(x):
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
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    clean_row = {k: _to_builtin_scalar(v) for k, v in row.items()}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(clean_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(clean_row)


def _get_model_cpu_state(model):
    state = {}
    for k, v in model.state_dict().items():
        if torch.is_tensor(v):
            state[k] = v.detach().cpu().clone()
    return state


def _select_metric(stats, metric_name):
    if stats is None:
        return None
    if metric_name in stats:
        return stats[metric_name]
    if 'accuracy' in stats:
        return stats['accuracy']
    return None


def _apply_partial_float_state(model, partial_state):
    if not partial_state:
        return
    cur = _get_model_cpu_state(model)
    for k, v in partial_state.items():
        if k in cur and isinstance(cur[k], torch.Tensor) and cur[k].shape == v.shape:
            cur[k] = v.to(dtype=cur[k].dtype)
    model.load_state_dict(cur, strict=False)


def _flatten_eval_row(prefix, stats, row):
    if stats is None:
        return
    for k, v in stats.items():
        row[f'{prefix}_{k}'] = v


def _add_per_class_to_row(prefix, details, row):
    if details is None:
        return
    pcr = details.get('per_class_recall', None)
    if pcr is not None:
        for i, v in enumerate(pcr):
            row[f'{prefix}_class_{i}'] = '' if np.isnan(v) else float(v)


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

def _parse_int_list(text, default=None):
    if default is None:
        default = []
    if text is None or str(text).strip() == '':
        return list(default)
    out = []
    for item in str(text).split(','):
        item = item.strip()
        if item == '':
            continue
        try:
            out.append(int(item))
        except Exception:
            pass
    return out if out else list(default)

def capture_lifecycle_snapshot(model, epoch_id, trainable_names, trainable_only=True):
    """Capture trainable floating tensors after an epoch for validation-only SWA window search."""
    state = model.state_dict()
    snap = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor) or not torch.is_floating_point(v):
            continue
        if trainable_only and k not in trainable_names:
            continue
        snap[k] = v.detach().cpu().float().clone()
    return {'epoch': int(epoch_id), 'state': snap}

def _average_adaptive_swa_window(snapshots, start_epoch, end_epoch):
    selected = [s for s in snapshots if int(start_epoch) <= int(s['epoch']) <= int(end_epoch)]
    if len(selected) == 0:
        return None, 0
    avg = {}
    count = 0
    for snap in selected:
        count += 1
        for k, v in snap['state'].items():
            if k not in avg:
                avg[k] = v.clone() / float(len(selected))
            else:
                avg[k].add_(v / float(len(selected)))
    return avg, count


def _adaptive_swa_candidate_windows(snapshots, args):
    epochs = sorted({int(s['epoch']) for s in snapshots})
    if len(epochs) == 0:
        return []
    ep_min = int(getattr(args, 'adaptive_swa_epoch_min', 1))
    ep_max = int(getattr(args, 'adaptive_swa_epoch_max', -1))
    if ep_max <= 0:
        ep_max = int(getattr(args, 'epochs', max(epochs)))
    min_len = max(1, int(getattr(args, 'adaptive_swa_min_len', 3)))
    max_len = max(min_len, int(getattr(args, 'adaptive_swa_max_len', 8)))
    stride = max(1, int(getattr(args, 'adaptive_swa_stride', 1)))
    available = [e for e in epochs if ep_min <= e <= ep_max]
    windows = []
    available_set = set(available)
    for start in available:
        if (start - ep_min) % stride != 0:
            continue
        for length in range(min_len, max_len + 1):
            end = start + length - 1
            if end > ep_max:
                continue
            # require a continuous epoch window with all snapshots available
            if all(e in available_set for e in range(start, end + 1)):
                windows.append((start, end, length))
    return windows


def _adaptive_swa_profile_defaults(args):
    """Default validation-only lifecycle priors.

    The main Module A path is intentionally generic. Historical profile names are
    accepted for old scripts, but model-specific priors now have to be supplied
    explicitly through the prior arguments below.
    """
    requested_profile = str(getattr(args, 'adaptive_swa_profile', 'generic') or 'generic').lower()
    profile = 'generic'
    ep_min = int(getattr(args, 'adaptive_swa_epoch_min', 1))
    ep_max = int(getattr(args, 'adaptive_swa_epoch_max', -1))
    if ep_max <= 0:
        ep_max = int(getattr(args, 'epochs', ep_min))
    max_len = max(1, int(getattr(args, 'adaptive_swa_max_len', 8)))

    d = {
        'base_metric': str(getattr(args, 'adaptive_swa_select_metric', 'selection_bacc_worst_std') or 'selection_bacc_worst_std'),
        'life_center': (ep_min + ep_max) / 2.0,
        'life_center_weight': 0.000,
        'early_weight': 0.000,
        'len_weight': 0.000,
        'std_lambda': 0.000,
        'hard_floor': -1.0,
        'hard_floor_lambda': 0.0,
        'late_end': float(ep_max),
        'late_lambda': 0.0,
        'tie_mode': 'short_early',
    }

    # User-supplied values override profile defaults. Negative means keep default.
    overrides = {
        'life_center': 'adaptive_swa_life_center',
        'life_center_weight': 'adaptive_swa_life_center_weight',
        'early_weight': 'adaptive_swa_early_weight',
        'len_weight': 'adaptive_swa_len_weight',
        'std_lambda': 'adaptive_swa_std_lambda',
        'hard_floor': 'adaptive_swa_hard_floor',
        'hard_floor_lambda': 'adaptive_swa_hard_floor_lambda',
        'late_end': 'adaptive_swa_late_end',
        'late_lambda': 'adaptive_swa_late_lambda',
    }
    for key, arg_name in overrides.items():
        val = float(getattr(args, arg_name, -1.0))
        if val >= 0.0:
            d[key] = val
    tie_mode = str(getattr(args, 'adaptive_swa_tie_mode', 'auto') or 'auto')
    if tie_mode != 'auto':
        d['tie_mode'] = tie_mode
    d['profile'] = profile
    d['requested_profile'] = requested_profile
    d['legacy_profile_alias'] = int(requested_profile != profile)
    d['ep_min'] = ep_min
    d['ep_max'] = ep_max
    d['max_len'] = max_len
    return d


def _adaptive_swa_metric(stats, details, args, start_epoch=None, end_epoch=None, length=None):
    """Validation-only window score. Never uses test stats.

    ASWA-v2 adds model-aware lifecycle priors to the validation score. The priors
    depend only on epoch index/window shape and validation per-class statistics,
    not on test performance.
    """
    if stats is None:
        return None, {}
    cfg = _adaptive_swa_profile_defaults(args)
    metric_name = cfg.get('base_metric') or str(getattr(args, 'adaptive_swa_select_metric', 'selection_bacc_worst_std') or 'selection_bacc_worst_std')
    score = _select_metric(stats, metric_name)
    if score is None:
        # For BIOT/LaBraM profile, fall back to balanced accuracy by design.
        score = stats.get('balanced_accuracy', None)
    if score is None:
        return None, {}

    start = float(start_epoch if start_epoch is not None else cfg['ep_min'])
    end = float(end_epoch if end_epoch is not None else start)
    length_f = float(length if length is not None else max(1.0, end - start + 1.0))
    center = (start + end) / 2.0
    ep_span = max(1.0, float(cfg['ep_max'] - cfg['ep_min'] + 1))
    max_len = max(1.0, float(cfg.get('max_len', length_f)))

    per_class = details.get('per_class_recall', None) if details is not None else None
    hard = _parse_int_list(getattr(args, 'adaptive_swa_hard_classes', ''), [])
    has_hard_classes = bool(hard)
    vals = []
    if per_class is not None:
        for c in hard:
            try:
                if c < len(per_class) and not np.isnan(per_class[c]):
                    vals.append(float(per_class[c]))
            except Exception:
                pass
    if len(vals) >= 2:
        imbalance = max(vals) - min(vals)
    else:
        imbalance = 0.0
    hard_min = float(min(vals)) if vals else 0.0
    hard_max = float(max(vals)) if vals else 0.0

    try:
        recall_std = float(stats.get('recall_std', 0.0))
    except Exception:
        recall_std = 0.0

    # Old optional direct imbalance penalty remains available, but defaults to 0
    # in lifecycle profiles so BIOT/LaBraM are not over-penalized by noisy hard classes.
    lam = float(getattr(args, 'adaptive_swa_balance_lambda', 0.0))
    hard_imbalance_penalty = lam * float(imbalance)

    life_center = float(cfg['life_center'])
    life_center_bonus = -float(cfg['life_center_weight']) * abs(center - life_center) / ep_span
    early_bonus = float(cfg['early_weight']) * (1.0 - (start - float(cfg['ep_min'])) / ep_span)
    len_bonus = float(cfg['len_weight']) * (length_f / max_len)
    std_penalty = float(cfg['std_lambda']) * recall_std

    hard_floor = float(cfg['hard_floor'])
    hard_floor_lambda = float(cfg['hard_floor_lambda'])
    hard_floor_penalty = 0.0
    if has_hard_classes and hard_floor >= 0.0:
        hard_floor_penalty = hard_floor_lambda * max(0.0, hard_floor - hard_min)

    late_end = float(cfg['late_end'])
    late_lambda = float(cfg['late_lambda'])
    late_penalty = 0.0
    if late_end >= 0.0:
        late_penalty = late_lambda * max(0.0, end - late_end) / ep_span

    final_score = (
        float(score)
        + float(life_center_bonus)
        + float(early_bonus)
        + float(len_bonus)
        - float(std_penalty)
        - float(hard_imbalance_penalty)
        - float(hard_floor_penalty)
        - float(late_penalty)
    )

    extra = {
        'aswa_profile': cfg['profile'],
        'aswa_requested_profile': cfg.get('requested_profile', cfg['profile']),
        'aswa_legacy_profile_alias': cfg.get('legacy_profile_alias', 0),
        'raw_metric_name': metric_name,
        'raw_metric': float(score),
        'life_center': float(life_center),
        'life_center_bonus': float(life_center_bonus),
        'early_bonus': float(early_bonus),
        'len_bonus': float(len_bonus),
        'recall_std_penalty': float(std_penalty),
        'hard_imbalance': float(imbalance) if has_hard_classes else '',
        'hard_imbalance_penalty': float(hard_imbalance_penalty) if has_hard_classes else '',
        'hard_min': hard_min if vals else '',
        'hard_max': hard_max if vals else '',
        'hard_floor': hard_floor if has_hard_classes else '',
        'hard_floor_penalty': float(hard_floor_penalty) if has_hard_classes else '',
        'late_end': float(late_end),
        'late_penalty': float(late_penalty),
        'tie_mode': cfg['tie_mode'],
    }
    return final_score, extra


def _prefer_adaptive_swa_candidate(new_item, best_item, tie_eps, args=None):
    if best_item is None:
        return True
    ns = float(new_item['score'])
    bs = float(best_item['score'])
    if ns > bs + float(tie_eps):
        return True
    if abs(ns - bs) <= float(tie_eps):
        cfg = _adaptive_swa_profile_defaults(args) if args is not None else {'tie_mode': 'short_early', 'life_center': 0.0}
        mode = str(cfg.get('tie_mode', 'short_early'))
        if mode == 'early_long':
            # LaBraM: early short-lifecycle states are valuable; if validation is close,
            # prefer earlier windows, then longer averaging coverage to avoid single-point selection.
            if int(new_item['start_epoch']) < int(best_item['start_epoch']):
                return True
            if int(new_item['start_epoch']) == int(best_item['start_epoch']) and int(new_item['length']) > int(best_item['length']):
                return True
        elif mode == 'long_mid':
            # BIOT: prefer a stable early/mid plateau; longer window then center closer to lifecycle prior.
            if int(new_item['length']) > int(best_item['length']):
                return True
            if int(new_item['length']) == int(best_item['length']):
                lc = float(cfg.get('life_center', 0.0))
                nc = (float(new_item['start_epoch']) + float(new_item['end_epoch'])) / 2.0
                bc = (float(best_item['start_epoch']) + float(best_item['end_epoch'])) / 2.0
                if abs(nc - lc) < abs(bc - lc):
                    return True
                if abs(nc - lc) == abs(bc - lc) and int(new_item['start_epoch']) < int(best_item['start_epoch']):
                    return True
        elif mode == 'hard_stable':
            # Compare hard-class stability only when the user configured a hard set.
            hard = _parse_int_list(getattr(args, 'adaptive_swa_hard_classes', ''), [])
            if hard:
                nh = new_item.get('hard_min', '')
                bh = best_item.get('hard_min', '')
                try:
                    nhf, bhf = float(nh), float(bh)
                    if nhf > bhf + 1e-6:
                        return True
                except Exception:
                    pass
            if int(new_item['length']) < int(best_item['length']):
                return True
            if int(new_item['length']) == int(best_item['length']) and int(new_item['start_epoch']) < int(best_item['start_epoch']):
                return True
        else:
            # v1-compatible default: shorter, then earlier.
            if int(new_item['length']) < int(best_item['length']):
                return True
            if int(new_item['length']) == int(best_item['length']) and int(new_item['start_epoch']) < int(best_item['start_epoch']):
                return True
    return False


def _adaptive_swa_forgetting_rows(window_details, final_details, args, start_epoch, end_epoch, length):
    if window_details is None or final_details is None:
        return None, []
    y_true = np.asarray(window_details.get('y_true', [])).reshape(-1).astype(np.int64)
    win_pred = np.asarray(window_details.get('y_pred', [])).reshape(-1).astype(np.int64)
    final_pred = np.asarray(final_details.get('y_pred', [])).reshape(-1).astype(np.int64)
    if y_true.size == 0 or y_true.size != win_pred.size or y_true.size != final_pred.size:
        return None, []

    win_ok = win_pred == y_true
    final_ok = final_pred == y_true
    retained = win_ok & final_ok
    learned = (~win_ok) & final_ok
    forgotten = win_ok & (~final_ok)
    always_wrong = (~win_ok) & (~final_ok)
    hard = set(_parse_int_list(getattr(args, 'adaptive_swa_hard_classes', ''), []))
    has_hard_classes = bool(hard)

    def count(mask):
        return int(np.asarray(mask, dtype=bool).sum())

    def rate(num, den):
        return float(num) / float(den) if int(den) > 0 else 0.0

    def transition_asymmetry(forget_num, learn_num):
        changed = int(forget_num) + int(learn_num)
        return float(int(forget_num) - int(learn_num)) / float(changed) if changed > 0 else 0.0

    total = int(y_true.size)
    hard_mask = np.asarray([int(c) in hard for c in y_true], dtype=bool)
    hard_total = count(hard_mask)
    win_correct = count(win_ok)
    final_correct = count(final_ok)
    retained_count = count(retained)
    learned_count = count(learned)
    forgotten_count = count(forgotten)
    always_wrong_count = count(always_wrong)
    transition_count = forgotten_count + learned_count
    lifecycle_retention_gap_count = forgotten_count - learned_count

    hard_retained_count = count(retained & hard_mask)
    hard_learned_count = count(learned & hard_mask)
    hard_forgotten_count = count(forgotten & hard_mask)
    hard_always_wrong_count = count(always_wrong & hard_mask)
    hard_transition_count = hard_forgotten_count + hard_learned_count
    hard_lifecycle_retention_gap_count = hard_forgotten_count - hard_learned_count

    summary = {
        'module_a_current': 'validation_guided_lifecycle_window',
        'split': 'val',
        'start_epoch': int(start_epoch),
        'end_epoch': int(end_epoch),
        'length': int(length),
        'total_count': total,
        'window_correct_count': win_correct,
        'final_correct_count': final_correct,
        'window_minus_final_correct': int(win_correct - final_correct),
        'retained_count': retained_count,
        'newly_learned_count': learned_count,
        'forgotten_count': forgotten_count,
        'always_wrong_count': always_wrong_count,
        'transition_count': transition_count,
        'lifecycle_retention_gap_count': lifecycle_retention_gap_count,
        'window_acc': rate(win_correct, total),
        'final_acc': rate(final_correct, total),
        'window_minus_final_acc': rate(win_correct - final_correct, total),
        'forgotten_rate': rate(forgotten_count, total),
        'newly_learned_rate': rate(learned_count, total),
        'transition_rate': rate(transition_count, total),
        'lifecycle_retention_gap': rate(lifecycle_retention_gap_count, total),
        'trajectory_forgetting_asymmetry': transition_asymmetry(forgotten_count, learned_count),
        'hard_classes': ','.join(str(x) for x in sorted(hard)),
        'hard_total_count': hard_total if has_hard_classes else '',
        'hard_retained_count': hard_retained_count if has_hard_classes else '',
        'hard_newly_learned_count': hard_learned_count if has_hard_classes else '',
        'hard_forgotten_count': hard_forgotten_count if has_hard_classes else '',
        'hard_always_wrong_count': hard_always_wrong_count if has_hard_classes else '',
        'hard_transition_count': hard_transition_count if has_hard_classes else '',
        'hard_lifecycle_retention_gap_count': hard_lifecycle_retention_gap_count if has_hard_classes else '',
        'hard_forgotten_rate': rate(hard_forgotten_count, hard_total) if has_hard_classes else '',
        'hard_newly_learned_rate': rate(hard_learned_count, hard_total) if has_hard_classes else '',
        'hard_transition_rate': rate(hard_transition_count, hard_total) if has_hard_classes else '',
        'hard_lifecycle_retention_gap': rate(hard_lifecycle_retention_gap_count, hard_total) if has_hard_classes else '',
        'hard_trajectory_forgetting_asymmetry': transition_asymmetry(hard_forgotten_count, hard_learned_count) if has_hard_classes else '',
    }

    class_rows = []
    for cls in range(int(getattr(args, 'nb_classes', 0) or 0)):
        cls_mask = y_true == cls
        support = count(cls_mask)
        if support <= 0:
            continue
        cls_win = count(win_ok & cls_mask)
        cls_final = count(final_ok & cls_mask)
        cls_learned = count(learned & cls_mask)
        cls_forgotten = count(forgotten & cls_mask)
        cls_transition = cls_forgotten + cls_learned
        cls_lifecycle_retention_gap_count = cls_forgotten - cls_learned
        class_rows.append({
            'module_a_current': 'validation_guided_lifecycle_window',
            'split': 'val',
            'start_epoch': int(start_epoch),
            'end_epoch': int(end_epoch),
            'length': int(length),
            'class_id': int(cls),
            'is_hard_class': int(cls in hard) if has_hard_classes else '',
            'support': support,
            'window_correct_count': cls_win,
            'final_correct_count': cls_final,
            'window_recall': rate(cls_win, support),
            'final_recall': rate(cls_final, support),
            'window_minus_final_recall': rate(cls_win - cls_final, support),
            'retained_count': count(retained & cls_mask),
            'newly_learned_count': cls_learned,
            'forgotten_count': cls_forgotten,
            'always_wrong_count': count(always_wrong & cls_mask),
            'transition_count': cls_transition,
            'lifecycle_retention_gap_count': cls_lifecycle_retention_gap_count,
            'forgotten_rate': rate(cls_forgotten, support),
            'newly_learned_rate': rate(cls_learned, support),
            'transition_rate': rate(cls_transition, support),
            'lifecycle_retention_gap': rate(cls_lifecycle_retention_gap_count, support),
            'trajectory_forgetting_asymmetry': transition_asymmetry(cls_forgotten, cls_learned),
        })
    return summary, class_rows

def run_lifecycle_window_search(args, model, data_loader_val, data_loader_test, device, metrics, snapshots, evaluate_fn=None, build_logit_adjust_bias_fn=None, eval_state_adjust_fn=None):
    """Validation-only adaptive lifecycle SWA.

    It searches continuous windows of captured trainable states using validation metrics only.
    Test is evaluated exactly once for the selected window and is not written for non-selected candidates.
    Optional model-specific evaluation-state adjustments are supplied by the training script.
    """
    if evaluate_fn is None:
        raise ValueError('run_lifecycle_window_search requires evaluate_fn')
    if not getattr(args, 'adaptive_swa_eval', False):
        return None
    if args.task_mod != 'Classification' or int(args.nb_classes) <= 1:
        print('[Adaptive-SWA] skipped: only multiclass classification is supported.')
        return None
    if data_loader_val is None or data_loader_test is None:
        print('[Adaptive-SWA] skipped: validation/test loader missing.')
        return None
    if snapshots is None or len(snapshots) == 0:
        print('[Adaptive-SWA] skipped: no epoch snapshots captured.')
        return None

    diag_dir = os.path.join(args.output_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)
    cand_csv = os.path.join(diag_dir, 'adaptive_swa_windows.csv')
    final_csv = os.path.join(diag_dir, 'adaptive_swa_eval.csv')
    forgetting_csv = os.path.join(diag_dir, 'adaptive_swa_forgetting_summary.csv')
    forgetting_class_csv = os.path.join(diag_dir, 'adaptive_swa_forgetting_by_class.csv')
    # fresh files for this run
    for p in [cand_csv, final_csv, forgetting_csv, forgetting_class_csv]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    final_state = _get_model_cpu_state(model)
    profile_cfg = _adaptive_swa_profile_defaults(args)
    windows = _adaptive_swa_candidate_windows(snapshots, args)
    if len(windows) == 0:
        print('[Adaptive-SWA] no candidate windows generated.')
        return None

    print(f"[Adaptive-SWA] searching {len(windows)} validation-only windows. profile={profile_cfg.get('profile', 'generic')}, requested_profile={profile_cfg.get('requested_profile', 'generic')}, metric={getattr(args, 'adaptive_swa_select_metric', 'selection_bacc_worst_std') or 'selection_bacc_worst_std'}, len=[{getattr(args, 'adaptive_swa_min_len', '')},{getattr(args, 'adaptive_swa_max_len', '')}]")
    best_item = None
    tie_eps = float(getattr(args, 'adaptive_swa_tie_eps', 0.002))

    for start, end, length in windows:
        avg_state, count = _average_adaptive_swa_window(snapshots, start, end)
        if avg_state is None or count <= 0:
            continue
        if int(getattr(args, 'adaptive_swa_filter_rank', -1)) > 0:
            avg_state = _rank_filter_lora_state(avg_state, int(getattr(args, 'adaptive_swa_filter_rank', -1)))
        model.load_state_dict(final_state, strict=False)
        _apply_partial_float_state(model, avg_state)
        if eval_state_adjust_fn is not None:
            eval_state_adjust_fn(model)
        val_stats, val_details = evaluate_fn(
            args, data_loader_val, model, device,
            header=f'Adaptive-SWA-Val-{start}-{end}:',
            ch_names=None, metrics=metrics, return_details=True
        )
        _add_selection_metrics(val_stats, val_details, args)
        score, extra = _adaptive_swa_metric(val_stats, val_details, args, start_epoch=start, end_epoch=end, length=length)
        if score is None:
            continue
        row = {
            'start_epoch': int(start),
            'end_epoch': int(end),
            'length': int(length),
            'count': int(count),
            'select_metric': getattr(args, 'adaptive_swa_select_metric', 'selection_bacc_worst_std') or 'selection_bacc_worst_std',
            'score': float(score),
            'score_is_validation_only': 1,
            'test_used_for_selection': 0,
            'tie_eps': tie_eps,
        }
        row.update(extra)
        _flatten_eval_row('val', val_stats, row)
        _add_per_class_to_row('val', val_details, row)
        _append_csv_row(cand_csv, row)
        item = dict(row)
        item['state'] = avg_state
        if _prefer_adaptive_swa_candidate(item, best_item, tie_eps, args=args):
            best_item = item

    model.load_state_dict(final_state, strict=False)

    if best_item is None:
        print('[Adaptive-SWA] no valid candidate after validation evaluation.')
        return None

    selected_state = best_item['state']
    model.load_state_dict(final_state, strict=False)
    _apply_partial_float_state(model, selected_state)
    if eval_state_adjust_fn is not None:
        eval_state_adjust_fn(model)

    # Re-evaluate selected window on validation to compute calibration bias if needed.
    val_stats, val_details = evaluate_fn(
        args, data_loader_val, model, device,
        header=f"Adaptive-SWA-Selected-Val-{best_item['start_epoch']}-{best_item['end_epoch']}:",
        ch_names=None, metrics=metrics, return_details=True
    )
    _add_selection_metrics(val_stats, val_details, args)

    test_logit_bias = None
    test_header = f"Adaptive-SWA-Selected-Test-{best_item['start_epoch']}-{best_item['end_epoch']}:"
    if getattr(args, 'eval_logit_adjust', False) and args.task_mod == 'Classification' and val_details is not None:
        if build_logit_adjust_bias_fn is None:
            raise ValueError('run_lifecycle_window_search requires build_logit_adjust_bias_fn when eval_logit_adjust is enabled')
        test_logit_bias = build_logit_adjust_bias_fn(
            val_details=val_details,
            nb_classes=args.nb_classes,
            strength=args.logit_adjust_strength,
            clip=args.logit_adjust_clip,
        )
        if test_logit_bias is not None:
            test_header = f"Adaptive-SWA-Selected-Test-Calib-{best_item['start_epoch']}-{best_item['end_epoch']}:"

    test_stats, test_details = evaluate_fn(
        args, data_loader_test, model, device,
        header=test_header,
        ch_names=None, metrics=metrics, return_details=True,
        logit_bias=test_logit_bias
    )
    _add_selection_metrics(test_stats, test_details, args)

    final_val_stats, final_val_details = None, None
    forgetting_summary, forgetting_class_rows = None, []
    try:
        model.load_state_dict(final_state, strict=False)
        if eval_state_adjust_fn is not None:
            eval_state_adjust_fn(model)
        final_val_stats, final_val_details = evaluate_fn(
            args, data_loader_val, model, device,
            header=f"Adaptive-SWA-Final-Val-{best_item['start_epoch']}-{best_item['end_epoch']}:",
            ch_names=None, metrics=metrics, return_details=True
        )
        _add_selection_metrics(final_val_stats, final_val_details, args)
        forgetting_summary, forgetting_class_rows = _adaptive_swa_forgetting_rows(
            window_details=val_details,
            final_details=final_val_details,
            args=args,
            start_epoch=best_item['start_epoch'],
            end_epoch=best_item['end_epoch'],
            length=best_item['length'],
        )
        if forgetting_summary is not None:
            _append_csv_row(forgetting_csv, forgetting_summary)
            for cls_row in forgetting_class_rows:
                _append_csv_row(forgetting_class_csv, cls_row)
            message = (
                f"[Adaptive-SWA] val forgetting diagnostic: "
                f"forgotten={forgetting_summary['forgotten_count']}, "
                f"newly_learned={forgetting_summary['newly_learned_count']}, "
                f"LRG={forgetting_summary['lifecycle_retention_gap']:.6f}"
            )
            if forgetting_summary.get('hard_classes'):
                message += (
                    f", hard_LRG={forgetting_summary['hard_lifecycle_retention_gap']:.6f}, "
                    f"hard_TFA={forgetting_summary['hard_trajectory_forgetting_asymmetry']:.6f}"
                )
            print(message)
    finally:
        model.load_state_dict(final_state, strict=False)
        _apply_partial_float_state(model, selected_state)
        if eval_state_adjust_fn is not None:
            eval_state_adjust_fn(model)

    final_row = {
        'mode': 'adaptive_swa_validation_selected',
        'module_a_current': 'validation_guided_lifecycle_window',
        'module_a_role': 'temporal_window_selector',
        'trajectory_diagnostic': 'window_vs_final_validation_forgetting' if forgetting_summary is not None else '',
        'adapter_target': str(getattr(args, 'lora_target', '')),
        'fb_recipe': str(getattr(args, 'fb_recipe', '')),
        'start_epoch': int(best_item['start_epoch']),
        'end_epoch': int(best_item['end_epoch']),
        'length': int(best_item['length']),
        'count': int(best_item['count']),
        'select_metric': getattr(args, 'adaptive_swa_select_metric', 'selection_bacc_worst_std') or 'selection_bacc_worst_std',
        'adaptive_swa_profile': getattr(args, 'adaptive_swa_profile', 'generic'),
        'adaptive_swa_profile_effective': profile_cfg.get('profile', 'generic'),
        'selection_score': float(best_item['score']),
        'score_is_validation_only': 1,
        'test_used_for_selection': 0,
        'candidate_count': int(len(windows)),
        'candidate_epoch_min': int(getattr(args, 'adaptive_swa_epoch_min', 1)),
        'candidate_epoch_max': int(getattr(args, 'adaptive_swa_epoch_max', -1)),
        'candidate_min_len': int(getattr(args, 'adaptive_swa_min_len', 3)),
        'candidate_max_len': int(getattr(args, 'adaptive_swa_max_len', 8)),
        'tie_eps': tie_eps,
    }
    _flatten_eval_row('val', val_stats, final_row)
    _flatten_eval_row('val_final', final_val_stats, final_row)
    _flatten_eval_row('test', test_stats, final_row)
    _add_per_class_to_row('val', val_details, final_row)
    _add_per_class_to_row('val_final', final_val_details, final_row)
    _add_per_class_to_row('test', test_details, final_row)
    if forgetting_summary is not None:
        for k, v in forgetting_summary.items():
            if k in ('module_a_current', 'split', 'start_epoch', 'end_epoch', 'length'):
                continue
            final_row[f'val_lif_{k}'] = v
    _append_csv_row(final_csv, final_row)

    if getattr(args, 'adaptive_swa_save_selected_ckpt', True):
        ckpt_dir = os.path.join(args.output_dir, 'monitor_checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, 'adaptive_swa_selected.pth')
        torch.save({
            'epoch': f"adaptive_swa_{best_item['start_epoch']}_{best_item['end_epoch']}",
            'model': _get_model_cpu_state(model),
            'args': vars(args),
            'adaptive_swa_selection': final_row,
        }, ckpt_path)
        print(f'[Adaptive-SWA] selected checkpoint saved to: {ckpt_path}')

    print(
        f"[Adaptive-SWA] selected window {best_item['start_epoch']}-{best_item['end_epoch']} "
        f"len={best_item['length']} by VAL score={best_item['score']:.6f}. "
        f"Final test BAcc={test_stats.get('balanced_accuracy', float('nan')) * 100:.2f}%"
    )

    # Restore original final training weights after reporting.
    model.load_state_dict(final_state, strict=False)
    return final_row
