import copy
import os
import sys
import pathlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


def _patch_pathlib_for_cross_os_checkpoints():
    """
    Some official checkpoints were saved on Linux and contain pathlib.PosixPath
    objects inside the pickled config. On Windows, unpickling PosixPath raises:
        NotImplementedError: cannot instantiate 'PosixPath' on your system

    Gram's checkpoints can hit this before we even reach weight loading, so we
    patch pathlib once inside this wrapper. The path objects are only config
    metadata; model tensors are unaffected.
    """
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath


class GramAdaBackbone(nn.Module):
    """
    AdaBrain-Bench wrapper for the official Gram fine-tuning model.

    This wrapper only adapts AdaBrain-Bench samples [B, C, T] into the official
    Gram input shape and keeps AdaBrain-Bench responsible for the few-shot split,
    optimizer, metrics, logging and output folders.

    Required files for a real foundation-model baseline:
      external/Gram/model/modeling_Gram_finetune.py
      checkpoints/gram_base.pth
      checkpoints/base_class_quantization.pth

    `gram_base.pth` is expected to contain the official Gram pretrained weights
    and usually a saved config object under key `cf`. We reuse that config when
    available, then override only downstream-safe fields such as number of
    classes, device and VQGAN checkpoint path.
    """

    def __init__(self, args, ch_names, num_t, from_pretrain: bool = False):
        super().__init__()
        _patch_pathlib_for_cross_os_checkpoints()
        self.args = args
        self.ch_names = self._normalize_ch_names(ch_names)
        self.num_t = int(num_t)
        self.sample_rate = int(getattr(args, "sampling_rate", 200))
        self.gram_root = self._resolve_gram_root(args)
        self.GramClass = self._import_gram_class(self.gram_root)

        self.ckpt_path = str(getattr(args, "gram_ckpt", "") or "./checkpoints/gram_base.pth")
        self.vqgan_path = str(getattr(args, "gram_vqgan_ckpt", "") or "./checkpoints/base_class_quantization.pth")
        self.allow_scratch = bool(getattr(args, "gram_allow_scratch", False))

        self.checkpoint = self._load_checkpoint_object(self.ckpt_path)
        self.cf = self._build_config_from_checkpoint(args, self.checkpoint)
        self.model = self.GramClass(self.cf)

        if from_pretrain:
            self._load_pretrained()

    @staticmethod
    def _normalize_ch_names(ch_names: Iterable[str]):
        out = []
        for ch in list(ch_names):
            name = str(ch).strip().upper()
            # Keep bipolar names such as FP1-F7 unchanged. Official Gram's
            # all_ch_list includes TUEV bipolar channels.
            out.append(name)
        return out

    def _resolve_gram_root(self, args) -> Path:
        candidates = []
        arg_root = str(getattr(args, "gram_root", "") or "").strip()
        env_root = os.environ.get("GRAM_ROOT", "").strip()
        if arg_root:
            candidates.append(Path(arg_root))
        if env_root:
            candidates.append(Path(env_root))
        candidates.extend([
            Path("external") / "Gram",
            Path("third_party") / "Gram",
            Path("models") / "Gram",
            Path("Gram"),
        ])
        for root in candidates:
            if (root / "model" / "modeling_Gram_finetune.py").exists():
                return root.resolve()
        raise FileNotFoundError(
            "Cannot find official Gram code. Put Gram-main under external/Gram "
            "or pass --gram_root / set GRAM_ROOT. Expected file: "
            "external/Gram/model/modeling_Gram_finetune.py"
        )

    def _import_gram_class(self, root: Path):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            from model.modeling_Gram_finetune import Gram
        except Exception as exc:
            raise ImportError(
                f"Failed to import official Gram from {root}. Check external/Gram and dependencies. "
                f"Original error: {repr(exc)}"
            ) from exc
        return Gram

    def _load_checkpoint_object(self, ckpt_path: str):
        if os.path.exists(ckpt_path):
            print(f"[Gram] found pretrained checkpoint: {ckpt_path}")
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not self.allow_scratch:
            raise FileNotFoundError(
                f"[Gram] pretrained checkpoint not found: {ckpt_path}\n"
                "For a real foundation-model baseline, put gram_base.pth at this path "
                "or pass --gram_ckpt. Use --gram_allow_scratch only for interface debugging."
            )
        print(f"[Gram] WARNING: checkpoint not found and --gram_allow_scratch is set: {ckpt_path}")
        return None

    def _default_all_ch_list(self):
        return [
            'FP1', 'FPZ', 'FP2',
            'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
            'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
            'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
            'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
            'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
            'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
            'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
            'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
            'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
            'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
            'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
            'T1', 'T2',
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        ]

    def _fallback_config(self, args):
        return SimpleNamespace(
            if_finetune=True,
            if_scratch=False,
            n_class=int(getattr(args, "nb_classes", 6)),
            num_classes=int(getattr(args, "nb_classes", 6)),
            sample_rate=int(getattr(args, "sampling_rate", 200)),
            device=getattr(args, "device", "cuda"),
            vocab_size=8196,
            vocab_emb=32,
            n_chans=1,
            out_chans=8,
            out_indices=[0, 2, 4, 6, 8, 11],
            n_embd=200,
            drop_path_rate=0.1,
            n_layer=12,
            n_head=10,
            target_n_embd=200,
            decoder_n_embd=200,
            decoder_drop_path_rate=0.1,
            decoder_n_layer=8,
            decoder_n_head=10,
            model_window_size=200,
            embd_pdrop=0.05,
            resid_pdrop=0.05,
            attn_pdrop=0.05,
            layer_scale_init_values=0.1,
            if_sandwich_norm=False,
            if_causal_attention=False,
            if_mff=True,
            if_mimic=True,
            if_pad_with_cls_token=True,
            window_size=1000,
            all_ch_list=self._default_all_ch_list(),
            vqgan_model_path=self.vqgan_path,
        )

    def _set_default(self, cf, name: str, value: Any):
        if not hasattr(cf, name):
            setattr(cf, name, value)

    def _build_config_from_checkpoint(self, args, checkpoint):
        if isinstance(checkpoint, dict) and "cf" in checkpoint:
            cf = copy.deepcopy(checkpoint["cf"])
            print("[Gram] using config object from gram_base.pth['cf']")
        else:
            cf = self._fallback_config(args)
            if checkpoint is not None:
                print("[Gram] checkpoint has no key 'cf'; using fallback config. Check matched tensor count carefully.")

        # Required downstream overrides.
        setattr(cf, "if_finetune", True)
        setattr(cf, "if_scratch", False)
        setattr(cf, "n_class", int(getattr(args, "nb_classes", 6)))
        setattr(cf, "num_classes", int(getattr(args, "nb_classes", 6)))
        setattr(cf, "sample_rate", int(getattr(args, "sampling_rate", 200)))
        setattr(cf, "device", getattr(args, "device", "cuda"))
        setattr(cf, "vqgan_model_path", self.vqgan_path)

        # Defensive defaults for older checkpoints.
        fallback = self._fallback_config(args)
        for name, value in vars(fallback).items():
            self._set_default(cf, name, value)

        if not os.path.exists(self.vqgan_path):
            raise FileNotFoundError(
                f"[Gram] base-class quantization checkpoint not found: {self.vqgan_path}\n"
                "Put base_class_quantization.pth under checkpoints/ or pass --gram_vqgan_ckpt. "
                "Official Gram fine-tune model needs this VQGAN/codebook checkpoint during construction."
            )

        # Check channel names early, otherwise official forward only prints and later fails.
        all_ch = [str(x).upper() for x in list(getattr(cf, "all_ch_list", []))]
        missing = [c for c in self.ch_names if c not in all_ch]
        if missing:
            print(f"[Gram] WARNING: {len(missing)} Ada channel names not found in Gram all_ch_list: {missing}")
            print("[Gram] If forward later prints 'new_ch_list does not match ori_ch_list', fix channel names before trusting results.")
        return cf

    def _state_from_checkpoint(self):
        checkpoint = self.checkpoint
        if checkpoint is None:
            return None
        if isinstance(checkpoint, dict):
            return checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        return checkpoint

    def _load_pretrained(self):
        state = self._state_from_checkpoint()
        if state is None:
            if self.allow_scratch:
                print("[Gram] WARNING: running initialized Gram because --gram_allow_scratch is set.")
                return
            raise RuntimeError("[Gram] no checkpoint state available for pretrained loading.")

        current = self.model.state_dict()
        matched = {}
        for key, value in state.items():
            clean_key = str(key).replace("module.", "")
            if "vqgan" in clean_key:
                continue
            if clean_key in current and hasattr(value, "shape") and current[clean_key].shape == value.shape:
                matched[clean_key] = value
        current.update(matched)
        missing, unexpected = self.model.load_state_dict(current, strict=False)
        print(f"[Gram] matched tensors={len(matched)}, missing={len(missing)}, unexpected={len(unexpected)}")
        if len(matched) == 0 and not self.allow_scratch:
            raise RuntimeError(
                "[Gram] matched tensor count is 0. This would be a random/scratch baseline, "
                "so the run is stopped. Check gram_base.pth and wrapper config."
            )

    def forward(self, x):
        # Ada samples are [B, C, T]. Official Gram fine-tuning uses [B, n*C, 200]
        # where n = T / sample_rate. For TUEV in Ada, T=1000 and sample_rate=200,
        # so n=5.
        if x.dim() != 3:
            raise ValueError(f"GramAdaBackbone expects [B, C, T], got shape={tuple(x.shape)}")
        b, c, t = x.shape
        if t % self.sample_rate != 0:
            raise ValueError(f"Input length {t} is not divisible by sample_rate={self.sample_rate}")
        x = x / 100.0
        x = x.reshape(b, c, t // self.sample_rate, self.sample_rate).permute(0, 2, 1, 3)
        x = x.reshape(b, (t // self.sample_rate) * c, self.sample_rate)
        out = self.model(x, self.ch_names)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
