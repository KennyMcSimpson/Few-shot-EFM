import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeurIPTAdaBackbone(nn.Module):
    """
    AdaBrain-Bench wrapper for the NeurIPT stage-2 classification model.

    This wrapper keeps AdaBrain-Bench responsible for few-shot split, optimizer,
    metrics, logging and output folders. NeurIPT only provides the model forward.

    Input from Ada: [B, C, T]
    NeurIPT stage-2 expects: [B, total_len, data_dim]

    For TUEV, the NeurIPT script uses data_dim=20 and total_len=320. Ada's TUEV
    processed data is commonly 16 bipolar channels with T=1000. Therefore this
    wrapper uses:
      - temporal interpolation T -> neuript_total_len, default 320;
      - a lightweight 1x1 channel adapter C -> neuript_data_dim, default 20.

    If you later regenerate data in the native NeurIPT format, set
    --neuript_data_dim to the actual channel count and the adapter will become
    Identity when C matches.
    """

    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        self.args = args
        self.ch_names = list(ch_names)
        self.num_t = int(num_t)
        self.source_root = self._resolve_source_root(args)
        self.Stage2Class = self._import_stage2(self.source_root)
        self.n_classes = int(getattr(args, "nb_classes", 6))
        self.data_dim = int(getattr(args, "neuript_data_dim", 20))
        self.total_len = int(getattr(args, "neuript_total_len", 320))
        self.d_model = int(getattr(args, "neuript_d_model", 768))
        self.d_ff = int(getattr(args, "neuript_d_ff", 768))
        self.n_heads = int(getattr(args, "neuript_n_heads", 8))
        self.dropout = float(getattr(args, "neuript_dropout", 0.1))
        self.c_layers = int(getattr(args, "neuript_c_layers", 6))
        self.part = str(getattr(args, "neuript_part", "Functional"))
        self.merge_layers = self._parse_int_list(getattr(args, "neuript_merge_layers", "1,4,1,2,1,2"), [1, 4, 1, 2, 1, 2])
        self.enc_expert = self._parse_int_list(getattr(args, "neuript_enc_expert", "0,0,2,4,4,6"), [0, 0, 2, 4, 4, 6])
        self.e_layers = len(self.merge_layers)
        if len(self.enc_expert) != self.e_layers:
            raise ValueError(
                f"NeurIPT enc_expert length must match merge_layers length, got "
                f"{len(self.enc_expert)} vs {self.e_layers}."
            )

        self.input_adapter = nn.Identity()
        if len(self.ch_names) != self.data_dim:
            print(f"[NeurIPT-Ada] channel adapter enabled: Ada channels={len(self.ch_names)} -> NeurIPT data_dim={self.data_dim}")
            self.input_adapter = nn.Conv1d(len(self.ch_names), self.data_dim, kernel_size=1, bias=True)

        self.nargs = self._build_neuript_args()
        self.model = self.Stage2Class(
            data_dim=self.nargs.data_dim,
            out_len=self.nargs.out_len,
            seg_len=self.nargs.seg_len,
            merge_layers=self.nargs.merge_layers,
            args=self.nargs,
            factor=self.nargs.factor,
            d_model=self.nargs.d_model,
            d_ff=self.nargs.d_ff,
            n_heads=self.nargs.n_heads,
            e_layers=self.nargs.e_layers,
            dropout=self.nargs.dropout,
            use_norm=self.nargs.use_norm,
            c_layers=self.nargs.c_layers,
            num_classes=self.nargs.num_classes,
            d_middle=self.nargs.d_middle,
            part=self.nargs.part,
            use_router=self.nargs.use_router,
        )

        if from_pretrain:
            self._load_pretrained(args)

    @staticmethod
    def _parse_int_list(text, default):
        if text is None:
            return list(default)
        if isinstance(text, (list, tuple)):
            return [int(x) for x in text]
        out = []
        for item in str(text).replace(";", ",").split(","):
            item = item.strip()
            if item:
                out.append(int(item))
        return out if out else list(default)

    def _resolve_source_root(self, args) -> Path:
        env_root = os.environ.get("NEURIPT_ROOT", "").strip()
        candidates = []
        arg_root = getattr(args, "neuript_source_root", "")
        if arg_root:
            candidates.append(Path(arg_root))
        if env_root:
            candidates.append(Path(env_root))
        candidates.extend([
            Path("external") / "NeurIPT",
            Path("third_party") / "NeurIPT",
            Path("models") / "NeurIPT",
            Path("NeurIPT"),
        ])
        for root in candidates:
            if (root / "cross_models" / "stage_2_model.py").exists():
                return root.resolve()
        raise FileNotFoundError(
            "Cannot find NeurIPT source code. Put it under external/NeurIPT, "
            "or set NEURIPT_ROOT / --neuript_source_root to the source root."
        )

    def _import_stage2(self, root: Path):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        try:
            from cross_models.stage_2_model import Stage_2_model
        except Exception as exc:
            raise ImportError(f"Failed to import NeurIPT Stage_2_model from {root}: {repr(exc)}") from exc
        return Stage_2_model

    def _build_neuript_args(self):
        d_model = self.d_model
        return SimpleNamespace(
            data="TUEV",
            cls_only=True,
            data_dim=self.data_dim,
            total_len=self.total_len,
            in_len=self.total_len,
            out_len=self.total_len,
            seg_len=1,
            d_model=d_model,
            d_ff=self.d_ff,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            merge_layers=self.merge_layers,
            enc_expert=self.enc_expert,
            factor=10,
            use_router=False,
            use_norm=False,
            c_layers=self.c_layers,
            num_classes=self.n_classes,
            d_middle=32,
            part=self.part,
            input_dim=d_model,
            output_dim=d_model,
            hidden_dim=int(getattr(self.args, "neuript_hidden_dim", 512)),
            hidden_dim_shared=int(getattr(self.args, "neuript_hidden_dim_shared", 768)),
            top_k=float(getattr(self.args, "neuript_top_k", 0.5)),
            use_shared_expert=True,
            noise_std=float(getattr(self.args, "neuript_noise_std", 0.001)),
            w_importance=float(getattr(self.args, "neuript_w_importance", 0.008)),
        )

    def _load_pretrained(self, args):
        ckpt_path = getattr(args, "neuript_ckpt", "") or "./checkpoints/neuript_stage2.pth"
        if not os.path.exists(ckpt_path):
            allow = bool(getattr(args, "neuript_allow_scratch", False))
            msg = f"[NeurIPT] pretrained checkpoint not found: {ckpt_path}"
            if allow:
                print(msg + " ; --neuript_allow_scratch is set, so this run is only a shape/scratch baseline.")
                return
            raise FileNotFoundError(
                msg + "\nFor a real foundation-model baseline, put the NeurIPT stage-2 checkpoint at this path "
                "or pass --neuript_ckpt. For only testing the wrapper shape, add --neuript_allow_scratch."
            )

        print(f"[NeurIPT] load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        state = {str(k).replace("module.", ""): v for k, v in state.items()}
        # Official code drops PE buffers before loading because they depend on downstream geometry.
        state = {k: v for k, v in state.items() if "temporal_PE" not in k and "spacial_PE" not in k}
        current = self.model.state_dict()
        matched = {k: v for k, v in state.items() if k in current and hasattr(v, "shape") and current[k].shape == v.shape}
        current.update(matched)
        missing, unexpected = self.model.load_state_dict(current, strict=False)
        print(f"[NeurIPT] matched tensors={len(matched)}, missing={len(missing)}, unexpected={len(unexpected)}")

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"NeurIPTAdaBackbone expects [B, C, T], got shape={tuple(x.shape)}")
        x = self.input_adapter(x)
        if x.shape[-1] != self.total_len:
            x = F.interpolate(x, size=self.total_len, mode="linear", align_corners=False)
        x = x.transpose(1, 2).contiguous()  # [B, total_len, data_dim]
        logits, aux_loss = self.model(x)
        # Keep aux loss available for future diagnostics without changing Ada's training loop.
        self.last_aux_loss = aux_loss
        return logits
