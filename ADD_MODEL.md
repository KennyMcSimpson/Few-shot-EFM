# Add a Backbone

Few-shot EFM wraps each EEG foundation model behind a small adapter class so
that the training loop can attach task heads, LoRA modules, diagnostics, and
functional-block metadata in a consistent way.

This guide describes the minimal steps for adding a new backbone without
changing the existing model behavior.

## 1. Add Model Code

Place the model implementation under `models/` or under `external/<name>/` if
you are vendoring a third-party project. Keep third-party code isolated when
possible.

Example:

```text
models/my_backbone.py
external/MyBackbone/
```

## 2. Register Checkpoint Path

Add the expected local checkpoint filename to `finetune_list` in
`run_finetuning.py`:

```python
finetune_list = {
    "MyBackbone": "./checkpoints/my_backbone.pth",
}
```

Do not commit checkpoint files to Git.

## 3. Write A Wrapper

The wrapper should expose:

- `self.main_model`: the pretrained backbone;
- `self.task_head`: an `nn.Module` attached by `get_models()`;
- `forward(x)`: input preprocessing, backbone call, task head call;
- optional metadata such as `input_channels` when needed by adapter logic.

Skeleton:

```python
class Ada_MyBackbone(nn.Module):
    def __init__(self, args, ch_names, num_t, from_pretrain=False):
        super().__init__()
        model = MyBackbone(...)
        if from_pretrain:
            state = torch.load(finetune_list[args.model_name], map_location="cpu")
            model.load_state_dict(state, strict=False)
        self.main_model = model
        self.task_head = nn.Identity()

    def forward(self, x):
        features = self.main_model(x)
        return self.task_head(features)
```

Keep model-specific reshaping inside the wrapper. The rest of the training
code should not need to know the original model's private input convention.

## 4. Attach Task Heads

Register the wrapper in `get_models()` in `run_finetuning.py`:

```python
elif args.model_name == "MyBackbone":
    model = Ada_MyBackbone(args, ch_names, num_t, from_pretrain=True)
    if args.task_mod == "Classification":
        model.task_head = LinearWithConstraint(hidden_dim, args.nb_classes, max_norm=1)
    elif args.task_mod == "Regression":
        model.task_head = RegressionLayers(hidden_dim, hidden_dim, 1)
    elif args.task_mod == "Retrieval":
        model.task_head = LinearWithConstraint(hidden_dim, 1024, max_norm=1)
```

Use the existing heads unless the backbone requires a different feature shape.

## 5. Add LoRA Target Mapping

If the backbone should support LoRA, update the target resolution logic in
`util/lora.py` and the functional-block registry in `util/fb_registry.py`.

The goal is not just to name modules. Map modules to meaningful EEGFM
functional regions such as:

```text
input_front
spatial
temporal
spectral
mixing
semantic
readout
```

This keeps adapter choices comparable across backbones.

## 6. Check The Integration

Run a short import or dry run before launching expensive experiments:

```bash
python run_finetuning.py \
  --dataset TUEV \
  --model_name MyBackbone \
  --task_mod Classification \
  --subject_mod fewshot \
  --finetune_mod linear \
  --epochs 1 \
  --batch_size 2
```

For LoRA integration, also check:

```bash
python run_finetuning.py \
  --dataset TUEV \
  --model_name MyBackbone \
  --task_mod Classification \
  --subject_mod fewshot \
  --finetune_mod lora \
  --lora_target semantic \
  --fb_enable \
  --fb_probe \
  --epochs 1 \
  --batch_size 2
```

Use local data and local checkpoints. Do not commit generated outputs.
