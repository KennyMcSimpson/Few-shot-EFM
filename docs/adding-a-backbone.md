# Adding a backbone

Few-shot EFM presents every backbone through a small wrapper so the training
loop can attach a task head, inject LoRA, and classify parameters by functional
role without knowing the backbone's private tensor conventions.

## 1. Add and attribute the source

Place original project code under `external/<name>/` when vendoring is required,
or add a focused wrapper under `models/` when the dependency can be imported.
Retain its license and original README, and add its origin and local changes to
`THIRD_PARTY_NOTICES.md`.

## 2. Register checkpoint handling

Add the expected local checkpoint path to `finetune_list` in
`run_finetuning.py`. Checkpoints must remain outside Git. Default behavior for a
foundation-model baseline should fail clearly when required pretrained weights
are absent; scratch initialization must be an explicit debugging option.

## 3. Implement the wrapper

The wrapper should expose:

- `main_model`, the pretrained feature extractor;
- `task_head`, an `nn.Module` attached by `get_models()`;
- `forward(x)`, including all model-specific reshaping;
- channel or sampling-rate metadata required by input adaptation.

Keep model-specific interpolation, channel projection, and output unwrapping in
the wrapper. The shared training loop should receive ordinary task logits.

## 4. Attach task heads

Register the wrapper in `get_models()` and attach an existing classification,
regression, or retrieval head when its feature shape permits. Add a shape test
for every supported task instead of relying on a full training run to discover
an interface mismatch.

## 5. Map functional regions

Update `util/fb_registry.py` and `util/lora.py` so LoRA targets resolve to real
parameters. Map architectural names onto the shared roles:

```text
input_front  spatial  temporal  spectral  mixing  semantic  readout
```

Module E remains the general structural-routing role. A backbone-specific
pattern table is an implementation mapping, not a new model-specific module.
Check both false positives and false negatives in parameter-name matching.

## 6. Test integration

At minimum, verify:

1. wrapper import and forward shape;
2. required checkpoint failure and explicit scratch-debug behavior;
3. resolved LoRA targets and trainable-parameter counts;
4. optimizer construction after LoRA injection;
5. one training/evaluation smoke step;
6. Module C branch construction if the backbone claims B/D/E coverage;
7. checkpoint resume compatibility.

Use local data and weights. Do not commit generated outputs.
