# Few-shot EFM

Few-shot EFM is a research codebase for few-shot adaptation of EEG foundation
models. It is built on the AdaBrain-Bench training framework, but the focus of
this repository is the project-specific adaptation layer: functional-block LoRA,
diagnostic module selection, and validation-only adapter lifecycle evaluation
for small-data EEG classification.

This is not a clean-room replacement for AdaBrain-Bench. The dataset loaders,
many baseline wrappers, and the main fine-tuning loop inherit from that
framework. The added research logic is organized around model-aware adapter
targets and diagnostics rather than a leaderboard copy of the original
benchmark.

## Scope

The repository tracks source code, configuration, preprocessing scripts, and
documentation. It does not track local datasets, generated split JSON files,
checkpoints, logs, or experiment result folders. Large artifacts and completed
runs are stored separately on Hugging Face:

```text
https://huggingface.co/datasets/KennySimpson/few-shot-EFM
```

## Main Components

```text
run_finetuning.py              Main training and evaluation entrypoint
engine_for_finetuning.py       Training loop, evaluation, loss, calibration, lifecycle hooks
module_a_lifecycle.py          Validation-only adapter lifecycle and adaptive-SWA utilities

models/                        EEG model wrappers and backbone integrations
util/lora.py                   LoRA injection and trainable-parameter control
util/fb_*.py                   Functional-block registry, policy, probes, and collection
util/module_b_*.py             Signal/input-side adaptation helpers
util/module_c_*.py             Matched low-budget Module C functional-action search
util/module_d_*.py             Semantic refinement utilities
util/module_e_*.py             Structural routing and pressure-guided helpers

preprocessing/                 Dataset preprocessing and split-generation scripts
dataset_config/                Dataset root/config definitions
external/                      Vendored integrations for Gram and NeurIPT
tools/                         Lightweight analysis utilities
```

## Installation

```bash
conda create -n fewshot-efm python=3.10
conda activate fewshot-efm
pip install -r requirements.txt
```

Place pretrained model weights under `checkpoints/` locally. The default
filenames expected by `run_finetuning.py` are:

```text
labram-base.pth
pretrained_weights.pth
eegpt_mcae_58chs_4s_large4E.ckpt
EEG-six-datasets-18-channels.ckpt
CSBrain.pth
gram_base.pth
neuript_stage2.pth
```

Model weights are not committed to Git.

## Dataset Preparation

See `DATASETS.md` for dataset organization and split-generation details. In
short:

```bash
bash preprocessing/data_preprocess.sh /path/to/data/root TUEV
bash preprocessing/json_process.sh /path/to/data/root TUEV cross
```

Generated files such as `preprocessing/TUEV/cross_subject_json/train.json`
contain local filesystem paths. They are intentionally ignored by Git and
should be regenerated on each machine.

## TUEV Split Integrity

The original Ada-style TUEV preprocessing can leave validation samples
physically present in both `processed_data/train_dir` and
`processed_data/eval_dir`: validation files are copied from `train_dir` into
`eval_dir`, but the original copies are not removed from `train_dir`.

This repository avoids that leakage in the training path. For TUEV,
`preprocessing/TUEV/cross_json_process.py` builds validation from `eval_dir`
and excludes any validation basename from the generated training JSON:

```python
val_basenames = {f for f in os.listdir(val_folder) if f.endswith(".pkl")}
train_files = [
    os.path.join(train_folder, f)
    for f in os.listdir(train_folder)
    if f.endswith(".pkl") and f not in val_basenames
]
```

Use `run_finetuning.py`, which reads the generated JSON indexes through
`CustomDataLoader` or `FewShotDataLoader`. Do not train TUEV by directly
enumerating `processed_data/train_dir` and `processed_data/eval_dir`, because
that bypasses the JSON-level de-overlap step.

For other datasets, the same rule applies at the JSON level: train, validation,
and test should not contain the exact same file path or basename. Some datasets
may use validation windows from training subjects; subject overlap alone is not
the same as file-level leakage.

## Running

Full fine-tuning example:

```bash
python run_finetuning.py \
  --dataset TUEV \
  --model_name LaBraM \
  --task_mod Classification \
  --subject_mod fewshot \
  --finetune_mod full \
  --k_shot 0.05 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --norm_method z_score \
  --sampling_rate 200 \
  --seed 0
```

LoRA with functional-block diagnostics:

```bash
python run_finetuning.py \
  --dataset TUEV \
  --model_name LaBraM \
  --task_mod Classification \
  --subject_mod fewshot \
  --finetune_mod lora \
  --k_shot 0.05 \
  --epochs 30 \
  --batch_size 16 \
  --lr 1e-4 \
  --loss_type sqrt_balanced_ce \
  --best_metric balanced_accuracy \
  --lora_target semantic \
  --lora_base_update full \
  --lora_rank 4 \
  --lora_alpha 8 \
  --fb_enable \
  --fb_probe \
  --fb_recipe sem_lif \
  --fb_collect
```

Module-C task-aligned B/D/E search:

```bash
python run_finetuning.py \
  --dataset TUEV \
  --task_mod Classification \
  --model_name LaBraM \
  --subject_mod fewshot \
  --k_shot 0.05 \
  --finetune_mod lora \
  --lora_target module_c \
  --module_c_candidates B,D,E \
  --module_c_preflight_train_batches 0 \
  --module_c_preflight_val_batches 0
```

The default `0/0` scope uses the complete formal-visible support epoch and the
complete validation split. Support uses `SequentialSampler` with
`drop_last=True`, so Ada excludes the raw tail exactly as formal training does;
validation uses `SequentialSampler` with `drop_last=False` and covers every
example once. Each visited subset receives the same matched support pass, and
ranking uses paired, subject-clustered validation log-loss. Add
`--module_c_preflight_only` to write the decision and timing diagnostics without
starting formal training.

## Reproducibility Notes

- Keep raw datasets and processed data local.
- Regenerate split JSON files locally after preprocessing.
- Keep checkpoints and experiment outputs outside Git.
- Use validation metrics for adapter selection. Do not select hyperparameters
  from test performance.
- When reporting TUEV results, state that validation basenames are excluded
  from the generated training JSON.

## Attribution

Few-shot EFM builds on AdaBrain-Bench for the baseline training framework,
dataset interface, and several backbone integrations. Please acknowledge the
original AdaBrain-Bench project when using this repository. The additional code
in this fork focuses on few-shot EEGFM adaptation, functional-block LoRA
diagnostics, and adapter lifecycle evaluation.
