# Few-shot EFM

Few-shot EFM is a research codebase for adapting EEG foundation models when
only a small labelled support set is available. It extends
[AdaBrain-Bench](https://github.com/Jamine-W/AdaBrain-Bench) with model-aware
functional-block LoRA, validation-only adapter selection, and a reproducible
Module C search over the B/D/E adaptation actions.

This repository is a research fork, not a clean-room reimplementation. The
training loop, dataset interface, preprocessing foundations, and several model
wrappers originate from AdaBrain-Bench. The project-specific additions and the
vendored integrations are documented in
[Architecture](docs/architecture.md) and
[Third-party notices](THIRD_PARTY_NOTICES.md).

## What is included

- Eleven training backbones: LaBraM, CBraMod, EEGPT, BIOT, CSBrain, Gram,
  NeurIPT, EEGNet, LMDA, EEGConformer, and ST-Transformer.
- Eight datasets connected to the training registry: BCI-IV-2A, EEGMAT, HMC,
  SEED-IV, SEED-VIG, Siena, Sleep-EDF, and TUEV.
- Five additional preprocessing-only integrations: SEED, SHHS, SHU, TUAB, and
  Things-EEG. They are not advertised as trainable until a dataset config is
  added and validated.
- Full fine-tuning, linear probing, and LoRA adaptation.
- Functional-block diagnostics and Modules A-E for few-shot adapter control.
- Portable Python entrypoints for dataset preparation and experiment
  manifests. The public tree intentionally contains no batch or shell runners.

Raw datasets, generated split indexes, checkpoints, logs, and result folders
are intentionally excluded from Git. Released artifacts belong in the
[Few-shot EFM Hugging Face collection](https://huggingface.co/datasets/KennySimpson/few-shot-EFM).

## Repository map

```text
run_finetuning.py          training and evaluation entrypoint
engine_for_finetuning.py   epoch loops, metrics, and lifecycle hooks
module_a_lifecycle.py      validation-only checkpoint/lifecycle selection
models/                    model wrappers and local backbone integrations
util/                      LoRA plus Modules B-E and diagnostics
preprocessing/             dataset-specific Python preprocessors
dataset_config/            training-visible dataset metadata
experiment_manifests/      portable experiment matrices
tools/                     dataset and manifest command-line utilities
external/                  isolated Gram and NeurIPT source integrations
tests/                     unit and repository-contract tests
docs/                      architecture, data, and reproducibility guides
```

## Installation

Python 3.10 is the reference environment.

```console
conda create -n fewshot-efm python=3.10
conda activate fewshot-efm
pip install -r requirements.txt
```

The pinned PyTorch build targets CUDA 11.8. Choose the matching official
PyTorch wheel for a different CUDA or CPU platform. DeepSpeed is optional and
Linux-only in this project:

```console
pip install -r requirements-optional.txt
```

Keep pretrained weights in a local `checkpoints/` directory. The default
filenames are defined by `finetune_list` in `run_finetuning.py`; weights are not
stored in this repository.

## Dataset preparation

The cross-platform dataset utility discovers the checked-in Python
preprocessors and separately reports whether each dataset is training
configured:

```console
python tools/dataset_cli.py list
python tools/dataset_cli.py prepare TUEV /path/to/eeg
python tools/dataset_cli.py split TUEV /path/to/eeg --mode cross
python tools/dataset_cli.py audit preprocessing/TUEV/cross_subject_json
```

The audit checks exact-path and basename overlap across `train.json`,
`val.json`, and `test.json`. Read [Dataset support](docs/datasets.md) before
using a new dataset; preprocessing availability and end-to-end training support
are deliberately reported as different states.

## Training examples

Full fine-tuning:

```console
python run_finetuning.py --dataset Sleep-EDF --model_name LaBraM \
  --task_mod Classification --subject_mod fewshot --finetune_mod full \
  --k_shot 0.05 --epochs 30 --batch_size 16 --lr 1e-4 --seed 0
```

Functional-block LoRA:

```console
python run_finetuning.py --dataset SEED-IV --model_name EEGPT \
  --task_mod Classification --subject_mod fewshot --finetune_mod lora \
  --k_shot 0.05 --lora_target semantic --lora_rank 4 --lora_alpha 8 \
  --fb_enable --fb_probe --fb_collect --seed 0
```

Module C exhaustive B/D/E preflight:

```console
python run_finetuning.py --dataset BCI-IV-2A --model_name BIOT \
  --task_mod Classification --subject_mod fewshot --finetune_mod lora \
  --k_shot 0.05 --lora_target module_c --module_c_candidates B,D,E \
  --module_c_preflight_train_batches 0 --module_c_preflight_val_batches 0 \
  --module_c_preflight_only
```

Module C evaluates all seven non-empty subsets with the same complete support
pass and complete validation pass. `EMPTY` is a reference only. The selected
subset minimizes sample-level class-macro validation log-loss; exact ties prefer
fewer actions, fewer adapter parameters, then canonical B/D/E order. Module C
selection is intentionally fail-closed when `world_size` is greater than one.

The checked 4-dataset, 6-model seed-0 matrix can be previewed or executed without
machine-specific runners:

```console
python tools/run_manifest.py experiment_manifests/module_c_exhaustive_seed0_4datasets.json
python tools/run_manifest.py experiment_manifests/module_c_exhaustive_seed0_4datasets.json --execute
```

Preview is the default. Training begins only when `--execute` is supplied.

## Documentation

- [Architecture and contribution boundary](docs/architecture.md)
- [Dataset support and split integrity](docs/datasets.md)
- [Reproducibility and experiment manifests](docs/reproducibility.md)
- [Adding a backbone](docs/adding-a-backbone.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Status and limitations

This is active research software. Passing unit tests establish interface and
protocol invariants; they do not substitute for dataset-level scientific
validation. The checked Module C manifest currently covers TUEV, Sleep-EDF,
BCI-IV-2A, and SEED-IV. Other training-configured datasets remain available for
the general training path but are not implicitly claimed as part of that
specific Module C matrix.

## License and attribution

Project-authored changes and the inherited AdaBrain-Bench framework are
distributed under the [MIT License](LICENSE). Bundled third-party portions keep
their own notices and terms. Retain upstream copyright notices and cite the
original model and dataset publications used in your experiments. See
[Third-party notices](THIRD_PARTY_NOTICES.md) for source boundaries, license
copies, and upstream links.
