# Datasets

This repository expects datasets to stay local. Raw data, processed data, and
generated split JSON files are not committed to Git and should not be uploaded
to the Hugging Face result repository unless explicitly needed for a separate
release.

Use a single data root such as:

```text
$DATA/
  SEED/
  SEED-IV/
  EEGMAT/
  SEED-VIG/
  BCI-IV-2A/
  SHU/
  Things-EEG/
  TUEV/
  TUAB/
  Siena/
  HMC/
  SHHS/
  Sleep-EDF/
```

## Directory Conventions

Each dataset should have a `raw_data/` folder before preprocessing and a
`processed_data/` folder after preprocessing:

```text
$DATA/TUEV/raw_data/
$DATA/TUEV/processed_data/
```

Dataset-specific raw layouts follow the original dataset providers. The
preprocessing scripts under `preprocessing/<dataset>/` assume the same layouts
used by AdaBrain-Bench.

## Preprocessing

Run preprocessing from the repository root:

```bash
bash preprocessing/data_preprocess.sh /path/to/data/root TUEV
```

The first argument is the data root, and the second argument is the dataset
name. The script dispatches to:

```text
preprocessing/<dataset>/data_process.py
```

## Split JSON Generation

After preprocessing, generate local JSON split indexes:

```bash
bash preprocessing/json_process.sh /path/to/data/root TUEV cross
```

The third argument is the split mode:

```text
cross   -> preprocessing/<dataset>/cross_json_process.py
multi   -> preprocessing/<dataset>/multi_json_process.py
```

Not every dataset has both scripts. If a script is missing, choose the split
mode supported by that dataset.

Generated split JSON files look like:

```text
preprocessing/TUEV/cross_subject_json/train.json
preprocessing/TUEV/cross_subject_json/val.json
preprocessing/TUEV/cross_subject_json/test.json
```

These files contain local absolute or relative data paths. They are ignored by
Git and should be regenerated on the machine that runs the experiments.

## TUEV Leakage Note

TUEV needs special care. The original preprocessing pattern can copy validation
files from `processed_data/train_dir` to `processed_data/eval_dir` without
removing the original copies from `train_dir`. That creates a physical
train/eval overlap if one directly enumerates the processed folders.

The supported training path in this repository is JSON-based. The TUEV
`cross_json_process.py` script excludes validation basenames from the generated
training JSON. Therefore, use the generated JSON indexes with
`run_finetuning.py` instead of directly loading `processed_data/train_dir` and
`processed_data/eval_dir`.

## Split Integrity Policy

For train, validation, and test JSON files:

- exact `file` path overlap should be zero;
- basename overlap should be zero;
- test files should be held out from training and validation;
- subject overlap can be dataset/protocol-specific and is not automatically
  equivalent to file-level leakage.

The local Ada working copy was checked under this rule before this repository
cleanup: no tracked split JSON had exact file-path or basename overlap across
train, validation, and test.

## Dataset Config

`dataset_config/Classification.json` and `dataset_config/Regression.json`
define which generated split folder each dataset uses:

```json
{
  "TUEV": {
    "root": {
      "multi": "./preprocessing/TUEV/cross_subject_json",
      "cross": "./preprocessing/TUEV/cross_subject_json",
      "fewshot": "./preprocessing/TUEV/cross_subject_json"
    },
    "num_classes": 6,
    "num_t": 5
  }
}
```

Update these config files only when the split folder or dataset metadata
changes.
