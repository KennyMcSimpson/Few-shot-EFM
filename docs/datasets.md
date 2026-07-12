# Dataset support and split integrity

## Support levels

The repository uses two explicit support levels:

- **Training-configured** means the dataset has preprocessing code and an entry
  in `dataset_config/Classification.json` or `dataset_config/Regression.json`.
- **Preprocessing-only** means conversion or split-generation code is present,
  but the public training registry does not yet expose the dataset.

The current inventory is:

| Dataset | Task in training registry | Prepare | Split modes | Status |
| --- | --- | --- | --- | --- |
| BCI-IV-2A | Classification | yes | cross, multi | training-configured |
| EEGMAT | Classification | yes | cross | training-configured |
| HMC | Classification | yes | cross | training-configured |
| SEED | - | yes | cross, multi | preprocessing-only |
| SEED-IV | Classification | yes | cross, multi | training-configured |
| SEED-VIG | Regression | yes | cross | training-configured |
| SHHS | - | yes | cross | preprocessing-only |
| SHU | - | yes | cross, multi | preprocessing-only |
| Siena | Classification | yes | cross | training-configured |
| Sleep-EDF | Classification | yes | cross | training-configured |
| TUAB | - | yes | cross | preprocessing-only |
| TUEV | Classification | yes | cross | training-configured |
| Things-EEG | - | yes | multi | preprocessing-only |

Run `python tools/dataset_cli.py list` to derive this view from the current tree
instead of relying on a stale handwritten list.

## Local directory convention

Raw and processed data remain local. A typical root is:

```text
/data/eeg/
  TUEV/raw_data/
  TUEV/processed_data/
  Sleep-EDF/raw_data/
  Sleep-EDF/processed_data/
```

Dataset providers use different raw layouts. Each
`preprocessing/<dataset>/data_process.py` documents the layout it expects. The
BCI dataset is logically named `BCI-IV-2A` in the training interface and stored
under `preprocessing/BCI-4-2A` in this repository; the dataset utility resolves
that alias explicitly.

## Cross-platform commands

From the repository root:

```console
python tools/dataset_cli.py list
python tools/dataset_cli.py prepare Sleep-EDF /data/eeg
python tools/dataset_cli.py split Sleep-EDF /data/eeg --mode cross
python tools/dataset_cli.py audit preprocessing/Sleep-EDF/cross_subject_json
```

The utility launches the selected Python script directly with `shell=False`.
Unsupported dataset or split-mode combinations fail with the available choices
instead of constructing a path and hoping it exists.

Generated `train.json`, `val.json`, and `test.json` files may contain local
absolute paths and therefore must not be committed. Regenerate them on the
machine that runs the experiment.

## Split integrity policy

Before training, audit every generated split directory:

```console
python tools/dataset_cli.py audit preprocessing/TUEV/cross_subject_json
```

The audit requires zero overlap for:

- normalized sample paths between every split pair;
- sample basenames between every split pair.

Exact file overlap is leakage. Basename overlap is treated as a conservative
warning because copied files can have different parent directories. Subject
overlap is protocol-dependent and must be reported separately; it is neither
automatically safe nor equivalent to file-level overlap.

## TUEV-specific safeguard

The inherited TUEV preprocessing layout can leave validation samples physically
present in both `processed_data/train_dir` and `processed_data/eval_dir`.
`preprocessing/TUEV/cross_json_process.py` prevents that from entering the
supported training path by excluding validation basenames from the generated
training JSON.

Always train through the generated JSON indexes used by `CustomDataLoader` or
`FewShotDataLoader`. Direct enumeration of the two processed folders bypasses
the safeguard.

This note is prominent because TUEV has an extra inherited hazard, not because
TUEV is the only supported dataset. The same JSON-level audit applies to every
dataset.

## Adding a training dataset

1. Add or validate the dataset-specific prepare and split scripts.
2. Generate and audit all three split JSON files.
3. Add the task metadata, split root, class count, and segment length to the
   appropriate file in `dataset_config/`.
4. Verify loader shapes and labels with the intended subject protocol.
5. Add tests that distinguish preprocessing discovery from training-registry
   support.

Do not label a dataset end-to-end supported solely because a directory exists
under `preprocessing/`.
