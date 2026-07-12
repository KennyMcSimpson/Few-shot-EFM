# Reproducibility

## Source and environment

Record the Git commit, Python version, PyTorch/CUDA versions, device type, and
the exact pretrained checkpoint identity. A filename alone is not a stable
checkpoint identifier; retain its source and checksum in the experiment record.

The public repository excludes raw data, generated splits, weights, and results.
Released artifacts should be stored separately and linked back to the source
commit that produced them.

## Data protocol

For every run, record:

- dataset release and preprocessing parameters;
- subject mode and split-generation mode;
- support fraction or shot count and random seed;
- train/validation/test sample counts;
- exact-path, basename, and subject-overlap audit results;
- normalization and sampling rate.

Adapter and checkpoint selection must use validation data only. Test metrics are
for final evaluation and must not feed Module A, Module C, early stopping, or
hyperparameter choice.

## Portable experiment manifests

`experiment_manifests/module_c_exhaustive_seed0_4datasets.json` is the checked
seed-0 Module C protocol over four datasets and six models. It contains only
repository-relative or logical values. Preview all expanded commands with:

```console
python tools/run_manifest.py experiment_manifests/module_c_exhaustive_seed0_4datasets.json
```

Filter without editing the manifest:

```console
python tools/run_manifest.py experiment_manifests/module_c_exhaustive_seed0_4datasets.json \
  --datasets TUEV,Sleep-EDF --models LaBraM,BIOT --seeds 0
```

Execution is opt-in:

```console
python tools/run_manifest.py experiment_manifests/module_c_exhaustive_seed0_4datasets.json \
  --output-root finetuning_results --execute
```

Runs execute sequentially with `shell=False`. `--continue-on-error` is available
for matrix sweeps, but failed entries must remain visible in the experiment log.

## Module C record

The Module C decision artifact should retain at least:

- selected and runner-up module subsets;
- selection status, gap, and observed gain;
- every branch score and adapter-parameter count;
- support and validation coverage;
- branch count and timing;
- dataset, model, seed, and anchored checkpoint identity.

The checked default `0/0` batch limits mean complete support and validation
passes, not zero data. Both passes are sequential and use `drop_last=False`.

Perform architecture selection with a single process. Distributed workers may
consume a fixed persisted decision later, but this repository intentionally
does not allow topology selection to proceed independently on multiple workers.

## Resume semantics

An exact resume requires compatible model topology, trainable parameters,
optimizer parameter groups and state, scheduler/scaler state, epoch position,
and active adaptation-controller state. The code validates the optimizer schema
before loading a legacy optimizer state. A mismatch is a deliberate failure,
not a reason to silently discard state.

When changing the adapter subset or other trainable topology, start a new run
from weights and document it as initialization from a prior checkpoint rather
than an exact resume.

## Verification

Run the repository test suite from its root:

```console
python -m unittest discover -s tests -p "test_*.py"
```

Also compile the public Python tree and preview the manifest before launching an
expensive run. Unit tests verify software contracts; they do not establish that
every model/dataset combination has completed a formal scientific experiment.
