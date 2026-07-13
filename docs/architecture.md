# Architecture and contribution boundary

## Project layers

Few-shot EFM is organized as four layers:

1. `run_finetuning.py` and `engine_for_finetuning.py` own training, evaluation,
   checkpointing, and the command-line contract.
2. `models/` and `external/` adapt EEG backbones to a common task-head interface.
3. `util/lora.py`, `util/fb_*.py`, and `util/module_*.py` implement functional
   regions, adapter injection, diagnostics, and adaptation controls.
4. `preprocessing/`, `dataset_config/`, `tools/`, and `experiment_manifests/`
   connect local datasets to reproducible runs.

## Upstream and project-specific code

The baseline fine-tuning framework, dataset loader conventions, preprocessing
foundations, and several backbone wrappers were inherited from
[AdaBrain-Bench](https://github.com/Jamine-W/AdaBrain-Bench). Few-shot EFM adds
or substantially extends:

- functional-block classification and LoRA target resolution;
- Module A validation-only lifecycle and adaptive checkpoint aggregation;
- Module B input/signal alignment actions;
- Module C exhaustive, matched B/D/E subset preflight;
- Module D semantic-boundary diagnostics and refinement;
- Module E structural-routing pressure control;
- safe resume checks for optimizer-schema compatibility;
- portable dataset and experiment-manifest command-line tools;
- unit tests for adaptation semantics, training safety, and public-repository
  contracts.

`external/Gram` is an isolated third-party source integration. It should not be
described as original Few-shot EFM model code. See `THIRD_PARTY_NOTICES.md` and
the original README retained in that directory.

## Adaptation modules

| Module | Responsibility | Primary code |
| --- | --- | --- |
| A | validation-only adapter lifecycle and checkpoint selection | `module_a_lifecycle.py` |
| B | input/signal alignment action | `util/module_b_signal_alignment.py` |
| C | matched exhaustive search over B/D/E subsets | `util/module_c_*.py` |
| D | semantic-boundary refinement action | `util/module_d_*.py` |
| E | spatial/temporal/mixing routing recalibration | `util/module_e_structural_routing.py` |

Module labels are action roles, not model names. For example, Module E is a
general structural-routing action. Model-specific registries only map that
general role onto the actual spatial, temporal, or mixing parameters exposed by
each backbone; a BIOT mapping does not create a separate “BIOT E”.

## B/D backbone contract

`util/backbone_contracts.py` is the formal source of truth for Module B bridge
sites and Module D semantic FFN sites. The scientific actions remain shared;
the contract only translates those actions into each backbone's real module
paths and expected PyTorch types.

| Backbone | B input residual | Optional B bridge | D semantic FFN |
| --- | --- | --- | --- |
| BIOT | raw EEG | `chan_conv` | `w1`, `w2` |
| EEGPT | raw EEG | `chan_conv` | `mlp.fc1`, `mlp.fc2` |
| LaBraM | raw EEG | none | `mlp.fc1`, `mlp.fc2` |
| CBraMod | raw EEG | none | `linear1`, `linear2` |
| CSBrain | raw EEG | conditional `chan_conv` | `linear1`, `linear2` |
| Gram | raw EEG | none | encoder `blocks.*.mlp.fc1/fc2` |

Missing required sites and unexpected module types fail before training.
`diagnostics/backbone_bd_contract.json` records the contract hash and the
resolved/injected B/D sites for each LoRA run.

The formal method uses `--lora_base_update full`: all original trainable
backbone parameters remain trainable and selected B/D/E sites receive
additional LoRA residuals. LoRA runs must state `full` or `freeze` explicitly;
there is no silent base-update default. The optimizer is checked to contain
every trainable parameter exactly once and no frozen parameters.

## Module C protocol

With candidates `B,D,E`, Module C scores `EMPTY`, `B`, `D`, `E`, `BD`, `BE`,
`DE`, and `BDE`. Every branch is restored to the same anchored model state and
receives a matched sequential support pass. Evaluation covers the full
validation split with `drop_last=False`. Selection uses validation labels only;
test data is outside this decision.

`EMPTY` is a reference baseline and cannot win. Among non-empty branches,
selection minimizes sample-level class-macro validation log-loss. Exact ties are
resolved by fewer actions, fewer injected adapter parameters, then canonical
B/D/E order.

The preflight uses the same Module E controller lifecycle as formal training:
the controller is bound to the optimizer, prepares each optimizer step, and is
notified whether an AMP-scaled step was actually applied. This keeps dynamic
pressure state aligned with real parameter updates.

Module C topology selection is fail-closed for distributed execution. Run the
selection with one process, persist its decision artifact, and launch any later
distributed training from that fixed decision. This avoids workers silently
making different architecture choices.

## Checkpoint boundary

Resume safety is stricter than a normal weight-only load. A checkpoint can carry
model weights, optimizer parameter groups and state, scheduler/scaler state,
epoch position, and adaptation-controller state. If the selected adapter set
changes the optimizer schema, the run fails clearly before loading incompatible
optimizer state. Use a weights-only initialization path when intentionally
changing the trainable topology; do not present that as an exact resume.

## Extension rules

- Add dataset training support through `dataset_config/`; a preprocessing script
  alone is not an end-to-end integration.
- Keep model-specific tensor reshaping inside its wrapper.
- Register B/D targets in `util/backbone_contracts.py`, keep structural routing
  in the Module E implementation, and test the exact resolved parameter set.
- Keep generated data, checkpoints, and results outside Git.
- Add protocol tests whenever a change affects selection, split integrity,
  optimizer stepping, or checkpoint restoration.
