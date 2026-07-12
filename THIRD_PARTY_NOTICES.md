# Third-party notices

Few-shot EFM is a research fork and includes or adapts code from upstream
research projects. The repository-level MIT license does not replace any
upstream notice, dataset agreement, model-weight license, or citation request.

## AdaBrain-Bench

- Source: https://github.com/Jamine-W/AdaBrain-Bench
- Local scope: baseline training/evaluation framework, dataset interfaces,
  preprocessing foundations, and several backbone wrappers.
- Local changes: few-shot adaptation modules, functional-block LoRA,
  validation-only selection, safety checks, portable tooling, and tests.

The root `LICENSE` retains the inherited MIT notice. Users should cite the
AdaBrain-Bench project and the underlying backbone papers as appropriate.

## Gram

- Source: https://github.com/iiieeeve/Gram
- Local source: `external/Gram/`
- Original project information and paper citation:
  `external/Gram/README_Gram_original.md`
- Local use: selected model/configuration utilities required by the Gram adapter
  in `models/gram_ada.py`.

Only the integration subset needed by this project is present. Upstream
instructions that refer to files outside this subset are historical reference,
not Few-shot EFM launch instructions.

The upstream repository did not expose a standalone repository license when
this notice was prepared. Individual imported Microsoft-derived files retain
their MIT notices. This notice does not assert a broader license for unmarked
Gram source; users who redistribute that subset should verify terms with the
upstream authors.

## NeurIPT

- Local source: `external/NeurIPT/`
- Original project information: `external/NeurIPT/README_NeurIPT_original.md`
- Retained license: `external/NeurIPT/LICENSE`
- Local use: selected model and utility code required by
  `models/neuript_ada.py`.

Only the integration subset needed by this project is present. The original
README may mention assets or runners that are not included in this repository.

## Other backbone implementations

`run_finetuning.py` and model wrappers acknowledge additional upstream code
bases, including:

- LaBraM: https://github.com/935963004/LaBraM
- EEGPT: https://github.com/BINE022/EEGPT (Apache-2.0; local copy at
  `licenses/Apache-2.0.txt`)
- CBraMod: https://github.com/wjq-learning/CBraMod
- BIOT: https://github.com/ycq091044/BIOT

`models/loss.py` contains Salesforce-derived BSD-3-Clause code; its license is
retained at `licenses/BSD-3-Clause-Salesforce.txt`.

Pretrained model weights are not distributed here. Obtain them from their
official sources and follow the corresponding terms and citation requirements.

## Datasets

No raw dataset is distributed in this repository. Users are responsible for
obtaining each dataset from its official provider and complying with access,
privacy, redistribution, and citation terms.
