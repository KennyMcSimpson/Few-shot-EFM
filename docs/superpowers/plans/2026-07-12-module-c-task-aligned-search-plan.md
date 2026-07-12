# Module C Task-Aligned Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Module C's random-head first-order proxy with matched low-budget B/D/E subset search, verify it locally and on GitHub, then deploy and start four dataset seed-0 remote runs.

**Architecture:** Pure policy/statistics live in a focused Module C policy file. The preflight orchestrator owns head anchoring, matched one-pass subset trials, validation loss collection, action ownership, and diagnostics. `run_finetuning.py` owns outer RNG isolation, formal model reconstruction, and preflight-only exit.

**Tech Stack:** Python 3, PyTorch, NumPy, SciPy, unittest, Git worktrees, PowerShell, remote Linux screen/bash.

## Global Constraints

- Final candidates are only nonempty subsets of the action registry; currently B/D/E.
- Module A and qv are never Module C candidates.
- Test labels never enter search.
- Formal B/D/E hyperparameters and interfaces are reused unchanged.
- Local active FullFT processes and remote retained patches/outputs must not be interrupted or overwritten.
- TDD red-green evidence is required for every production behavior change.

---

### Task 1: Pure Paired-Risk Policy

**Files:**
- Replace: `util/module_c_risk_policy.py`
- Modify: `tests/test_module_c_risk_policy.py`

**Interfaces:**
- Produces `PairedRiskEvidence`, `ActionTrial`, `SearchDecision`, `cluster_jackknife_evidence(...)`, `holm_adjust(...)`, and `choose_action(...)`.
- Consumes subject-class paired loss differences and adapter parameter counts; contains no Torch model code.

- [x] Write failing tests for subject/class macro gain, cluster jackknife uncertainty, Holm adjustment, safe primary selection, weak nonempty fallback, tie-to-smaller action, conditional addition, alternative-pair rescue, and the absence of backward deletion.
- [x] Run `python -m unittest tests.test_module_c_risk_policy -v` and confirm failures are missing new interfaces or old policy behavior.
- [x] Implement the minimum pure policy and rerun until all policy tests pass.
- [x] Remove old first-order effect, exact-zero gate, additive independent-effect combination, and `forced_nonempty_least_harm` semantics.

### Task 2: Candidate Ownership Audit

**Files:**
- Modify: `util/module_c_preflight_policy.py`
- Modify: `util/lora.py` only if BIOT or another model cannot expose disjoint actions
- Modify: `tests/test_module_c_lora_interfaces.py`

**Interfaces:**
- Produces `ActionOwnership` with action-to-parameter names/counts and fatal validation of zero, overlap, and unowned adapters.
- Consumes the existing action registry and `apply_lora_to_eegfm(...)` output.

- [x] Write failing interface tests asserting B/D/E are nonempty and pairwise disjoint for each supported model fixture, with BIOT `to_q/to_v` owned by E and FFN owned by D.
- [x] Run the focused interface tests and confirm missing ownership interfaces.
- [x] Implement registry-based ownership and fail-fast validation.
- [x] Rerun interface and policy tests.

### Task 3: Head Anchor and Matched Branch Probes

**Files:**
- Replace: `util/module_c_preflight_policy.py`
- Modify: `tests/test_module_c_preflight_smoke.py`

**Interfaces:**
- Produces `run_module_c_preflight_selection(...) -> ModuleCPreflightResult` with branch traces and class/subject evidence.
- Consumes model builder callback, support/validation datasets, formal optimizer configuration, candidate actions, and criterion builder.

- [x] Write failing smoke tests proving the head is trainable before action trials, matched reference/candidate branches see identical support data, direct validation loss replaces `-<g,delta>`, and conditional combinations are evaluated rather than summed. The self-review removed the brittle uniform-loss hard gate and retained it as a diagnostic.
- [x] Run the smoke tests and confirm old zero-update implementation fails them.
- [x] Implement one-pass head anchoring, branch snapshot/restore, formal trainability controls, complete support scans, per-example validation losses, and subject metadata grouping.
- [x] Implement dynamic forward additions and singleton-only alternative-pair rescue using the pure policy; remove backward deletion.
- [x] Write CSV/JSON diagnostics containing head behavior, branch budgets, per-class effects, confidence evidence, search trace, ownership, and evidence strength.
- [x] Rerun all Module C tests.

### Task 4: RNG Isolation and Preflight-Only Execution

**Files:**
- Modify: `run_finetuning.py`
- Modify: `util/fb_policy.py`
- Modify: `tests/test_module_c_preflight_smoke.py`

**Interfaces:**
- Produces `--module_c_preflight_only`, restores all random states before formal model construction, and exits successfully after writing preflight artifacts when requested.

- [x] Write failing tests for parser exposure and RNG restoration.
- [x] Run the tests and confirm failure under the old parser behavior.
- [x] Add RNG snapshot/restore and the preflight-only path without changing non-C execution.
- [x] Retain batch caps only as explicitly marked debug controls.
- [x] Rerun Module C tests and `py_compile` on all touched Python files.

### Task 5: Integration Verification, Backup, and Publication

**Files:**
- Modify only verified Module C integration files and tests in the GitHub worktree.
- Back up touched local Ada `.py` files under `backup/moduleC_task_aligned_search_<timestamp>/`.

- [ ] Run the full focused unittest suite and no-training smoke tests in the GitHub worktree.
- [ ] Run `git diff --check`, inspect the diff, and commit the isolated branch.
- [ ] Fast-forward or merge the verified branch into GitHub `main` and push.
- [ ] Copy only committed touched files to local Ada after backups; rerun tests there and compare SHA-256 hashes against GitHub.
- [ ] Preserve active local processes and verify they remain alive after file synchronization.

### Task 6: Remote Patch Preservation and Seed-0 Queue

**Files:**
- Create a new remote runner for the new C method; do not reuse the stopped validation-risk queue as-is.
- Preserve remote modifications to `models/gram_ada.py`, dataset path scripts, and remote-only output/symlink files.

- [ ] Record remote dirty files and back up any remotely modified tracked file that overlaps the new commit.
- [ ] Fetch GitHub and update tracked source files without resetting unrelated remote changes.
- [ ] Run remote `py_compile`, focused Module C tests, parser smoke, one preflight-only real-model smoke, and dry-run all planned commands.
- [ ] Build a seed-0 queue for TUEV, Sleep-EDF, BCI-IV-2A, and SEED-IV across the supported EEG foundation models.
- [ ] Start with three stable lanes on CPU ranges that avoid the previously unstable `0-7` range; inspect GPU memory, process health, and first decision artifacts before increasing concurrency.
- [ ] Persist `run_status.csv`, per-job logs, output roots, exact Git commit, and command lines. Do not start three-seed expansion.

## Plan Self-Review

- Every design requirement maps to a task.
- No old weighted, RGFS, or validation-risk selector remains as a second active path.
- The runtime path is bounded and emits enough evidence to distinguish strong, weak, and mandatory selections.
- The remote step preserves dirty patches and starts only after local/GitHub/remote verification.
