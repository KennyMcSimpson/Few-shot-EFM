# Module C Exhaustive Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Module C's path-dependent statistical selector with matched exhaustive one-pass evaluation of every nonempty B/D/E subset and a cached explanation certificate.

**Architecture:** Keep the existing disposable-model, common-head-anchor, action-ownership, optimizer-parity, RNG-restoration, and formal-rebuild infrastructure. Replace the risk-policy layer with a small deterministic exhaustive policy that ranks branch summaries by sample-level class-macro log-loss and derives conditional/factorial diagnostics from the same cache.

**Tech Stack:** Python 3, PyTorch, `unittest`, JSON/CSV diagnostics, Git worktrees.

## Global Constraints

- Do not modify the active `codex/core-training-control-fixes` worktree.
- Preserve concurrent E/A/training-control commits and integrate their latest stable commit before publication.
- Evaluate exactly `EMPTY,B,D,E,BD,BE,DE,BDE` for the default registry.
- Never select `EMPTY`; never delete or retrain after global selection.
- Use one complete sequential support pass with `drop_last=False` and complete validation.
- Use no subject metadata, statistical thresholds, weighted scores, or action-specific ranking metric.
- Preserve unrelated local Ada and remote dirty files.

---

### Task 1: Freeze the exhaustive policy contract

**Files:**
- Create: `tests/test_module_c_exhaustive_policy.py`
- Modify: `tests/test_module_c_preflight_smoke.py`
- Modify: `tests/test_module_c_runner_contract.py`

**Interfaces:**
- Produces: `enumerate_action_subsets(candidate_order)`, `select_exhaustive_subset(branches, candidate_order)`, and an explanation payload consumed by preflight diagnostics.

- [ ] Write tests asserting canonical enumeration yields seven nonempty subsets for B/D/E and no duplicates.
- [ ] Write tests asserting the global macro-loss minimum wins even when the old singleton-first route would reject it.
- [ ] Write tests asserting exact ties prefer fewer actions and no near-tie tolerance exists.
- [ ] Write tests asserting negative gain still returns the best nonempty subset with `forced_nonempty_best_observed`.
- [ ] Write tests asserting conditional contributions and pair/triple interactions use cached branch values.
- [ ] Update the smoke test to require all eight branches exactly once and no search stages or retired actions.
- [ ] Update the runner contract to require sequential `drop_last=False` Module C support and the new status field.
- [ ] Run the focused tests and confirm they fail because the new policy/API and exhaustive behavior are absent.

Run:

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
C:\Users\Kenny\.conda\envs\EEG\python.exe -m unittest tests.test_module_c_exhaustive_policy tests.test_module_c_preflight_smoke tests.test_module_c_runner_contract
```

Expected RED: import/assertion failures naming the missing exhaustive policy and old seven-branch/path-dependent behavior.

### Task 2: Implement deterministic exhaustive ranking

**Files:**
- Create: `util/module_c_exhaustive_policy.py`
- Delete: `util/module_c_risk_policy.py`
- Delete: `tests/test_module_c_risk_policy.py`

**Interfaces:**
- `enumerate_action_subsets(candidate_order: Sequence[str]) -> Tuple[Tuple[str, ...], ...]`
- `select_exhaustive_subset(branches: Mapping[Tuple[str, ...], BranchRisk], candidate_order: Sequence[str]) -> ExhaustiveDecision`
- `build_explanation_certificate(...) -> Mapping[str, Any]`

- [ ] Implement canonical power-set enumeration with combinations of sizes `1..m`.
- [ ] Implement lexicographic ranking by macro loss, subset size, adapter count, and canonical order.
- [ ] Implement positive/forced-nonempty status from the winner's raw EMPTY-referenced gain.
- [ ] Implement runner-up gap, per-class gains, selected-action conditional contributions, all pair interactions, and the B/D/E triple interaction.
- [ ] Keep the implementation free of tolerances, p-values, subject IDs, or additional evaluations.
- [ ] Run Task 1 tests and confirm GREEN.

### Task 3: Replace the preflight search loop

**Files:**
- Modify: `util/module_c_preflight_policy.py`
- Modify: `util/module_c_lora_search.py`

**Interfaces:**
- Consumes: exhaustive policy functions from Task 2.
- Produces: `module_c_preflight_scores.csv` with one row per nonempty candidate and `module_c_preflight_decision.json` containing branch and explanation certificates.

- [ ] Remove subject metadata lookup and `module_c_risk_policy` imports.
- [ ] Extend `_BranchEvaluation` with micro loss and preserve per-example/per-class losses.
- [ ] Evaluate `EMPTY`, then every canonical nonempty subset exactly once through the existing cache.
- [ ] Select globally from cached summaries and write the new status, runner-up, gap, gains, contributions, and interactions.
- [ ] Remove search steps, primary/final evidence, retired actions, and deletion/rescue wording.
- [ ] Keep head anchoring, ownership validation, E controller, formal optimizer schedule, RNG restoration, and formal-state isolation unchanged.
- [ ] Run all Module C tests and fix only regressions caused by the new contract.

### Task 4: Align loaders, CLI, runner, and documentation

**Files:**
- Modify: `run_finetuning.py`
- Modify: `util/fb_policy.py`
- Modify: `README.md`
- Delete: tracked Module C `.sh` and `.bat` runners
- Create: `experiment_manifests/module_c_exhaustive_seed0_4datasets.json`
- Modify: `tests/test_module_c_lora_interfaces.py`
- Modify: `tests/test_module_c_runner_contract.py`

**Interfaces:**
- Module C support loader is deterministic and no-drop; formal training loader remains random and `drop_last=True`.

- [ ] Change only `_make_module_c_preflight_loaders` support to `drop_last=False`.
- [ ] Update CLI help and metadata names to exhaustive low-fidelity subset search.
- [ ] Store the public experiment matrix and `selection_status` artifact contract in a JSON manifest; keep machine-specific runners outside Git.
- [ ] Update README/spec references and remove subject/Holm/hierarchical wording from active paths.
- [ ] Run runner dry-run and parser tests without starting training.

### Task 5: Verify, review, and synchronize

**Files:**
- Back up every touched local Ada `.py` file under `backup/moduleC_exhaustive_<timestamp>/`.
- Synchronize only reviewed tracked C files to local Ada and remote after GitHub publication.

- [ ] Fetch the concurrent branch and integrate its latest stable non-C changes; resolve overlaps by preserving its formal-training/E-controller fixes and this branch's C search semantics.
- [ ] Run focused tests, complete Module C tests, `py_compile`, runner syntax/dry-run, and a no-training toy preflight.
- [ ] Dispatch an independent code-review agent against the base/head diff and fix Critical/Important findings.
- [ ] Commit and push the C branch, then update GitHub main without force-push after re-fetching origin.
- [ ] Back up and copy touched files to local Ada without replacing unrelated files.
- [ ] Pull or copy the same reviewed commit to remote without cleaning dirty outputs or non-C files.
- [ ] Verify SHA-256 parity for every synchronized file and rerun no-training checks on all three code locations.
- [ ] Commit a final EEG project-memory checkpoint with commit IDs, backups, tests, risks, and next experiment step.
