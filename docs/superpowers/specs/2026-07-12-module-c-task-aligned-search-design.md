# Module C Task-Aligned Search Design

## Goal

Replace the current one-step validation-risk proxy with a disposable, low-budget, task-aligned search that returns a nonempty subset of functional actions. The current action registry exposes B, D, and E, but the search must remain registry-driven rather than hard-code seven subsets.

## Failure Being Repaired

The current selector freezes a newly initialized task head, derives one AdamW virtual update per action, and ranks `-<g_val, delta>`. This makes the decision depend on a random decision boundary and on the number of active adapter coordinates. All inspected new runs consequently failed the exact-zero safety gate and selected B through `forced_nonempty_least_harm`. This is not acceptable evidence that B is genuinely safest.

## Search Contract

1. Build a disposable model from the same pretrained checkpoint used by formal training.
2. Audit every candidate action before probing. Every action must own a nonempty, disjoint adapter parameter set. Unowned or multiply owned parameters are fatal errors.
3. Anchor the real task head for one complete support pass while the pretrained feature extractor is frozen. Record its class-balanced validation cross-entropy before and after anchoring, together with the uniform reference `log(C)`, as diagnostics. These values are not hard selection gates: a short head pass can be weak on a difficult dataset, while all subsequent action branches are still matched from the same anchored state.
4. At each search state `S`, create one matched reference branch and one branch for every addition `S + a`. Every branch starts from the same state, sees the same complete support pass in the same order, and uses the formal optimizer, learning rate, weight decay, trainability policy, LoRA rank/alpha/dropout, task-head policy, and Module E controller settings. The only intended difference is the candidate action.
5. Compute paired per-example validation cross-entropy improvement between the matched reference and candidate branches. Aggregate windows within `(subject, class)`, average subjects within each class, then average classes. This produces a common action utility without giving any action a mechanism-specific bonus.
6. Estimate uncertainty with delete-one-subject cluster jackknife pseudo-values. Use one-sided `alpha=0.05` tests and Holm correction within each search stage. Window-level independence must never be claimed. Because sequential stages reuse the same validation split, these p-values are stability screens rather than confirmatory post-selection inference.
7. A supported addition has Holm-adjusted positive evidence for class-balanced gain and no Holm-adjusted evidence of harm for any observed class. Choose the largest point gain among supported actions, then the largest worst-class gain; adapter parameter count breaks only an exact evidence tie.
8. If no primary action is safe, still return one action because Module C is defined as nonempty. Prefer an action without supported class harm, then the largest class-balanced gain. If all actions show supported harm, maximize the worst class effect. Mark these outcomes as weak or mandatory evidence, never as predicted improvement.
9. After selecting a primary action, repeat matched one-action additions. If every one-action extension fails while the selected set is still a singleton, evaluate pairs formed only from the remaining actions directly against that singleton. A supported alternative pair replaces the singleton, retires the old action, and may then grow only through supported one-action additions. Once a multi-action path stalls, the search stops; it never deletes an accepted action or jumps directly to a higher-order set. Thus B/D/E can all be selected through a supported chain such as `B -> B+D -> B+D+E`, but not by jumping from failed `B+D` and `B+E` trials directly to `B+D+E`.
10. Discard every probe model, optimizer, gradient, and task-head state. Restore Python, NumPy, Torch CPU, Torch CUDA, sampler, and DataLoader random state. Rebuild formal training from the original pretrained checkpoint with only the selected actions.

## Metrics

For validation example `i`, action `a`, and current subset `S`:

`d[i,a|S] = loss_i(reference(S)) - loss_i(candidate(S+a))`.

For subject `u` and class `c`:

`d[u,c] = mean_{i in (u,c)} d[i,a|S]`.

Class gain and class-balanced gain are:

`Delta[a|S,c] = mean_{u containing c} d[u,c]`

`G[a|S] = mean_c Delta[a|S,c]`.

Positive values mean lower validation loss. Balanced accuracy, worst-class recall, recall standard deviation, and per-class recall remain diagnostics and do not rank candidates.

## Explainability Boundary

The selector uses only a common downstream risk measure for ranking. Module B signal-alignment diagnostics, Module D semantic-boundary diagnostics, and Module E structural-pressure diagnostics remain action-specific explanations and cannot alter the common score. Equal treatment means equal measurement, not equal selection frequency.

## Fixed Design Choices

- One support epoch per matched trial is the minimum complete data-exposure unit.
- The head anchor receives exactly one complete support pass; its validation behavior is reported rather than thresholded.
- `alpha=0.05` is fixed statistical error control.
- Three subject clusters are the minimum for reported jackknife evidence, giving at least two t-reference degrees of freedom; classes represented by fewer than two subjects cannot support a class-harm test.
- Interaction order two is the lowest order that represents synergy or conflict.
- All LoRA and B/D/E training hyperparameters come from formal training unchanged.
- Adapter parameter count is descriptive and an exact tie-break only. A weighted complexity penalty is deliberately excluded because it would add an uncalibrated coefficient and change the scientific target from downstream risk to a hand-designed score.
- There are no score weights, numerical safety epsilons, hard-class top-k values, fixed probe batch counts, qv candidates, or E bonuses.

## Runtime Bound

With B/D/E, the implementation uses one head-anchor pass plus at most seven distinct equal-budget subset branches on either the supported forward path or the singleton alternative-pair path. This is at most eight support passes, about 16 percent of a 50-epoch formal run and about 2.3 percent of seven 50-epoch subset runs. Full validation scans can dominate on Sleep-EDF and TUEV, so every decision records measured wall time and `--module_c_preflight_only` is used before formal queues.

## Validity Failures

Selection must fail explicitly when an expected class is absent from validation, an action owns zero parameters, action parameter sets overlap, subject grouping is unavailable, or a matched branch does not complete. A weak head anchor remains visible in diagnostics but does not silently redirect selection to B.

## Validation Before Formal Experiments

- Unit-test primary selection, weak nonempty fallback, supported forward additions, singleton alternative-pair rescue, absence of backward deletion, Holm correction, and binary classification.
- Smoke-test real B/D/E LoRA injection and disjoint ownership for every supported model, including independent BIOT E ownership.
- Verify preflight-only mode writes complete decision and branch diagnostics without entering formal training.
- Verify formal model initialization hashes are identical with and without preflight for the same seed.
- Run seed-0 preflight/formal smoke across TUEV, Sleep-EDF, BCI-IV-2A, and SEED-IV before any three-seed expansion.

## Claim Boundary

Module C is a low-fidelity validation-guided functional subset search. It estimates short-horizon marginal utility; it does not guarantee the 50-epoch test winner and must not be described as a zero-update or exact predictor.
