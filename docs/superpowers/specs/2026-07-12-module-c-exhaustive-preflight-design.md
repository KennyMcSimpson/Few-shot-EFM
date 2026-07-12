# Module C Exhaustive Preflight Design

## Scope

Module C chooses a nonempty subset of the finalized B, D, and E actions before formal training. The search evaluates `B`, `D`, `E`, `B+D`, `B+E`, `D+E`, and `B+D+E` directly. `EMPTY` is measured only as an explanatory reference and is never a selectable candidate.

This version is deliberately scoped to three binary actions. It does not claim scalable neural architecture search for a larger registry.

## Matched Probe Protocol

1. Build one disposable pretrained model and anchor its exposed task head with one support pass.
2. Install the union of B/D/E probe adapters once and verify disjoint action ownership.
3. Snapshot the anchored union model and RNG state.
4. For `EMPTY` and every nonempty subset, restore the same snapshot and RNG state, apply the same formal optimizer and epoch-zero schedule, train on the same sequential support batches exactly once, and evaluate the same complete validation examples.
5. Use a Module C support loader with `SequentialSampler` and `drop_last=False`, so a zero batch cap means every support example is visible once.
6. Discard every probe state, restore RNG, rebuild the formal model, and inject only the selected subset.

Module E's pressure controller remains part of E whenever the formal E configuration requests it. It is not an E-specific score bonus.

## Selection Risk

For subset `S`, compute per-example negative log likelihood and then:

```text
class_loss[c, S] = mean(loss_i[S] for y_i == c)
macro_loss[S] = mean(class_loss[c, S] for every observed class c)
micro_loss[S] = mean(loss_i[S] for every validation example i)
```

The selected subset is the nonempty subset with the smallest `macro_loss`. Exact macro-loss ties prefer fewer actions, then fewer adapter parameters, then canonical B/D/E order. There is no tolerance, weighted score, p-value, subject aggregation, class-harm veto, or backward deletion.

`EMPTY` defines observed gain:

```text
gain[S] = macro_loss[EMPTY] - macro_loss[S]
```

If the winner has positive gain, its status is `positive_gain`. Otherwise it remains the mandatory nonempty winner with status `forced_nonempty_best_observed`.

## Explanation Certificate

All explanation values use already cached branches. They never trigger another training or validation pass.

For selected subset `S` and action `a` in `S`:

```text
conditional_contribution[a | S] = macro_loss[S without a] - macro_loss[S]
```

For pair `a,b`:

```text
pair_interaction[a,b] = gain[{a,b}] - gain[{a}] - gain[{b}]
```

For B/D/E:

```text
triple_interaction = gain[BDE] - gain[B] - gain[D] - gain[E]
                     - pair_interaction[B,D]
                     - pair_interaction[B,E]
                     - pair_interaction[D,E]
```

The decision artifact also records every branch loss, per-class losses and gains, adapter count, elapsed time, the runner-up subset, and the raw selection gap. These values explain the low-fidelity decision; they do not claim the subset remains optimal after full training.

## Removed Behavior

- Subject IDs and subject-cluster jackknife.
- One-sided tests, Holm correction, and supported/weak/mandatory evidence labels.
- Singleton-first selection, conditional forward addition, alternative-pair rescue, retired actions, and path-dependent stopping.
- Floating or backward deletion.
- E-only structural residuals and all hand-weighted module scores.

## Claim Boundary

Module C is an exhaustive low-fidelity selector over three predefined functional actions. It guarantees matched branch evaluation and complete search-space coverage under the probe budget, not the final full-training winner. Final effectiveness is established by matched formal controls, multiple seeds, and a representative full-combination oracle audit.
