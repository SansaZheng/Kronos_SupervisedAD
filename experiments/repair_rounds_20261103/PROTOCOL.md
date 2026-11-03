# Predeclared repair rounds — 2026-11-03

This protocol is committed before new experiment results. Same source splits, feature schemas, target ownership, training-only preprocessing and original Kronos-mini initialization are kept for every candidate. Continuous features get a new additive raw-feature head, preserving numeric amplitudes and causal raw changes alongside the pretrained temporal representation. Sequence classification uses last/mean/std/max/min over valid input only; this is offline sequence classification, not an online future-aware point detector.

| Round | Head | Maximum epochs | Head LR | Backbone LR ratio | Positive-weight cap |
| --- | --- | ---: | ---: | ---: | ---: |
| R0 | existing multiscale reference | 5 | 1e-4 | .1 | none |
| R1 | hybrid raw residual | 5 | 1e-4 | .1 | none |
| R2 | hybrid, faster/longer optimization | 12 | 1e-3 | .1 | none |
| R3 | hybrid, conservative capped weighting | 12 | 3e-4 | .3 | 100 |

R1 vs R0 isolates the architectural addition under the same training budget. R2/R3 are optimization bundles, not single-factor causal ablations. Seed 42 for selection. Early stopping uses validation AP, patience 3. Each dataset selects the candidate by validation AP, then validation F1, then the earlier/simpler candidate. The reference stays eligible; no unconditional replacement of the default model. Validation-selected F1 thresholds remain fixed during testing. A better validation AP does not guarantee better test AP, F1 or recall.

The test NPZ is renamed away during all primary training trials, and each training call uses `--defer-test`. A selection JSON and its SHA-256 are fixed before evaluation. Previously published test results have already been seen: this is a reassessment on a historical holdout, not a claim of a new untouched external test. No test-driven retries, thresholds or candidate changes.

Public evaluation covers the same nine tasks. The exact same label-independent 499,968/99,968 training/validation target caps are used for the three large transaction datasets, with complete time holdouts. For ULB, PaySim and TabFormer, fixed selected recipes are checked against R0 again with paired seeds 13 and 2026 if a non-reference candidate was selected. Confirmation scores cannot change the selected recipe. Every trial/negative result is retained as metrics; only newly generated nonselected model tensors and temporary input copies may be removed to keep transfers bounded. Existing inputs, models, experiments, environments and other people's jobs remain intact.

CQ CPU qualification -> GPU smoke -> formal GPU batch, all pinned to the same code commit, free Kaggle only. Formal batch maximum 2 hours, no paid rentals or Colab CU. IEEE-CIS source/derived data stay local and are excluded from the public cloud batch.

Design motivation: Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) discusses strong residual tabular baselines and the absence of a universally best architecture. Our additive head is a project-specific extension, not a claimed reproduction of that paper's exact architecture or reported scores.
