# LeukemiPrediag v2.1 — Probabilistic Clone Inference Audit

**Status:** research-only candidate-clone inference; not a phylogenetic reconstruction and not a leukemia diagnostic.

## Real-data run

Input: `data/processed/v2/real_serial_filtered.csv` built from GSE178936, GSE288742 and SardiNIA.

- Patients: **265**
- Patient-variant trajectories: **423**
- Pairwise relationships with enough shared timepoints: **238**
- Default posterior draws: **1000**
- Default pair-support threshold: **0.80**

## Noise model

Empirical log-VAF residual noise floor by cohort:

- GSE178936: **0.074**
- GSE288742: **0.092**
- SardiNIA: **0.126**

For LBC cohorts, per-call sequencing depth is also propagated into log-VAF uncertainty. For SardiNIA, the public supplement lacks per-call depth, so the inference uses the empirical cohort noise floor rather than treating the adapter's nominal QC depth as observed read depth.

## Candidate-clone result

At support >= 0.80:

- High-support variant pairs: **21**
- Multi-variant candidate clones: **18**
- Patients with at least one supported multi-variant clone: **18**
- Multi-variant clones by cohort: **{'GSE178936': 2, 'GSE288742': 1, 'SardiNIA': 15}**

These are **candidate co-moving clone groups**, not proof that the variants occupy the same physical clone.

## Uncertainty gain from clone pooling

For the 18 multi-variant groups, the clone-level common-slope posterior had a median SD ratio of **0.709** relative to the mean SD of its member-variant slopes. That is approximately a **29.1% reduction in slope uncertainty**. Clone pooling reduced uncertainty in **100%** of these groups.

This is a measurement-stability result, not an outcome-prediction result.

## Threshold sensitivity

|   pair_support_threshold |   high_support_pairs |   multivariant_clones |   patients_with_supported_structure |
|-------------------------:|---------------------:|----------------------:|------------------------------------:|
|                      0.7 |                   29 |                    24 |                                  24 |
|                      0.8 |                   21 |                    18 |                                  18 |
|                      0.9 |                   13 |                    11 |                                  11 |

The structure count changes with the support threshold, as expected. This sensitivity should be shown in technical review rather than presenting a single threshold as ground truth.

## v2.1 outputs

- `trajectory_posteriors.csv` — uncertainty-aware variant growth posteriors
- `pair_support.csv` — pairwise co-movement support
- `clone_assignments.csv` — conservative complete-link candidate groups
- `clone_posteriors.csv` — clone-level common growth slopes and uncertainty
- `patient_clone_features.csv` — clone-aware patient features
- `real_patient_features_cloneaware.csv` — existing v2 real patient features plus probabilistic clone features
- `threshold_sensitivity.csv` — support-threshold sensitivity analysis

## What this unlocks

`fusion_model.py` now accepts these clone-aware features in M1. When an outcome-linked pre-diagnostic cohort is obtained, the intended experiment remains:

`M0 clinical/static -> M1 + clone dynamics -> M2 + BioNeMo/ESM-2`.

Until that cohort exists, do not interpret clone growth probability as leukemia progression probability.
