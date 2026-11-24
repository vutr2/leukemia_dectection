# LeukemiPrediag v2.3 Model Audit

**Status:** research use only; not a diagnostic device.

## Why v2.3

The v2.2 feature-engineering layer produced a broad discovery catalogue, but the
current real-data matrix has 76 numeric research features for 265 participants.
A correlation audit found 70 feature pairs with absolute Pearson correlation at
least 0.90, including 19 pairs at least 0.98. Feeding the entire catalogue into
a small outcome cohort by default would create unnecessary redundancy and
increase overfitting risk.

v2.3 therefore separates:

- a compact **confirmatory core** feature set for future outcome-linked testing;
- the full **extended** feature catalogue for predeclared sensitivity/discovery analyses;
- explicitly **experimental** acceleration features.

## Compact v2.3 longitudinal core

1. posterior maximum growth rate;
2. median posterior growth uncertainty;
3. maximum raw dVAF/dt;
4. number of high-support longitudinal variant pairs;
5. fraction of evaluated pairs with high support;
6. number of multi-variant candidate clones;
7. clone VAF entropy;
8. clone growth dispersion;
9. number of trajectories observed to cross VAF 20%.

The 20% crossing feature is a covariate, not a diagnostic rule.

## CHRS baseline cleanup

The published CHRS high-risk mutation category is now represented explicitly as:

- SRSF2, SF3B1, ZRSR2;
- JAK2, TP53;
- IDH1, IDH2, FLT3, RUNX1.

`single_dnmt3a` is defined as exactly one CH mutation and that mutation being
DNMT3A. The component-level static baseline is kept separate from a
`chrs_score`-only reference so the same information is not intentionally counted
twice in a confirmatory model.

## Outcome-free temporal validation

v2.3 adds leave-last-timepoint-out forecasting. For each eligible mutation
trajectory, the final VAF measurement is hidden and predicted from previous
measurements.

Current real-data audit:

- 265 participants;
- 423 eligible patient-variant trajectories;
- carry-forward baseline MAE(VAF): 0.01814;
- unshrunk log-linear MAE(VAF): 0.03148;
- patient-cross-fitted shrunk log-linear MAE(VAF): 0.01676;
- carry-forward mean absolute log-VAF error: 0.28595;
- shrunk log-linear mean absolute log-VAF error: 0.27882;
- outer-fold shrinkage weights: 0.20, 0.25, 0.20, 0.25, 0.20.

The improvement is modest and heterogeneous. The shrunk forecast improved over
carry-forward in SardiNIA, was approximately neutral in GSE288742, and was worse
in GSE178936 on mean absolute log-VAF error. Therefore v2.3 does **not** claim a
universal forecasting win. This result instead shows that unregularized growth
extrapolation is too noisy in the present sparse data and that conservative
shrinkage is necessary.

## Scientific interpretation

Longitudinal growth remains biologically motivated, but an estimated slope is
not automatically predictive. v2.3 explicitly tests forward VAF prediction and
keeps a no-growth carry-forward baseline. Future clone-dynamics models should
beat that baseline in independent cohorts before stronger trajectory-prediction
claims are made.

## Next model work

1. hierarchical / partially pooled clone-growth model with cohort and gene
   effects while preserving patient-held-out validation;
2. posterior predictive intervals and calibration for future VAF;
3. mutation-type representations beyond missense;
4. leakage-safe landmark/index construction for outcome cohorts;
5. survival-aware progression head once true event/censoring data are available;
6. external cohort validation and calibration drift testing.
