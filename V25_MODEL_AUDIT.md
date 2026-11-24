# LeukemiPrediag v2.5 — Unfiltered / Censoring-Aware Model Audit

**Status:** research use only; not a diagnostic device. Outcome-free VAF forecasting only.  
**Date:** 2026-10-07

## 1. Reproducibility fixes

Notebook 16 was repaired and re-run using the current `src/temporal_validation.py` logic.

- Corrupted BEL text `\u0007lpha` was replaced by `\alpha` in the markdown equation.
- The regenerated `temporal_validation_audit.json` now contains the `interpretation` key.
- Notebook-16 numerical results were reproduced:
  - carry-forward MAE(VAF): 0.018139
  - shrunk log-linear MAE(VAF): 0.016765
  - carry-forward mean |log error|: 0.285946
  - shrunk mean |log error|: 0.278819
  - fold alphas: 0.20 / 0.25 / 0.20 / 0.25 / 0.20

Notebook 17 was also re-run without tuning to the audit targets. The requested seed-17 values match `V24_MODEL_AUDIT.md` exactly to 5 decimals:

| Model | Re-run | V24 target | Match |
|---|---:|---:|---|
| carry_forward | 0.28605 | 0.28605 | yes |
| global_alpha | 0.27961 | 0.27961 | yes |
| alpha_vaf_stratum | 0.27098 | 0.27098 | yes |
| eb_cohort | 0.31094 | 0.31094 | yes |

## 2. Recovery of sub-floor observations

`real_serial_filtered.csv` was confirmed to apply a 1% VAF floor. The supplied raw tables were inspected directly and contain many sub-1% and zero-VAF observations.

A new file was created without overwriting the filtered file:

```text
data/processed/v2/real_serial_unfiltered.csv
```

Total rows increased from 1,483 to 18,081. See `DATA_UNFILTERED_AUDIT.md` for cohort-level details and source limitations.

Important: the LBC source tables are themselves labelled `1PCT_VAF`, so v2.5 recovers below-floor measurements for represented variants but does not remove source-level variant ascertainment.

## 3. Censoring-aware growth model

New versioned layer:

```text
src/growth_censored.py
notebooks/18_censoring_aware_growth.ipynb
```

The model uses a latent logit-linear VAF trajectory.

- GSE178936 / GSE288742: exact binomial likelihood from `alt_reads` (`AO`) and `depth` (`DP`), including zero and sub-1% measurements.
- SardiNIA: reported VAF with left-censoring below the documented operational 0.2% detection definition because Supplementary Table 2 does not provide per-timepoint counts.
- A fixed broad slope prior SD of 0.1/year is used (`ridge_lambda = 50`). It is fixed in advance rather than tuned on test data.

The v2.4 2% `alpha_vaf_stratum` threshold was held fixed and remains exploratory.

### Evaluation set

- 18,081 unfiltered rows
- 5,426 represented trajectories
- 5,042 trajectories with >=3 timepoints eligible for leave-last forecasting
- 459 patients

### Seed-17 patient-held-out 5-fold CV

Mean absolute log-VAF error; lower is better:

| Model | MAE log-VAF |
|---|---:|
| carry_forward | 1.17135 |
| global_alpha | 1.17135 |
| drift_global | 1.17135 |
| alpha_vaf_stratum | 1.17087 |
| **censored_growth** | **1.18906** |

On the unfiltered table, training-fold optimization usually collapses `global_alpha` and `drift_global` back to near carry-forward. This is a real change from the filtered-table result and is consistent with the original concern that the 1% preprocessing floor created optimistic growth signal.

### 20 repeated CV seeds

| Model | Mean [min, max] |
|---|---|
| carry_forward | 1.17135 [1.17135, 1.17135] |
| global_alpha | 1.17135 [1.17135, 1.17135] |
| drift_global | 1.17194 [1.17135, 1.17412] |
| alpha_vaf_stratum | 1.17083 [1.17063, 1.17114] |
| censored_growth | 1.18906 [1.18906, 1.18906] |

### Leave-one-cohort-out

| Model | LOCO MAE log-VAF |
|---|---:|
| carry_forward | 1.17135 |
| global_alpha | 1.17135 |
| drift_global | 1.17407 |
| alpha_vaf_stratum | 1.17180 |
| censored_growth | 1.18906 |

### Patient-cluster bootstrap

Censored-growth minus comparator; negative would favour the new model:

- vs carry_forward: +0.01771, 95% CI [-0.07481, +0.14150]
- vs global_alpha: +0.01771, 95% CI [-0.07481, +0.14150]
- vs drift_global: +0.01771, 95% CI [-0.07481, +0.14150]
- vs alpha_vaf_stratum: +0.01819, 95% CI [-0.07413, +0.14198]

### Cohort heterogeneity

The censoring-aware model improves GSE288742 relative to carry-forward (1.13061 vs 1.19494) but is worse in GSE178936 (1.08867 vs 1.06821) and SardiNIA (1.54489 vs 1.44240). The overall acceptance criterion therefore fails.

## 4. Decision on censoring-aware point forecaster

**REJECTED as a core point forecaster on the current data.**

Pre-specified acceptance required all of:

1. beat `alpha_vaf_stratum` with a patient-cluster bootstrap CI excluding 0;
2. beat `drift_global` with a patient-cluster bootstrap CI excluding 0;
3. not become worse under leave-one-cohort-out evaluation.

None is satisfied. This is recorded as a negative result. `growth_censored.py` is retained as a versioned experimental layer rather than replacing v2.4/v2.3 code.

The negative result does **not** mean sub-floor observations are unimportant. It means this simple latent linear point forecaster is not adequate across these heterogeneous assays/cohorts.

## 5. Cohort-aware 80% interval calibration

A separate training-only cohort-aware conformal scaling layer was tested on the original filtered v2.4 trajectory set. The point forecasts were not changed.

For the exploratory `alpha_vaf_stratum` model:

| Cohort | Pooled coverage | Cohort-aware coverage | Pooled mean log-width | Cohort-aware mean log-width |
|---|---:|---:|---:|---:|
| GSE178936 | 0.882 | 0.809 | 0.758 | 0.622 |
| GSE288742 | 0.868 | 0.816 | 1.001 | 0.784 |
| SardiNIA | 0.743 | 0.795 | 0.885 | 1.004 |
| **Overall** | **0.799** | **0.801** | **0.855** | **0.862** |

This addresses the original SardiNIA under-coverage while reducing over-coverage in GSE178936. The overall interval width changes only slightly. Cohort-aware interval scaling is therefore supported as a calibration procedure for research reporting, while the underlying `alpha_vaf_stratum` point forecaster remains exploratory.

For completeness, SardiNIA cohort-aware coverage becomes approximately:

- carry_forward: 0.783 (from 0.727)
- global_alpha: 0.795 (from 0.751)
- drift_global: 0.791 (from 0.727)
- alpha_vaf_stratum: 0.795 (from 0.743)

## 6. Files

New:

```text
src/unfiltered_calls.py
src/growth_censored.py
src/interval_calibration.py
notebooks/18_censoring_aware_growth.ipynb
DATA_UNFILTERED_AUDIT.md
V25_MODEL_AUDIT.md
data/processed/v2/real_serial_unfiltered.csv
data/processed/v2/growth_censored_v25/*
```

Modified:

```text
notebooks/16_temporal_forecast_validation.ipynb  # markdown corruption only
CLAUDE.md                                        # handoff log
```

Not modified:

```text
src/growth_shrinkage.py
src/temporal_validation.py
```

## 7. Remaining limitations / next scientific step

1. The LBC raw sources remain variant-ascertained by their supplied `1PCT_VAF` universe; a true all-locus pileup matrix would be better.
2. SardiNIA lacks per-timepoint read counts/depth in Supplementary Table 2, limiting a unified count model.
3. The simple latent linear censored model is not cross-cohort robust.
4. The 2% `alpha_vaf_stratum` rule remains exploratory because its threshold was selected post hoc in v2.4.
5. The strongest positive v2.5 result is **calibration**, not a new point forecaster: cohort-aware interval scaling fixes much of the SardiNIA under-coverage without materially inflating overall width.
6. Do not add a more complex state-space model merely because this one failed. The next model should require either a better assay-level observation matrix or an explicit demonstration that additional latent-state complexity improves held-out forecasts.
