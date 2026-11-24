# LeukemiPrediag v2.4 Growth-Shrinkage Audit

**Status:** research use only; not a diagnostic device. Outcome-free VAF forecasting only.
**Author:** Claude (2026-10-07), continuing the ChatGPT v2.3 handoff.

## Question

Does a hierarchical / empirical-Bayes (EB) growth model beat the simple v2.3
forecasters under patient-held-out validation? Rule from `CLAUDE.md` §10: no new
complexity unless it beats a simpler baseline.

## 1. Notebook 16 re-check

From `data/processed/v2/real_serial_filtered.csv` (1,483 rows, 265 patients, 423
trajectories, all eligible), notebook 16 reproduces **exactly**: carry-forward
MAE(VAF) 0.01814, shrunk MAE(VAF) 0.01676, |log err| 0.28595 → 0.27882, fold alphas
0.20/0.25/0.20/0.25/0.20, and the same per-cohort errors.

Problems found (none change the v2.3 numbers):

1. **No uncertainty on the gain.** With a patient-cluster bootstrap, the global-alpha
   gain over carry-forward is −0.0064 |log err|, 95% CI [−0.0171, +0.0036]. **It is not
   significant.** It is also significantly *worse* than carry-forward in GSE178936
   (+0.020 [+0.010, +0.030]).
2. **Split-sensitive.** Over 20 CV seeds, global-alpha ranges from 0.2791 to 0.2863. The
   seed-17 result is near the favourable end.
3. **Does not transport.** Leave-one-cohort-out global-alpha gives 0.2965, worse than
   carry-forward (0.2861).
4. **Individual slopes add little over a population drift.** A one-parameter model with
   no individual slope, `log VAF_last + mu*dt` (mu ≈ 0.015–0.02/yr, tuned on training
   data), matches global-alpha: 0.2789 vs 0.2796, difference −0.0007 [−0.009, +0.008].
5. **Detection-floor selection.** Every row has VAF ≥ 1%, so trajectories are filtered
   on detectability. Silent variants (expected near-neutral) also show positive drift,
   so some of the "growth" may be truncation artefact rather than biology.
6. Cosmetic issues:
   - The markdown cell in notebook 16 contains a corrupted `\u0007lpha` (BEL
     character) where `\alpha` was intended.
   - The saved notebook has no outputs.
   - `temporal_validation_audit.json` on Drive has a `by_cohort` key and no
     `interpretation` key, so it was written by a different code version than the
     current notebook cell. This is a provenance mismatch; re-run notebook 16 to
     regenerate it.

## 2. Candidates (all fitted on training patients only)

| model | params | idea |
|---|---|---|
| carry_forward | 0 | no change |
| global_alpha (v2.3) | 1 | alpha × OLS slope |
| drift_global | 1 | population drift, no individual slope |
| drift_cohort / alpha_cohort | 3 | cohort-specific drift / alpha |
| **eb_cohort** | MoM | precision-weighted EB slope: depth-aware binomial + extra-binomial noise; cohort prior mean and tau² |
| **eb_gene** | MoM | eb_cohort with a second level: gene means partially pooled toward the cohort mean |
| alpha_vaf_stratum | 2 | separate alpha for last VAF < 2% vs ≥ 2% (exploratory) |

## 3. Results (mean |log-VAF error|; lower is better)

The global_alpha error here is 0.2796, not notebook 16's 0.2788, because of the log-eps
convention. The fold alphas are identical.

| model | seed-17 CV | 20-seed mean [min, max] | LOCO |
|---|---|---|---|
| carry_forward | 0.2861 | 0.2861 | 0.2861 |
| global_alpha | 0.2796 | 0.2812 [0.2791, 0.2863] | 0.2965 |
| drift_global | 0.2789 | 0.2793 [0.2784, 0.2811] | 0.2859 |
| alpha_cohort | 0.2731 | 0.2733 [0.2717, 0.2761] | 0.2965 (unseen cohort → falls back) |
| **eb_cohort** | **0.3109** | **0.3104** | **0.3035** |
| **eb_gene** | **0.3113** | **0.3093** | **0.3035** |
| alpha_vaf_stratum | **0.2710** | **0.2710 [0.2694, 0.2739]**, best in 20/20 seeds | **0.2795** |

Cluster-bootstrap deltas (95% CI):

- eb_cohort − carry_forward: +0.025 [−0.001, +0.052]
- eb_cohort − global_alpha: +0.031 [+0.012, +0.052]
- eb_gene ≈ eb_cohort (gene pooling adds nothing)
- alpha_vaf_stratum − carry_forward: −0.015 [−0.029, −0.001]
- alpha_vaf_stratum − global_alpha: −0.009 [−0.015, −0.002]
- alpha_vaf_stratum − drift_global: −0.008 [−0.020, +0.004] (not significant)

By cohort, alpha_vaf_stratum minus carry-forward:

- SardiNIA: −0.028 [−0.049, −0.006]
- GSE288742: −0.011 [−0.073, +0.048]
- GSE178936: +0.007 [−0.000, +0.017]. This cohort is still not improved.

Threshold sensitivity is flat from 1.5% to 3% (deltas between −0.013 and −0.016). It
degrades at 1.2% and at 5%. Learned alphas are 0.0 below 2% and 0.30–0.40 above 2% in
every fold.

80% split-conformal intervals:

- Pooled coverage is about 0.80 for all models.
- Coverage is uneven by cohort: SardiNIA 0.73–0.76 (under-covered), GSE178936 0.86–0.90
  (over-covered).
- alpha_vaf_stratum gives the narrowest intervals (mean log-width 0.855 vs 0.904 for
  carry-forward).

## 4. Decision

1. **The hierarchical / EB growth model is NOT justified on current data. Rejected.** It
   is worse than carry-forward and v2.3 shrinkage in CV, in 20 repeated seeds and in
   LOCO. It is not improved by:
   - a gene level (eb_gene);
   - anchoring at the fitted state instead of the last observation (tested, worse:
     0.315).

   Why it fails:
   - 62% of trajectories have only 2 training points.
   - Near the 1% floor, binomial noise dominates.
   - Training-window slopes are biased upward. The EB prior mean is about 0.05/yr,
     while the best forecasting drift is about 0.02/yr.
   - Noise in the last-observation anchor is correlated with the slope (regression to
     the mean). A slope-only EB model does not correct for this.

   The v2.3 "shrinkage" mostly corrects measurement noise; it is not biological pooling.
2. **Keep v2.3 global_alpha as the reference.** Report its gain as non-significant and
   not transportable.
3. **alpha_vaf_stratum is a candidate v2.4 forecaster, marked EXPLORATORY.** It:
   - beats carry-forward and global-alpha with CIs that exclude 0;
   - wins all 20 seeds and transports under LOCO;
   - has a scientifically anchored threshold (2% = conventional CHIP VAF threshold
     and 2× the 1% floor).

   It must not be promoted to core yet:
   - the threshold was chosen after inspecting errors;
   - it does not beat drift_global significantly;
   - it does not help GSE178936.

   Confirm it with the 2% threshold pre-registered on an independent or unfiltered
   cohort.
4. **The structural next step is not more shrinkage.** It is a measurement model that
   treats sub-1% observations as left-censored (a state-space model with binomial
   observation and censoring). This needs the **unfiltered** calls (including
   below-floor and absent measurements) from the raw cohort files. Build it only if
   that data can be recovered, and accept it only if it beats alpha_vaf_stratum and
   drift_global under the same protocol.
5. Interval calibration needs cohort-aware scaling, especially for SardiNIA. Do this
   before any predictive interval is reported per cohort.

## Files

- `src/growth_shrinkage.py`: models, EB noise model, cross-fitting, LOCO, cluster
  bootstrap.
- `notebooks/17_growth_shrinkage_audit.py` / `.ipynb`: reproducible runner, about 30 s
  on CPU. Outputs go to `data/processed/v2/growth_shrinkage_v24/`.
- No existing file was modified. `temporal_validation.py` and notebook 16 are unchanged.
