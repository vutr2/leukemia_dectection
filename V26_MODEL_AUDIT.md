# LeukemiPrediag v2.6 — Detection-Aware Forecast Audit

**Status:** research use only; not a diagnostic device. Outcome-free VAF forecasting only.
**Author:** Claude (2026-10-07), continuing from the ChatGPT v2.5 handoff.

## Why

I re-checked v2.5 before building on it. The v2.5 forecast table on Drive reproduces
its reported scores (carry-forward 1.1713), but two problems in the evaluation make its point-forecast conclusions
uninterpretable:

1. **The error metric is dominated by exact zeros.** The v2.5 metric
   `|log(pred) − log(true)|` clips VAF = 0 to 1e-5.
   - 12% of trajectories have a last or held-out VAF of exactly 0.
   - Those trajectories contribute **48.5%** of the carry-forward error. The error for
     a zero is set by the eps choice, not by biology.
   - This is why every model scored ≈1.17 and the bootstrap CIs were ±0.1.
2. **30% of evaluated trajectories were ascertained by the held-out wave.** The LBC
   source tables are `1PCT_VAF` variant universes, and SardiNIA re-queries a site at
   all timepoints once it is found at any one.
   - 1,508 of 5,042 trajectories never reached 1% in their training window.
   - Of those, 68% have a held-out VAF ≥ 1%, versus 24% of prospectively visible
     trajectories.
   - Their inclusion therefore leaks the future.

A third finding changes how to read the LBC data. **Most recovered LBC measurements are
below the assay's own 95% minimum detectable AF** (`X95MDAF`, median about 1.5%):

- 87% of GSE178936 rows and 75% of GSE288742 rows (SardiNIA: 29% below the 0.2%
  operational limit).
- In the prospective set, 94.5% of LBC held-out values are below their row's limit.

## What v2.6 changes (evaluation only; no new forecaster)

New files:

- `src/forecast_scoring_v26.py`
- `notebooks/19_detection_aware_forecast_audit.py` / `.ipynb`

Outputs go to `data/processed/v2/forecast_scoring_v26/`.

Scoring:

- **Primary score — detection-aware error.** Floor both truth and forecast at the
  held-out row's detection limit L, then take `|log max(pred, L) − log max(true, L)|`.
  Two values below L count as equal.
- **Secondary scores:**
  - MAE on the VAF scale;
  - binomial negative log-likelihood (NLL) of the held-out alt-read count, for LBC
    only (counts exist there).

Primary analysis set: **prospectively ascertained** trajectories, meaning the training
window already contains VAF ≥ 1%. This gives 3,534 trajectories from 367 patients. I
fixed this set and the scoring rules before computing any v2.6 result.

Models and fitting:

- Same models as v2.5: carry-forward, global-alpha, drift_global, `alpha_vaf_stratum`,
  and censored_growth.
- The 2% threshold is frozen from v2.4.
- censored_growth predictions are reused unchanged from v2.5; it is a per-trajectory
  fit with no cross-patient tuning.
- Alpha and drift are tuned on training patients with the same detection-aware loss.
- Protocol is unchanged from v2.4: 5-fold patient-held-out CV (seed 17), 20 seeds,
  leave-one-cohort-out (LOCO), and patient-cluster bootstrap.

## Results (PRIMARY: prospective set)

| model | det. error | 20-seed [min, max] | LOCO | MAE VAF | binom NLL (LBC) |
|---|---|---|---|---|---|
| carry_forward | 0.1501 | constant | 0.1501 | 0.0127 | 9.38 |
| global_alpha | 0.1501 (alpha → 0 every fold) | constant | 0.1511 | 0.0127 | 9.38 |
| drift_global (mu ≈ −0.07/yr) | **0.1129** | 0.1120–0.1146, wins 20/20 | 0.1340 | 0.0106 | 6.85 |
| alpha_vaf_stratum | 0.1484 | 0.1483–0.1489 | 0.1498 | 0.0126 | 9.39 |
| censored_growth (v2.5) | 0.1323 | constant | 0.1323 | **0.0078** | **6.60** |

Detection-aware error by cohort, and LOCO:

| cohort | carry-forward | drift_global | alpha_vaf_stratum | censored_growth |
|---|---|---|---|---|
| GSE178936 | 0.099 | **0.041** (LOCO 0.061) | 0.100 | 0.058 |
| GSE288742 | 0.103 | **0.043** (LOCO 0.044) | 0.102 | 0.133 |
| SardiNIA | 0.520 | 0.643 (LOCO **0.704**) | **0.503** | 0.534 |

Bootstrap deltas (95% CI; negative favours the first model):

- drift_global − carry_forward: −0.037 [−0.054, −0.018]
- censored_growth − carry_forward:
  - overall −0.018 [−0.053, +0.014];
  - GSE178936 −0.041 [−0.090, −0.004];
  - SardiNIA +0.014 [−0.012, +0.041]. On the full set, SardiNIA is +0.021
    [+0.005, +0.037], significantly worse.
- censored_growth − drift_global: +0.019 [−0.011, +0.050]
- Binomial NLL (LBC):
  - censored − carry-forward: −2.78 [−4.22, −1.19];
  - censored − drift: −0.25 [−1.61, +1.43];
  - censored − drift in GSE178936 only: −1.34 [−2.50, −0.35].
- alpha_vaf_stratum − carry_forward: −0.0018 [−0.0041, +0.0002]. Within SardiNIA it is
  −0.016 [−0.032, −0.001].

**Test of the frozen 2% rule on trajectories not used to develop it** (3,111 new
prospective trajectories):

- Overall: Δ vs carry-forward −0.0001 [−0.0019, +0.0013].
- Only 52 new trajectories have a last VAF ≥ 2%, which is where the rule acts. There:
  Δ −0.005 [−0.104, +0.074].
- On the v2.4 development trajectories it still helps (−0.014), but that is not
  independent evidence.

## Interpretation

1. **Forecast skill is set by the measurement regime, not by clone biology.**
   - In LBC, almost every held-out value is below the assay's 95% detection limit.
     Models that pull the forecast down win: negative drift and the count-aware
     censored fit. This is regression to the mean toward "undetectable"; it is not
     evidence of clone shrinkage.
   - In SardiNIA, values are mostly above the limit. Carry-forward or modest growth
     wins, and the LBC-learned negative drift is badly wrong (LOCO 0.704 vs 0.520).
2. **No forecaster is cross-cohort robust.** drift_global wins pooled CV only because
   LBC dominates the trajectory count, and it fails transport to SardiNIA.
   censored_growth is the best count-based scorer in LBC but is worse in SardiNIA.
   **Neither meets the acceptance rule.**
3. **v2.5's rejection of censored_growth stands, but for a different reason.**
   - Under proper scoring it is not worse than carry-forward overall, and it is
     clearly better on LBC counts.
   - It is rejected because it does not transport to the count-free cohort and does
     not beat drift_global.
4. **The 2% `alpha_vaf_stratum` rule is unconfirmed.** It is not refuted, but current
   public data cannot test it: too few new trajectories have a last VAF ≥ 2%. Stop
   iterating on it.
5. **Point forecasting has hit the data ceiling for these cohorts.** Further model
   complexity cannot be justified on them.

## Decisions

- **Adopt v2.6 scoring as the standard for any unfiltered or count data:**
  - detection-aware error;
  - VAF-scale MAE;
  - binomial NLL where counts exist;
  - the prospective-ascertainment eligibility rule.

  Do **not** use the eps-clipped log error on data containing zeros.
- **Freeze point forecasting.** The reference forecaster stays carry-forward with
  cohort-aware conformal intervals (v2.5). drift_global, censored_growth and
  alpha_vaf_stratum stay exploratory and regime-specific.
- Report forecasting results **per cohort / assay regime**, never pooled alone.
- Treat LBC sub-1% measurements as below the limit of detection (sub-LoD), not as
  informative VAF. Do not interpret them as clone shrinkage or growth.

## Concerns / housekeeping found

- **Duplicate files on Drive:**
  - `V25_MODEL_AUDIT.md` (2 copies, same size);
  - `DATA_UNFILTERED_AUDIT.md` (2);
  - `data/processed/v2/real_serial_unfiltered.csv` (2).

  In Colab only one of each name is visible per folder path, so the owner should
  delete the older copies.
- **CLAUDE.md had two v2.5 handoff entries** describing the same work. I kept both
  verbatim.
- **GSE288742 time axis (unverified).** `unfiltered_calls.py` and the v2 adapter map
  LBC waves to `time = wave × 3 years` for both cohorts. That fits LBC1936. LBC1921
  (GSE288742) waves are reported to be unequally spaced (approximate mean ages 79, 83,
  87, 90, 92). Verify against the GEO/cohort metadata. If the spacing is unequal,
  GSE288742 horizons and slopes are mis-scaled in every version since v2.

## What would move the model forward (data, not architecture)

1. Assay-level pileups with alt/depth at all interrogated loci and timepoints, which
   removes source ascertainment.
2. Per-timepoint counts for a SardiNIA-like, above-LoD cohort.
3. Any independent serial CH cohort, to test the frozen rules prospectively.
4. Outcome-linked data (CLAUDE.md §12) for the actual M0/M1/M2 question.

Meanwhile, the highest-value model work on current data is **measurement robustness of
the confirmatory dynamics features**. Check that v2.2/v2.3 core features (posterior
max growth, growth uncertainty, clone structure) keep their rank order when computed
from:

- prospectively ascertained trajectories, and
- detection-aware VAF,

compared with the 1%-filtered table.
