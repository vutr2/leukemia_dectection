# CLAUDE.md — LeukemiPrediag AI Collaboration Guide

> This file is the shared operating context for Claude Code, ChatGPT, and the project owner.
> Read this file before modifying the repository.
> Project files/code should be written in English unless the owner explicitly asks otherwise.

## 1. Project mission

LeukemiPrediag is a biomedical AI research/startup project focused on **longitudinal clonal evolution in clonal hematopoiesis (CH)**.

The near-term scientific goal is:

> Quantify which abnormal blood clones are expanding, how fast they are expanding, how uncertain that estimate is, and which mutations are likely to move together as candidate clone structures.

The long-term goal is:

> Predict progression to myeloid malignancy (e.g. AML/MDS) from genuinely pre-diagnostic, outcome-linked longitudinal data.

The current public cohorts do **not** provide a clean prospective leukemia outcome endpoint. Therefore the current model must be treated as a **measurement and representation engine**, not a validated leukemia predictor.

Research use only. Not a diagnostic device.

---

## 2. Non-negotiable scientific boundaries

Never make any of the following claims unless a future outcome-linked cohort supports them:

- "LeukemiPrediag detects leukemia early."
- "This model predicts who will get leukemia."
- "VAF >= 20% means the patient will develop leukemia."
- "Two mutations with high co-movement support are proven to be in the same biological clone."
- "A high BioNeMo score is leukemia risk."
- "BeatAML diagnosis-time performance validates pre-diagnostic screening."

Always distinguish:

- CH / CHIP / CCUS != leukemia.
- Clone expansion != inevitable malignant progression.
- Protein pathogenicity != clone fitness.
- Candidate co-clonal structure != proven phylogeny.
- Diagnosis-time classification != pre-diagnostic prediction.

Do not fabricate labels such as `VAF > 0.20`, high-risk gene presence, or clone growth as a proxy for leukemia outcome.

---

## 3. Current architecture

```text
serial NGS / VCF + dates + CBC + age
                |
                v
        harmonize variants
                |
                v
        patient trajectories
                |
                v
       clone inference layer
                |
      +---------+----------+
      |                    |
      v                    v
growth posterior      clone structure
uncertainty           co-movement support
kinetics              competition / dominance
      |                    |
      +---------+----------+
                |
                v
        M1 dynamics features
                |
                +------------------+
                |                  |
                v                  v
      static clinical M0     BioNeMo / ESM2
                                  |
                                  v
                            M2 molecular layer
                |
                v
      future progression model
      (only with valid outcome data)
```

Model ladder:

- **M0** = static clinical/genetic baseline
- **M1** = M0 + longitudinal clonal dynamics
- **M2** = M1 + NVIDIA BioNeMo / ESM-2 residue-delta representation

The key future questions are:

- `Delta_clone = performance(M1) - performance(M0)`
- `Delta_BioNeMo = performance(M2) - performance(M1)`

BioNeMo is the molecular-intelligence layer, **not** the entire model.

---

## 4. Project location in Google Colab

Canonical project root:

```python
/content/drive/MyDrive/leukemia-prediagnose/blood-cancer-early-detection
```

Do not silently replace this with:

```python
/content/drive/MyDrive/blood-cancer-early-detection
```

unless the runtime confirms that path exists.

Recommended Colab setup:

```python
from pathlib import Path
import os, sys

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/leukemia-prediagnose/blood-cancer-early-detection"
)
os.chdir(PROJECT_ROOT)

SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

When possible, notebooks should auto-detect the project root and fail with an explicit path diagnostic.

---

## 5. Repository areas

Important source modules:

```text
src/
├── embedding.py
├── embedding_bionemo.py
├── fitness.py
├── longitudinal.py
├── real_data_adapter.py
├── clone_dynamics.py
├── clone_inference.py
├── progression_features.py
├── temporal_validation.py
├── growth_shrinkage.py      (v2.4 audit layer)
├── unfiltered_calls.py      (v2.5 sub-floor recovery)
├── growth_censored.py       (v2.5 experimental censoring-aware forecaster)
├── interval_calibration.py  (v2.5 cohort-aware conformal scaling)
├── model_spec.py
├── chrs_baseline.py
├── fusion_model.py
└── inference_v2.py
```

Important audits / documentation:

```text
README.md
REAL_DATA_AUDIT.md
CLONE_INFERENCE_AUDIT.md
V22_FEATURE_AUDIT.md
V23_MODEL_AUDIT.md
V24_MODEL_AUDIT.md
V25_MODEL_AUDIT.md
DATA_UNFILTERED_AUDIT.md
CLAUDE.md
```

Do not delete legacy work unless explicitly instructed.

---

## 6. Current real-data assets

Current longitudinal cohorts include:

- GSE178936
- GSE288742
- SardiNIA

Current Stage-B build is approximately:

- 265 participants
- 423 tracked patient-variant trajectories
- 1,483 longitudinal rows
- 238 evaluated within-patient variant pairs
- 21 high-support variant pairs at support threshold 0.80
- 18 multi-variant candidate clone groups
- 18 patients with supported multi-variant structure

The current data are useful for:

- VAF trajectory measurement
- growth-rate estimation
- uncertainty estimation
- clone co-movement inference
- temporal forecasting tests
- feature engineering

They are **not** sufficient for validated future-leukemia prediction.

Known data limitation (v2.4): `real_serial_filtered.csv` keeps only VAF >= 1%. Below-floor / absent measurements were dropped, so trajectories are selected on detectability and near-floor growth is biased. Recover the unfiltered calls before building any censoring-aware growth model.

---

## 7. BioNeMo / ESM-2 policy

Do not remove BioNeMo/ESM-2 from the project.

Current representation idea:

```text
delta = ESM2(mutant)[mutated_residue] - ESM2(wildtype)[mutated_residue]
```

The current molecular layer is optional and should be evaluated by ablation.

Required comparison when outcome data become available:

```text
M0: static
M1: static + longitudinal dynamics
M2: static + longitudinal dynamics + BioNeMo
```

Also compare any apparent BioNeMo gain to a random-embedding null.

A prior negative fitness-regression result does not justify deleting BioNeMo. It only means that the current standalone protein representation did not generalize for that noisy target.

---

## 8. Clone inference rules

`src/clone_inference.py` is the current uncertainty-aware candidate clone engine.

It estimates:

- posterior log-VAF growth
- growth SD / intervals
- `P(growth > 0)`
- `P(growth > fast_growth_threshold)`
- pairwise co-movement support
- conservative complete-link candidate clone groups
- clone-level pooled common growth

Candidate clone grouping is based on longitudinal evidence and must not be described as proven single-cell phylogeny.

Default research pair-support threshold:

```text
0.80
```

Keep threshold-sensitivity analyses (e.g. 0.70 / 0.80 / 0.90).

---

## 9. v2.2 kinetics / feature layer

`src/progression_features.py` extends Stage-B measurements without creating outcome labels.

Important feature families:

- VAF burden
- raw `dVAF/dt`
- log-VAF growth
- positive-growth doubling time
- posterior growth uncertainty
- candidate clone architecture
- clone dominance / diversity
- supported co-mutation architecture
- configurable gene-presence flags
- experimental acceleration

Acceleration:

```text
d^2 log(VAF) / dt^2
```

is experimental and should only be estimated with sufficient timepoints (currently >=4). It is noise-sensitive.

VAF thresholds such as 10% or 20% are covariates only, never diagnostic rules.

---

## 10. v2.3 model-hardening policy

Do not assume "more features = better model."

Current priority is **generalization**, not feature count.

Use a compact confirmatory feature set for the future progression model. Keep large exploratory feature catalogues separate.

Key development rule:

> No new complexity is accepted unless it beats a simpler baseline under patient-held-out validation.

Temporal validation should include a leave-last-timepoint forecast:

```text
past VAF observations
        |
        v
estimate growth
        |
        v
predict held-out future VAF
```

A useful conservative forecast form is:

```text
log(VAF_future_hat)
=
log(VAF_last)
+
alpha * estimated_growth * delta_t
```

where `alpha` must be chosen on training patients only.

Current evidence suggests raw extrapolation is too aggressive and strong shrinkage is preferable. Treat this as a model-engineering result, not clinical evidence.

v2.4 update (see `V24_MODEL_AUDIT.md`):

- The v2.3 global-alpha gain over carry-forward is **not significant** under a patient-cluster bootstrap. It is worse in GSE178936 and does not transport under leave-one-cohort-out.
- Hierarchical / empirical-Bayes slope shrinkage (cohort and gene levels) was tested and **rejected**. It is worse than carry-forward.
- Every future forecaster must be compared against carry-forward, `global_alpha` and `drift_global` (population drift, no individual slope). It must report:
  - a patient-cluster bootstrap CI,
  - a 20-seed repeated CV,
  - leave-one-cohort-out results.

---

## 11. Notebook workflow

Do **not** run notebooks numerically from 01 to 16.

Current active flow:

```text
12_build_real_features.ipynb
        |
        v
13_infer_probabilistic_clones.ipynb
        |
        +--> 14_analyze_clone_results.ipynb
        |       analysis / plots only
        |
        v
15_engineer_progression_features.ipynb
        |
        v
16_temporal_forecast_validation.ipynb
        |
        v
17_growth_shrinkage_audit.ipynb   (v2.4; needs only real_serial_filtered.csv)
        |
        v
18_censoring_aware_growth.ipynb  (v2.5; uses real_serial_unfiltered.csv)
```

Known user execution state:

- Notebook 12: completed
- Notebook 13: completed
- Notebook 14: available; execution status may need confirmation
- Notebook 15: available; execution status may need confirmation
- Notebook 16: outputs present on Drive (2026-10-07 07:03). Claude reproduced them exactly from `real_serial_filtered.csv`. The saved `.ipynb` has no cell outputs.
- Notebook 17: added by Claude on 2026-10-07. It ran end-to-end in a clean Python 3 kernel outside Colab and has not yet been run in Colab.

Do not run `10_train_fusion.ipynb` for leukemia progression until valid outcome labels exist.

Do not run `11_fusion_inference_demo.ipynb` until a legitimate fusion artifact has been trained.

---

## 12. Future outcome-linked data contract

A true progression cohort should ideally provide:

- patient ID
- serial blood samples before diagnosis
- exact sampling dates / ages
- mutation and VAF per timepoint
- sequencing depth / assay information
- CBC indices
- age
- incident AML/MDS/myeloid malignancy event
- diagnosis date
- adequate follow-up / censoring for non-events

Preferred final modeling target:

```text
event
time_to_event_or_censor
```

Survival analysis should be preferred over inventing simplistic fixed labels when censoring matters.

---

## 13. Statistical modeling principles

1. Fit all learned preprocessing inside training folds.
2. Keep patient identity grouped across train/test splits.
3. Never select thresholds on held-out test labels.
4. Report uncertainty and calibration, not just AUC.
5. Use AUPRC when outcomes are rare.
6. Preserve simple baselines.
7. Use bootstrap confidence intervals for incremental gains.
8. Compare BioNeMo to a random-embedding null.
9. Test cohort heterogeneity explicitly.
10. Never hide a negative result.

Before promoting a component to "core", require evidence that it improves at least one of:

- forward temporal prediction
- uncertainty calibration
- stability / reproducibility
- outcome prediction on a valid future cohort

---

## 14. Feature governance

Three tiers:

### Confirmatory
Small, pre-specified feature set intended for formal validation.

Examples:

- max/median posterior growth
- growth uncertainty
- VAF burden
- supported clone structure
- clone dominance/diversity
- supported co-mutation burden
- CBC / age / CHRS-style baseline

### Exploratory
May be investigated, but must not silently migrate into confirmatory analyses.

Examples:

- acceleration
- large gene-pair catalogue
- many correlated VAF summaries
- higher-order competition features
- v2.4 `alpha_vaf_stratum` forecaster (2% VAF split chosen post hoc)

### Molecular
BioNeMo/ESM-2 and mutation-type representations.

Evaluate separately so the project can determine whether molecular representation truly adds incremental value.

---

## 15. Next model-development priorities

Priority order:

1. Validate temporal forecasting and uncertainty.
2. Improve growth estimation with conservative hierarchical / empirical-Bayes shrinkage.
   - **v2.4 status: tested and rejected on current filtered data.**
   - Next: recover unfiltered (below-1%) calls.
   - Then: a censoring-aware state-space growth model. Accept it only if it beats `alpha_vaf_stratum` and `drift_global`.
3. Add predictive intervals for future VAF, not only point predictions.
   - v2.4: split-conformal 80% intervals reach about 0.80 pooled coverage.
   - Coverage is uneven by cohort (SardiNIA about 0.73–0.76), so the intervals need cohort-aware scaling.
4. Quantify cohort-specific calibration / heterogeneity.
5. Develop conservative clone-competition features.
6. Reduce redundant features before any final clinical model.
7. Expand molecular support beyond missense:
   - truncating
   - frameshift
   - splice
   - in-frame indels
   - hotspot / domain context
8. Prepare survival-aware progression head.
9. Train progression model only when valid outcomes exist.

Do not jump directly to transformers, graph neural networks, or large multimodal architectures unless simpler models have been beaten under valid held-out evaluation.

---

## 16. Claude/ChatGPT collaboration protocol

When either AI changes the repository:

1. Read `CLAUDE.md`, `README.md`, and the newest `V*_MODEL_AUDIT.md`.
2. Inspect the actual input schemas before writing code.
3. Preserve working v2.x code unless there is a documented reason to replace it.
4. Prefer adding a new versioned layer over silently changing a previously audited result.
5. Run syntax checks and minimal data sanity checks.
6. Record:
   - what changed,
   - why it changed,
   - files modified,
   - expected inputs,
   - outputs,
   - validation result,
   - unresolved limitations.
7. Update the `AI HANDOFF LOG` below after a meaningful model change.

If Claude finds a bug or disagrees with a modeling choice, it should write the concern explicitly rather than silently rewriting the scientific assumption.

---

## 17. AI HANDOFF LOG

Use newest entries first.

### 2026-10-07 — ChatGPT (v2.5 unfiltered / censoring-aware audit)

**What / why:** Completed the v2.4 handoff tasks: fixed reproducibility issues, recovered sub-1% observations from the supplied raw cohorts, tested a simple censoring-aware growth forecaster under the same strict validation protocol, and corrected cohort-specific interval miscalibration.

**Files added:**

- `src/unfiltered_calls.py`
- `src/growth_censored.py`
- `src/interval_calibration.py`
- `notebooks/18_censoring_aware_growth.ipynb`
- `DATA_UNFILTERED_AUDIT.md`
- `V25_MODEL_AUDIT.md`
- `data/processed/v2/real_serial_unfiltered.csv`
- `data/processed/v2/growth_censored_v25/*`

**Files modified:**

- `notebooks/16_temporal_forecast_validation.ipynb` — fixed corrupted `\alpha` markdown only; computational logic unchanged.
- `CLAUDE.md` — this handoff.

**Files explicitly NOT modified:**

- `src/temporal_validation.py`
- `src/growth_shrinkage.py`

**Reproducibility:**

- Notebook 16 re-run reproduces carry-forward MAE(VAF) `0.018139`, shrunk MAE(VAF) `0.016765`, |log error| `0.285946 -> 0.278819`, fold alphas `0.20/0.25/0.20/0.25/0.20`; regenerated audit includes `interpretation`.
- Notebook 17 re-run matches V24 targets exactly to 5 decimals: carry-forward `0.28605`, global-alpha `0.27961`, alpha-VAF-stratum `0.27098`, EB-cohort `0.31094`. No tuning was done to force agreement.

**Unfiltered recovery:**

- New table: 18,081 rows, 5,426 represented trajectories, 459 patients; 11,355 rows are below 1% VAF and 1,406 rows have VAF=0.
- GSE178936: 10,173 rows; GSE288742: 3,616; SardiNIA: 4,292.
- All three supplied files contain real sub-floor observations, so the censoring experiment was allowed.
- Important ascertainment caveat: the LBC source files are themselves labelled `1PCT_VAF`. The new table is trajectory-unfiltered within that supplied variant universe, not a complete assay-level all-locus call matrix.

**v2.5 point-forecast result:**

- Tested a simple latent logit-linear model: binomial AO/DP observations for LBC and left-censored logit-VAF observations for SardiNIA; fixed slope prior SD `0.1/year` (`ridge_lambda=50`).
- Seed-17 mean |log-VAF error|: carry-forward `1.17135`, global-alpha `1.17135`, drift-global `1.17135`, alpha-VAF-stratum `1.17087`, censoring-aware `1.18906`.
- 20-seed and LOCO evaluations give the same decision.
- Bootstrap delta censoring-aware minus alpha-VAF-stratum: `+0.01819`, 95% CI `[-0.07413, +0.14198]`.
- **Decision: censoring-aware point forecaster REJECTED as core.** It fails the pre-specified acceptance rule and is kept only as an experimental versioned layer.

**Interval calibration result:**

- Training-only cohort-aware 80% conformal scaling is supported for research uncertainty reporting.
- For `alpha_vaf_stratum`, SardiNIA coverage improves from `0.743` to `0.795`; GSE178936 over-coverage falls from `0.882` to `0.809`; pooled coverage remains approximately `0.80`; overall mean log-width changes `0.855 -> 0.862`.
- This changes interval scaling only; it does not promote the exploratory 2% point-forecast rule to core.

**Unresolved limitations / next recommendation:**

1. Obtain assay-level pileup/VCF matrices that include all interrogated loci, not only a source-level 1%-ascertained variant universe.
2. Obtain per-timepoint read counts for SardiNIA-like cohorts or another independent serial cohort with counts.
3. Do not add a more complex state-space model merely because v2.5 failed; require a better observation matrix or a pre-specified reason that new latent-state complexity should improve held-out forecasts.
4. Keep the 2% `alpha_vaf_stratum` threshold exploratory until prospectively fixed and confirmed on independent or truly unfiltered data.
5. Cohort-aware interval calibration can be retained as a research reporting layer; point-forecast generalization remains the main unresolved model problem.

### 2026-10-07 — ChatGPT (v2.5 unfiltered / censoring-aware audit)

**What / why:** Continued from Claude v2.4 exactly as requested. Main goal was to test whether the apparent near-floor growth signal survives after recovering sub-1% / zero-VAF measurements, and to fix cohort-specific predictive-interval miscalibration.

**Reproducibility:**
- Repaired notebook 16 markdown corruption (`\u0007lpha` -> `\alpha`).
- Re-ran notebook 16 with current code; regenerated `temporal_validation_audit.json` now includes `interpretation`.
- Re-ran notebook 17 without tuning to audit targets. Seed-17 mean |log-VAF error| reproduced V24 exactly to 5 decimals:
  - carry_forward = 0.28605
  - global_alpha = 0.27961
  - alpha_vaf_stratum = 0.27098
  - eb_cohort = 0.31094

**Unfiltered-call recovery:**
- Confirmed the v2 build applies `VAF >= 1%` in `src/real_data_adapter.py` / notebook 12.
- Inspected the supplied raw files for GSE178936, GSE288742 and SardiNIA. All three contain sub-1% and zero-VAF measurements for represented variants.
- Built `data/processed/v2/real_serial_unfiltered.csv` without overwriting `real_serial_filtered.csv`.
- Rows increased from 1,483 to 18,081; 11,355 rows are <1% VAF and 1,406 are zero-VAF.
- Important limitation: the LBC source files are themselves `1PCT_VAF` variant universes, so the recovered table is trajectory-unfiltered within the supplied selected variant set, not a complete all-locus assay matrix.
- See `DATA_UNFILTERED_AUDIT.md`.

**Censoring-aware point forecaster:**
- Tested a simple latent logit-linear / count-aware model on the recovered table under the v2.4 protocol.
- Evaluation set: 18,081 rows, 5,426 represented trajectories, 5,042 eligible leave-last trajectories, 459 patients.
- Seed-17 MAE |log-VAF|:
  - carry_forward = 1.17135
  - global_alpha = 1.17135
  - drift_global = 1.17135
  - alpha_vaf_stratum = 1.17087
  - censored_growth = 1.18906
- 20-seed and LOCO results confirm no promotion. Patient-cluster bootstrap CIs versus all required baselines include 0.
- **Decision: REJECT censoring-aware point forecaster as core.** Retain only as a negative experimental result. Do not add a more complex state-space model just because this failed.

**Interval calibration:**
- Added training-only cohort-aware scaling for 80% split-conformal intervals on the filtered v2.4 trajectory set.
- For exploratory `alpha_vaf_stratum`, SardiNIA coverage improved from 0.743 to 0.795 while overall coverage stayed ~0.80 (0.799 -> 0.801).
- GSE178936 over-coverage decreased (0.882 -> 0.809); GSE288742 0.868 -> 0.816.
- **Positive v2.5 result: cohort-aware interval calibration is supported for research reporting.** The underlying point forecaster remains exploratory.

**Files / outputs:**
- `V25_MODEL_AUDIT.md`
- `DATA_UNFILTERED_AUDIT.md`
- `data/processed/v2/real_serial_unfiltered.csv`
- v2.5 forecast/calibration output tables
- notebook 16 repaired only; `src/growth_shrinkage.py` and `src/temporal_validation.py` were not modified.

**Unresolved limitations:**
1. LBC raw inputs remain source-level 1%-ascertained variant universes.
2. SardiNIA lacks per-timepoint alt-read/depth counts in Supplementary Table 2.
3. The simple censoring-aware point model is not cross-cohort robust.
4. The 2% `alpha_vaf_stratum` threshold remains post-hoc/exploratory.
5. A stronger point model should wait for a true assay-level pileup / all-locus observation matrix or independent data.

**Next recommendation:**
> Freeze the negative point-forecast result. Prioritize assay/ascertainment quality and calibrated uncertainty. Do not escalate model complexity without a better observation matrix or a clear patient-held-out gain over carry-forward, drift_global and alpha_vaf_stratum.

### 2026-10-07 — Claude (v2.4 growth-shrinkage audit)

**What / why:** I validated the notebook 16 results and tested whether a hierarchical growth model is justified, as the ChatGPT handoff asked.

**Files added** (nothing existing was modified except this file):

- `src/growth_shrinkage.py`
- `notebooks/17_growth_shrinkage_audit.py`
- `notebooks/17_growth_shrinkage_audit.ipynb`
- `V24_MODEL_AUDIT.md`

The previous copy of this file was renamed on Drive to `CLAUDE_backup_2026-10-07_pre-v24.md`.

**Inputs:** `data/processed/v2/real_serial_filtered.csv`, with columns `patient_id, cohort, variant_id, time_years, vaf, depth, gene, kind`.

**Outputs:** `data/processed/v2/growth_shrinkage_v24/`, containing:

- `v24_growth_shrinkage_audit.json`
- `v24_crossfit_forecasts.csv`

**Validation result:**

- **Notebook 16 reproduces exactly:**
  - MAE(VAF) 0.01814 → 0.01676; |log err| 0.28595 → 0.27882;
  - fold alphas 0.20/0.25/0.20/0.25/0.20.
- **The notebook 16 gain is weak:**
  - patient-cluster bootstrap Δ|log err| vs carry-forward is −0.0064, 95% CI [−0.017, +0.004], not significant;
  - significantly worse in GSE178936;
  - leave-one-cohort-out global-alpha is worse than carry-forward (0.2965 vs 0.2861);
  - a 1-parameter population drift (no individual slope) matches it.
- **Hierarchical / EB slope shrinkage is REJECTED.**
  - Tested a depth-aware binomial + extra-binomial noise model with a cohort prior, plus a gene-pooling level.
  - It is worse than carry-forward and global-alpha in CV (0.311), in 20 seeds and in LOCO (0.304).
  - Cause: 62% of trajectories have only 2 training points, the 1% VAF floor biases near-floor growth, and anchor-noise / regression to the mean dominates.
- **Exploratory candidate: `alpha_vaf_stratum`.**
  - It uses alpha = 0 when the last VAF is < 2% and alpha ≈ 0.35 when it is ≥ 2%.
  - Results: 0.2710; Δ vs carry-forward −0.015 [−0.029, −0.001]; Δ vs global-alpha −0.009 [−0.015, −0.002].
  - It wins 20/20 CV seeds and gives LOCO 0.2795.
  - Not significant vs drift_global; GSE178936 still not improved.
  - The threshold was chosen post hoc, so it is **not promoted to core**.

**Concerns raised:**

- Notebook 16 markdown has a corrupted `\u0007lpha` (BEL char) instead of `\alpha`.
- `temporal_validation_audit.json` on Drive has `by_cohort` and lacks `interpretation`, so it was written by a different code version than the current notebook 16 cell. Re-run notebook 16 to regenerate it.
- Silent variants also show positive drift, consistent with detection-floor selection.

**Unresolved:**

- The censoring-aware growth model needs unfiltered below-1% calls.
- Cohort-aware interval calibration (SardiNIA under-covered).
- Notebook 17 not yet run in Colab.

**Next recommendation:**

> Recover unfiltered calls for the three cohorts. Pre-register the 2% `alpha_vaf_stratum` rule and test it against carry-forward and drift_global on any new or independent serial cohort. Do not build further hierarchical layers on the filtered table.

### 2026-10-07 — ChatGPT
Current focus changed to **model completion as priority #1**.

Added / hardened:

- v2.2 progression feature engineering
- compact vs exploratory feature governance
- temporal leave-last forecasting
- shrinkage-based growth extrapolation
- patient-held-out validation policy
- stronger separation between measurement and clinical prediction

Current next recommendation:

> Run / inspect notebook 16, then use its results to decide whether a hierarchical growth model is justified. Do not add model complexity unless it beats carry-forward and shrunk-growth baselines.

### Earlier state
- v2.1 uncertainty-aware clone inference established.
- real data pipeline established for GSE178936, GSE288742, and SardiNIA.
- BioNeMo/ESM-2 retained as molecular layer.
- clinical progression experiment remains blocked by lack of clean outcome-linked longitudinal data.

---

## 18. Definition of "model complete" for the current data stage

The model is considered technically mature for the current Stage-B data when:

- real-data ingestion is reproducible,
- clone growth is uncertainty-aware,
- clone grouping is conservative and sensitivity-tested,
- forward VAF forecasting beats or matches strong simple baselines,
- predictive uncertainty is calibrated,
- feature redundancy is controlled,
- cohort heterogeneity is quantified,
- BioNeMo is cleanly isolated as an ablation layer,
- code runs end-to-end from notebooks 12 -> 16,
- results are reproducible from a clean Colab runtime,
- all scientific claims stay inside the evidence.

Clinical completion requires a separate future stage with outcome-linked longitudinal validation.
