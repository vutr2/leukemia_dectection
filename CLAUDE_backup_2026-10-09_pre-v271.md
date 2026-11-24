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
├── forecast_scoring_v26.py  (v2.6 detection-aware / ascertainment-aware scoring)
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
V26_MODEL_AUDIT.md
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

v2.6 update (see `V26_MODEL_AUDIT.md`):

- On data containing zeros or sub-LoD values (`real_serial_unfiltered.csv`), score forecasts with the detection-aware error (truth and forecast floored at the held-out detection limit), VAF-scale MAE and, where counts exist, binomial NLL. Never use the eps-clipped log error.
- Evaluate only prospectively ascertained trajectories (training window already >= 1%); variants ascertained by the held-out wave leak the future.
- Point forecasting is frozen on current cohorts: reference = carry-forward + cohort-aware conformal intervals. No candidate forecaster transports across cohorts.

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
18_censoring_aware_growth.ipynb   (v2.5; uses real_serial_unfiltered.csv)
        |
        v
19_detection_aware_forecast_audit.ipynb   (v2.6; re-scores v2.4/v2.5 forecasters)
```

Known user execution state:

- Notebook 12: completed
- Notebook 13: completed
- Notebook 14: available; execution status may need confirmation
- Notebook 15: available; execution status may need confirmation
- Notebook 16: outputs present on Drive (2026-10-07 07:03). Claude reproduced them exactly from `real_serial_filtered.csv`. The saved `.ipynb` has no cell outputs.
- Notebook 17: added by Claude on 2026-10-07; re-run by ChatGPT (v2.5) and reproduced V24 exactly.
- Notebook 18: added by ChatGPT (v2.5).
- Notebook 19: added by Claude on 2026-10-07 (v2.6). It ran end-to-end in a clean Python 3 kernel outside Colab and has not yet been run in Colab.

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
   - v2.5: sub-floor calls recovered (within the 1%-ascertained variant universe); simple censored model rejected.
   - v2.6: re-scored with detection-aware metrics; no forecaster transports across cohorts. **Point forecasting frozen.** Next: measurement-robustness of confirmatory dynamics features.
3. Add predictive intervals for future VAF, not only point predictions.
   - v2.4: split-conformal 80% intervals reach about 0.80 pooled coverage.
   - Coverage is uneven by cohort (SardiNIA about 0.73–0.76), so the intervals need cohort-aware scaling.
   - v2.5: cohort-aware conformal scaling fixes this (SardiNIA 0.743 -> 0.795) and is retained for research reporting.
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

## 16A. AI roles and independent code-review gate (effective 2026-10-09)

**Owner decision: Claude Code implements; ChatGPT independently reviews.** This supersedes earlier instructions that both assistants may freely implement changes. It does not change the scientific constraints or unlock outcome-model training.

### Claude Code — implementation owner

- Write and modify code, notebooks, versioned model specifications, tests, and reproducible artifacts.
- Before coding, read this file and the newest audit; state the planned change and its acceptance criteria.
- Keep previously audited versions immutable; make focused, inspectable commits or diffs.
- Execute appropriate tests and checks. Never report Colab execution unless it actually ran in Colab.
- Submit a review package including: goal, changed files, Git commit hash or diff, data input/output schema, exact commands executed, pass/fail test logs, reproducibility metrics, provenance and scientific limitations.
- Resolve reviewer findings explicitly; do not mark a change accepted yourself or silently alter criteria after inspecting results.

### ChatGPT — independent reviewer, not primary code author

- Default to reviewing changes, not implementing new project code. Only write implementation code if the owner explicitly changes this assignment.
- Read actual diffs and inspect the affected call sites, source data contracts, tests, and newest audit. Do not accept an implementation based solely on Claude's summary.
- Prioritize in order: **scientific validity**, label/time/ascertainment leakage, assay LoD/censoring, patient grouping, statistical calibration and uncertainty, negative-result reporting, reproducibility, correctness, tests, maintainability, and documentation.
- Reproduce key results where files and a runnable environment are available; distinguish independently executed checks from claimed tests or results that cannot be reproduced here.
- Produce a concrete report with severity **BLOCKER / MAJOR / MINOR / SUGGESTION**, file/line references where possible, evidence, impact, and an actionable request.
- End each review with exactly one disposition: **APPROVE**, **REQUEST CHANGES**, or **BLOCKED — INSUFFICIENT EVIDENCE**. Approval means acceptable for the stated research scope, **not** clinical clearance, journal acceptance, or authorization to deploy.

### Merge/promotion gate

1. Claude submits a review-ready change set with test evidence and no hidden edits to frozen baselines.
2. ChatGPT reviews independently; BLOCKER/MAJOR findings must be addressed before approval.
3. Claude fixes the issues and submits a new diff or commit for another review.
4. The project owner decides whether to merge/release. Neither AI should claim that a review alone authorizes clinical use.
5. Update the newest audit and AI handoff with **review status and outstanding findings**. Preserve negative results.

**Default constraints remain:** point forecasting frozen; no fusion/outcome training or leukemia prediction claims on current cohorts; keep exploratory/confirmatory decisions pre-specified; no invented below-LoD values, event labels, timepoints, or measurements.

---

## 17. AI HANDOFF LOG

Use newest entries first.

### 2026-10-09 — Owner workflow decision: Claude implements, ChatGPT reviews

**Scope:** Applies to all future LeukemiPrediag development. Claude Code is the code author and test executor; ChatGPT serves as independent scientific/technical code reviewer. The owner controls merging and promotion. This is a collaboration-process change only; no algorithm, baseline, data, or clinical claim was changed.

**Required next action for Claude:** Read section 16A before the next change. Include an inspectable diff/commit, data contract, tests, metrics and limitations in every review handoff. Do not interpret silence as approval. Await an **APPROVE** disposition before asking the owner to promote a change.

### 2026-10-09 — ChatGPT (v2.7 assay-aware feature robustness and publication readiness)

**What / why:** Improved *measurement validity* of the existing core without tuning any new point forecaster or using leukemia-outcome labels. V2.6 revealed that many LBC numeric VAFs were below per-row assay LoD; treating these as ordinary log-VAF points can cause artificial growth/candidate-clone signals. The v2.3 feature matrix therefore needed an ascertainment/measurement robustness audit before considering publication or clinical pilot design.

**New files:** `src/feature_robustness_v27.py`, `src/finalize_v27.py`, `src/model_spec_v27.py`, `src/test_feature_robustness_v27.py`, `notebooks/20_feature_robustness_v27.ipynb`, `V27_MODEL_AUDIT.md`, and `RESEARCH_TO_CLINICAL_VALIDATION_PLAN.md`.

**Inputs:** Frozen `data/processed/v2/real_serial_filtered.csv` (V0) and `real_serial_unfiltered.csv` (V1/V2). No earlier feature, clone, forecasting or original time columns are overwritten. Keep original and APPROXIMATE wave-age GSE288742 sensitivity runs separate.

**Scientific QC and deviation from naive numeric-floor inference:** The raw `vaf_lod_floored` field is saved in the v2.7 audit-only provenance table, but below-LoD rows are **NOT** fed into the unchanged log-VAF clone inference as if they were measured VAFs. Only >=3 quantifiable timepoints per variant (LoD known, VAF >= LoD; LBC DP >=500) are allowed for retrospective growth/structure features. Variant selection for V2 requires reliable >=1% detection strictly before the patient last sampled wave. This is more conservative than numeric-floor fitting; it prevents fictitious co-movement among thousands of all-floored variants. No missing assay LoD is filled with a cohort median.

**Validation and dataset attrition:** 18,081 unfiltered input rows; 866 missing LoD (116 GSE178936, 750 GSE288742); 10 additional extremely poor-quality/uninformative LoD values; 4,403 quantifiable rows after LoD and LBC depth QC. Inference cohorts: V0 1,483 rows / 265 patients; V1 reliable 2,836 rows / 362 patients; V2 pre-index 1,904 rows / 295 patients. Matched V0 vs V2 = 49 GSE178936, 4 underpowered GSE288742, 180 SardiNIA patients. Unmodified v2.1 clone inference and v2.2 progression features were run for each dataset with 500 draws and seed 17, including all-but-last-patient-wave sensitivity. Unit tests: 4/4 passed.

**Pre-specified internal Spearman rho >= 0.8 results:** 2/9 v2.3 core features passed in the two adequately sized cohorts: `max_delta_vaf_per_year` (rho 0.914 GSE178936 / 0.920 SardiNIA), and `n_variants_crossed_vaf_ge_20pct` (rho 1.000 / 0.931). The VAF crossing indicator is sparse, 80th-percentile ties invalidate a top-quintile concordance claim, and it is NOT a clinical threshold. The other seven features, especially pair/clone architecture and posterior growth, are **exploratory_v27**, not safe confirmatory defaults. This is a negative but important finding; do not hide it. The all-vs-prior-wave V2 rank correlations for max raw growth and the 20% crossing count are ~0.826 and ~0.760, respectively; see full per-feature table and 95% patient bootstrap CI in audit. The approximate GSE288742 time mapping does not change the classification; do NOT call it verified individual collection dates.

**Outputs:** `data/processed/v2/feature_robustness_v27/` with per-version patient feature matrices, per-row LoD provenance, Spearman/top-quintile outputs, prior-wave stability, bootstrap CIs and input SHA256 manifest. `model_spec.py` (v2.3) remains unchanged; new `model_spec_v27.py` records preliminary, INTERNAL ranking only. Point forecasting stays FROZEN.

**Limitations:** Selected-locus rather than all-locus data, SardiNIA operational (not laboratory-validated) LoD, missing/poor quality LBC LoD, no per-patient GSE288742 collection dates, insufficient reliable GSE288742 matching patients, co-movement not true clone phylogeny, no outcomes, no external analytical validation. The present data cannot establish clinical readiness or AML/MDS prognosis.

**Next task for Claude:** Read V27_MODEL_AUDIT.md and RESEARCH_TO_CLINICAL_VALIDATION_PLAN.md. Verify notebook 20 cleanly runs in Colab and that exact QC/provenance and rank decisions reproduce. Secure a second independent cohort with true all-locus AO/DP/LoD and exact ages/dates, validate `max_delta_vaf_per_year` in a locked protocol, and only then discuss an ethics-approved non-interventional shadow-mode pilot. Do not tune point forecasting or train AML/MDS prediction from present cohorts.


### 2026-10-07 — Claude (v2.6 detection-aware / ascertainment-aware forecast audit)

**What / why:** I continued from ChatGPT v2.5. Before building on it, I checked the v2.5 evaluation itself. It had two flaws that make its point-forecast numbers uninterpretable, so I re-scored all forecasters. No new forecaster was added.

**Files added:**

- `src/forecast_scoring_v26.py`
- `notebooks/19_detection_aware_forecast_audit.py`
- `notebooks/19_detection_aware_forecast_audit.ipynb`
- `V26_MODEL_AUDIT.md`

**Files modified:** only this file, which gets this entry plus small status notes in §5, §10, §11 and §15.

**Not modified:** all v2.3–v2.5 code and outputs.

**Inputs:**

- `data/processed/v2/real_serial_unfiltered.csv`
- `real_serial_filtered.csv` (used only to flag the trajectories that developed the v2.4 rule)
- `growth_censored_v25/v25_crossfit_forecasts.csv` (the v2.5 censored predictions, reused unchanged)

**Outputs:** `data/processed/v2/forecast_scoring_v26/`

**Flaws found in v2.5:**

1. **The eps-clipped log error is dominated by exact zeros.** 12% of trajectories have a zero last or held-out VAF, and they produce 48.5% of the carry-forward error. That is why every model scored ≈1.17 with CIs of ±0.1.
2. **Future ascertainment.** 30% of evaluated trajectories (1,508 of 5,042) never reached 1% before the held-out wave. 68% of those are ≥ 1% at the held-out wave, versus 24% for the rest. Their inclusion leaks the future.
3. **Most LBC data are below the assay limit.** 87% of GSE178936 rows and 75% of GSE288742 rows are below the row's own 95% minimum detectable AF (`X95MDAF`, median ~1.5%). In the prospective set, 94.5% of LBC held-out values are below that limit.

**v2.6 protocol (fixed before computing results):**

- Primary score: detection-aware error, with truth and forecast floored at the held-out detection limit.
- Secondary scores: VAF-scale MAE, and binomial NLL of the held-out alt-read count (LBC).
- Primary set: prospectively ascertained trajectories (training window ≥ 1%): 3,534 trajectories, 367 patients.
- Validation unchanged: patient-held-out CV, 20 seeds, LOCO, cluster bootstrap.

**Results (primary set, detection-aware error):**

| model | score |
|---|---|
| carry-forward | 0.1501 |
| global-alpha | 0.1501 (alpha = 0 in every fold) |
| drift_global | **0.1129** (mu ≈ −0.07/yr; wins 20/20 seeds) |
| alpha_vaf_stratum | 0.1484 |
| censored_growth | 0.1323 |

- **No model transports across cohorts.** In SardiNIA, drift_global gives 0.643 (LOCO 0.704) versus carry-forward 0.520. censored_growth is better than carry-forward in GSE178936 (−0.041 [−0.090, −0.004]) and worse in SardiNIA.
- **On LBC counts, censored_growth is the best scorer.** Binomial NLL vs carry-forward: −2.78 [−4.22, −1.19]. Versus drift_global it is not significant overall.
- **Frozen 2% `alpha_vaf_stratum` rule on 3,111 trajectories not used to develop it:**
  - overall Δ vs carry-forward −0.0001 [−0.0019, +0.0013];
  - only 52 new trajectories have a last VAF ≥ 2%, giving Δ −0.005 [−0.10, +0.07].
  - The rule is **unconfirmed**: current data are too underpowered to test it.

**Decision:**

- **Adopt v2.6 scoring for all unfiltered and count data.** Never use the eps-clipped log error on data containing zeros. Always use prospective ascertainment.
- **Freeze point forecasting.** The reference stays carry-forward with v2.5 cohort-aware conformal intervals.
- drift_global, censored_growth and alpha_vaf_stratum stay exploratory and regime-specific. None meets the acceptance rule.
- Forecast skill on these cohorts reflects the measurement regime: LBC is sub-LoD regression to the mean, SardiNIA is above the limit. It does not reflect clone biology.
- Treat LBC sub-1% values as sub-LoD, never as evidence of growth or shrinkage.

**Concerns / housekeeping:**

- **Duplicate files on Drive:** `V25_MODEL_AUDIT.md` ×2, `DATA_UNFILTERED_AUDIT.md` ×2, `real_serial_unfiltered.csv` ×2. The owner should delete the older copies.
- **Duplicated v2.5 log entries.** There were two v2.5 entries in this log; both are kept verbatim below.
- **GSE288742 time axis is unverified.** It uses `time = wave × 3 years`. LBC1921 waves are reportedly unequally spaced (ages ~79, 83, 87, 90, 92). Verify; if true, GSE288742 horizons and slopes have been mis-scaled since v2.

**Next recommendation:**

> Stop point-forecast model development on these cohorts.
>
> The next model task is **measurement robustness of the confirmatory dynamics features**. Recompute the v2.3 core features from prospectively ascertained, detection-aware trajectories. Check rank stability against the 1%-filtered features, using Spearman correlation and patient-level agreement. Promote only the features that are stable.
>
> In parallel, pursue the data that would actually move the model:
> - assay-level all-locus pileups;
> - counts for an above-LoD cohort;
> - any independent serial CH cohort;
> - outcome-linked data.

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
