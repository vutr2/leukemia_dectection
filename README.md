# LeukemiPrediag v2 — Longitudinal Clonal Dynamics for Blood-Cancer Risk Research

**Status:** research use only; not a diagnostic device.

LeukemiPrediag v2 is an interpretation layer for hematology NGS data. The current scientific target is **not** a direct binary claim that a person will develop blood cancer. The primary target is to quantify how blood clones evolve over time and test whether longitudinal clonal dynamics add information beyond established clinical/genetic baselines.

## Scientific question

Given serial blood sequencing (and, when available, CBC/clinical data), estimate:

1. clone growth / fitness,
2. uncertainty in that estimate,
3. likely clonal structure (co-clonal vs competing trajectories),
4. mutation acquisition timing when identifiable,
5. whether those dynamics improve risk stratification beyond static gene/VAF/CBC baselines.

The progression endpoint remains a downstream validation task that requires outcome-linked longitudinal cohorts.

## Why v2

The original project tested mutation-level ESM-2 representations for predicting clonal fitness and for binary classification. The key result was negative but useful: on the current LBC + SardiNIA missense set, protein-language-model features did not generalize better than simple baselines. Therefore the next bottleneck is **data + target definition + clone-level dynamics**, not a larger regressor.

## Core architecture

```text
serial NGS / VCF + sampling dates + CBC + age
                |
                v
        harmonize variants
                |
                v
        patient trajectories
                |
                v
    clone structure / competition
                |
                +----> clonal fitness posterior
                +----> acquisition-time estimate
                +----> uncertainty / QC
                |
                v
     static clinical baseline (CHRS-like)
                |
                v
 longitudinal incremental-value model
                |
                v
research report -> hematologist / investigator
```

## Repository map

### Keep as core

- `src/fitness.py` — current longitudinal VAF fitness measurement and synonymous-noise calibration.
- `src/longitudinal.py` — censoring/landmark/CBC utilities for outcome-linked cohorts.
- `src/mutation_utils.py`, `src/data_processing.py`, `src/sequences.py` — ingestion and variant utilities.
- `src/embedding.py`, `src/embedding_bionemo.py` — retained as **optional mutation representation**, not the primary scientific claim.
- `notebooks/07_fitness.ipynb` — current strongest measurement analysis.

### Keep as secondary / supporting experiment

- `src/beataml.py`
- `notebooks/06_beataml_demo.ipynb`

BeatAML is post-diagnosis and therefore **cannot validate early detection**. It can be used as a separate representation/generalization experiment.

### Legacy

- `notebooks/04_train_classifier.ipynb`
- `notebooks/05_inference_demo.ipynb`
- old binary-classifier logic in `src/model.py` / `src/inference.py`

Do not delete them; treat them as v1 research history. Do not use diagnosis-vs-CHIP labels as evidence of pre-diagnostic performance.

## v2 modules to add

- `src/chrs_baseline.py` — reproducible static baseline using published CHRS-style variables.
- `src/clone_dynamics.py` — trajectory matrix construction, co-movement features, clone-group candidates, competition features, posterior-ready data structures.
- later: `src/acquisition_time.py` — estimate age/time of mutation acquisition only after the clone model is validated.
- later: `src/progression.py` — survival/landmark models on outcome-linked cohorts; compare static baseline vs +dynamics vs +variant representation.

## Evaluation ladder

Do not jump directly to “cancer probability.”

### Stage A — measurement validity

- synonymous trajectories center near zero,
- known driver growth ordering is recovered,
- sensitivity analyses across VAF/DP/wave thresholds,
- cohort-specific noise calibration,
- bootstrap confidence intervals for each trajectory / gene estimate.

### Stage B — clonal dynamics validity

- infer whether variants move together across serial samples,
- distinguish co-clonal vs competing trajectories where data support it,
- report uncertainty rather than forcing a single clone assignment,
- compare clone-aware fitness with the current independent-slope estimator.

### Stage C — clinical incremental value

On a cohort with outcomes:

1. static clinical/genetic baseline,
2. baseline + longitudinal VAF dynamics,
3. baseline + dynamics + mutation representation.

Use survival-aware metrics (C-index / time-dependent AUC / calibration / decision curves where appropriate), and report fixed-horizon sensitivity/specificity/PPV only with censoring handled correctly.

## Data priorities

Highest priority is not a new neural architecture. It is an outcome-linked longitudinal cohort containing, ideally:

- serial blood samples before diagnosis,
- mutation/VAF per time point,
- exact sampling dates or ages,
- CBC indices,
- incident myeloid malignancy and date,
- adequate follow-up for non-events.

Controlled-access sources should be pursued through an institutional/clinical partner. Public LBC/SardiNIA-style data remain useful for **dynamics measurement**, but not as proof of early-detection performance.

## Mutation representations

Missense residue-delta ESM-2 is retained as one feature family. v2 must not be missense-only. Add type-specific representations for:

- truncating / frameshift,
- splice,
- in-frame indels,
- recurrent hotspot / domain context,
- gene and pathway context.

The model should be allowed to conclude that these features add no value beyond clinical and trajectory baselines.

## Claims policy

Allowed now:

- we can measure longitudinal clonal growth on public cohorts,
- synonymous variants provide an empirical measurement-noise reference,
- current off-the-shelf missense representations did not beat simple baselines on the present small dataset,
- the project is testing whether clone dynamics add value beyond established static risk factors.

Not allowed yet:

- “detects leukemia early,”
- “predicts which carrier will get cancer,”
- clinical sensitivity/specificity without an outcome-linked pre-diagnostic validation cohort,
- treating BeatAML diagnosis-time performance as early-detection evidence.

## Near-term implementation order

1. Freeze/reproduce the current fitness result.
2. Add `chrs_baseline.py` and a baseline table schema.
3. Add `clone_dynamics.py`; run clone-aware analysis on LBC/SardiNIA.
4. Quantify whether clone-aware fitness is more stable than independent trajectory slopes.
5. Add mutation-type support beyond missense.
6. Only after obtaining outcome-linked data: train/evaluate progression models.
7. Keep a hematologist in the loop for label definition, interpretation, and any clinical-facing output.

## v2 executable fusion pipeline (Oct 2026)

New executable files:

- `src/fusion_model.py` — M0/M1/M2 ablation ladder with fold-safe imputation, scaling and adaptive PCA; calibration diagnostics; bootstrap incremental-value tests; random-embedding null control; artifact save/load.
- `notebooks/10_train_fusion.py` — CLI/Colab-friendly end-to-end trainer from prepared patient, serial VAF, mutation and BioNeMo embedding tables.
- `src/inference_v2.py` — research inference for a frozen fusion artifact. It refuses to silently replace a required BioNeMo vector with zeros.

The evaluation ladder is:

```text
M0  static clinical/genetic baseline
M1  M0 + longitudinal clone dynamics
M2  M1 + NVIDIA BioNeMo / ESM-2 molecular representation
```

For fundraising or scientific reporting, the key quantities are the incremental gains `M1-M0` and `M2-M1`, together with bootstrap confidence intervals and the random-embedding null. Do not report M2 as a validated leukemia predictor until the same frozen pipeline is evaluated on an outcome-linked pre-diagnostic cohort.


## v2.1 uncertainty-aware clone inference (Oct 2026)

New executable files:

- `src/clone_inference.py` — posterior growth-rate estimates, pairwise co-movement support, conservative complete-link candidate clone groups, clone-level common-slope posteriors and patient-level clone features.
- `notebooks/13_infer_probabilistic_clones.py` / `.ipynb` — run the real-data clone inference and export audit artifacts.
- `CLONE_INFERENCE_AUDIT.md` — frozen real-data audit and threshold-sensitivity summary.

The current real-data build contains 265 participants and 423 tracked patient-variant trajectories. At the default pair-support threshold of 0.80, v2.1 identifies 18 multi-variant candidate clone groups. Pooling variants inside these groups reduced the posterior slope SD by about 29% at the median relative to estimating member variants independently. This is a **measurement-stability result**, not evidence that the system predicts leukemia progression.

`fusion_model.py` now accepts both the original trajectory features and these clone-aware features in M1. The clinical progression experiment remains blocked until an outcome-linked pre-diagnostic cohort is available.

## v2.2 kinetics + co-mutation feature layer (Oct 2026)

The Stage-B model now has a label-free feature-engineering layer that sits above
v2.1 clone inference without changing the frozen clone-grouping algorithm:

- `src/progression_features.py` — builds patient-level research features from
  serial VAF, trajectory posteriors, pairwise co-movement support and candidate
  clone posteriors.
- `notebooks/15_engineer_progression_features.ipynb` — Colab-ready feature build
  using the existing `data/processed/v2` paths.

New feature families include:

- raw `dVAF/dt` and log-VAF growth,
- positive-growth doubling time,
- experimental log-VAF acceleration for trajectories with at least four unique
  timepoints,
- posterior growth uncertainty,
- VAF threshold indicators/crossings used only as covariates,
- candidate-clone diversity, dominance, growth dispersion and effective clone
  number proxies,
- evaluated and high-support co-mutation-pair counts,
- longitudinally supported gene-pair counts,
- configurable gene-presence flags that are *features*, not hard-coded risk
  labels.

On the current real-data build, 129/265 participants have at least one trajectory
with enough timepoints for an acceleration estimate. The current 0.80 pair-support
threshold still yields 21 high-support variant pairs and 18 patients with supported
multi-variant clone structure. These are measurement/feature-engineering results;
they do not establish leukemia-progression performance.

`fusion_model.py` now recognizes the v2.2 longitudinal features when they are
provided to M1. M0/M1/M2 progression training remains deferred until a clean
outcome-linked pre-diagnostic cohort is available.

## v2.3 model-hardening layer (Oct 2026)

The current priority is model completion and scientific hardening before clinical
claims. v2.3 adds:

- `src/model_spec.py` — compact confirmatory feature specification separated from
  the larger exploratory v2.2 feature catalogue;
- `src/temporal_validation.py` — leave-last-timepoint-out VAF forecasting and
  patient-cross-fitted growth shrinkage;
- `notebooks/16_temporal_forecast_validation.ipynb` — reproducible temporal
  forecast audit;
- `V23_MODEL_AUDIT.md` — frozen design/audit notes.

A current-data audit showed that the v2.2 discovery matrix contains substantial
feature redundancy, so `fusion_model.py` now defaults to a compact longitudinal
core rather than entering every correlated transformation into a confirmatory
model. The full feature catalogue remains available for sensitivity analyses.

Temporal validation also showed that unshrunk log-linear extrapolation is too
aggressive on the present sparse cohorts. A patient-cross-fitted shrunk-growth
forecast modestly improved aggregate next-sample VAF error versus carry-forward,
but performance was heterogeneous by cohort. This is treated as model-hardening
evidence, not clinical progression validation.
