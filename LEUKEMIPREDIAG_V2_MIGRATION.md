# LeukemiPrediag v2 Migration Plan

## Decision log

### 1. Primary scientific target
**Old:** binary malignant-transformation classifier.

**v2:** longitudinal clonal dynamics (fitness + clone structure + uncertainty), followed by outcome modeling only when a valid pre-diagnostic cohort exists.

Reason: current Drive results show negative cross-validated R² for ESM-2 fitness prediction and no suitable public pre-diagnostic outcome labels. This makes model architecture a secondary bottleneck.

### 2. Static benchmark
Every future progression experiment must include an established static clinical/genetic baseline (CHRS-like variables where the cohort supports them): age, mutation count, high-risk genotype, maximum VAF, CHIP/CCUS context, MCV, RDW, and relevant cytopenia/CBC features.

### 3. Incremental-value question
The publishable question is:

> Do longitudinal clonal dynamics improve prediction/calibration beyond static clinical/genetic risk factors?

Protein embeddings are a third-stage add-on, not the baseline.

## File-by-file status

| File | v2 status | Action |
|---|---|---|
| `src/fitness.py` | CORE | Keep; add uncertainty/bootstrap and clone-aware comparison later |
| `src/longitudinal.py` | CORE | Keep; use for outcome cohorts and CBC handling |
| `src/data_processing.py` | CORE | Keep; extend schemas for exact dates/age/CBC |
| `src/mutation_utils.py` | CORE | Keep; extend variant classes beyond missense |
| `src/embedding.py` | OPTIONAL | Keep as feature extractor |
| `src/embedding_bionemo.py` | OPTIONAL | Keep as backend; do not imply BioNeMo fixes biology/data |
| `src/model.py` | LEGACY/V1 | Freeze for reproducibility; new progression model should live separately |
| `src/inference.py` | LEGACY/V1 | Do not expose as clinical inference |
| `src/beataml.py` | SECONDARY | Keep separate from early-detection evidence |
| `08_upgrade_model.py` | NEGATIVE RESULT | Archive/retain; do not keep tuning same n≈260 missense table |
| `07_fitness.ipynb` | CORE | Reproduce and version outputs |
| `04_train_classifier.ipynb` | LEGACY/V1 | Label clearly as non-clinical/leaky fallback work |
| `05_inference_demo.ipynb` | LEGACY/V1 | Retain only as software demo |
| `06_beataml_demo.ipynb` | SECONDARY | Audit split construction; no early-detection claim |

## New modules

### `src/chrs_baseline.py`
Purpose: explicit static comparator. It should never be described as the project's proprietary model; it is a published reference baseline.

### `src/clone_dynamics.py`
First version should remain simple and inspectable:

- pivot patient × time × variant VAF,
- compute pairwise trajectory correlation/distance only when enough shared time points exist,
- propose clone groups via conservative clustering,
- derive patient-level features: number of growing clones, max fitness, fitness dispersion, co-expansion score, competition score, max VAF slope, clone dominance change,
- include uncertainty flags when assignments are underdetermined.

Do not pretend VAF alone always identifies phylogeny. The module should output candidate structure + uncertainty.

## Required experiments before progression modeling

1. Reproduce LBC fitness calibration from a clean environment.
2. Bootstrap synonymous noise floor and per-driver estimates.
3. Leave-one-participant-out / cohort sensitivity where feasible.
4. Compare independent trajectory slope vs clone-aware trajectory estimates.
5. Confirm findings separately in LBC and SardiNIA before pooled standardization.

## Progression-model evaluation contract

When a valid cohort arrives:

- patient-level split only,
- preferably site/biobank external validation,
- censoring-aware survival analysis,
- no post-index variables in the baseline,
- no diagnosis-derived labels/features leaking into pre-diagnostic prediction,
- compare static baseline -> +dynamics -> +variant representation,
- report calibration and PPV at realistic prevalence, not AUC alone,
- pre-specify the prediction horizon (e.g. 3y/5y) and index time.

## BeatAML correction

The secondary-vs-de-novo AML task is diagnosis-time classification. Myelodysplasia-related gene mutations are important diagnostic/prognostic features, but the output must not be framed as a direct CPX-351 treatment selector. Keep this study as a representation benchmark and validate against explicit clinical-history labels.

## Stop conditions

Stop or narrow the product claim if:

- longitudinal dynamics do not improve an established static baseline on external outcome data,
- calibration fails in an independent cohort,
- performance is driven almost entirely by CBC/cytopenia measured near diagnosis,
- a mutation representation adds no incremental value after dynamics + clinical baseline.

A negative result is a valid research outcome.
