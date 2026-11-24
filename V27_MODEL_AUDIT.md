# LeukemiPrediag v2.7 — Assay-Aware Feature Measurement Robustness Audit

**Status:** label-free, research-use-only, not clinically validated. **Date:** 2026-10-09. **Point forecasting FROZEN.** No original v2.x module was changed.

## Research question and scope

Test whether the pre-frozen 9 longitudinal v2.3 core features remain rank-stable when the assay's limits of detection (LoD), low read depth, and prospectively observable variants are respected. This is an **observational measurement-validity audit**, not AML/MDS prediction.

## Versioned files

- `src/feature_robustness_v27.py`: versioned QC/coordination wrapper, calls unmodified `clone_inference.py` and `progression_features.py`.
- `src/finalize_v27.py`: derive descriptive comparisons, patient bootstrap rho CIs and tie-aware top-quintile metrics from stored feature tables; no fitting.
- `src/model_spec_v27.py`: NEW governance spec; frozen v2.3 `model_spec.py` untouched.
- `src/test_feature_robustness_v27.py`: four anti-leakage/LoD/tie/time unit checks.
- `notebooks/20_feature_robustness_v27.ipynb`: Colab-ready reproducible job.
- `data/processed/v2/feature_robustness_v27/`: output tables, SHA256 manifest and bootstrap CIs.
- `RESEARCH_TO_CLINICAL_VALIDATION_PLAN.md`: research, analytical-validation, prospective shadow-mode study gates.

## Fixed protocol

- V0: historical 1% filtered input (1,483 rows / 265 patients / 423 trajectories).
- V1-raw: all 18,081 selected-locus calls, carrying each known row's detection limit; a separate `vaf_lod_floored` column exists for **audit only**. Crucially, we **do not feed** numeric floors of censored calls to a Gaussian log-VAF clone model: that would manufacture growth and pairwise similarity at the detection limit.
- **V1-RELIABLE**: ≥3 distinct **quantifiable** timepoints per patient-variant; each row must have a valid per-row LoD, VAF ≥ LoD; LBC has depth ≥500. No missing LoD is guessed; the original SardiNIA operational 0.2% LoD is retained as an explicitly operational, non-validated assumption.
- **V2-PREINDEX-RELIABLE**: V1 plus at least one **quantifiable** VAF ≥1% strictly earlier than the final wave of that **patient**, computed with no future wave. In all-minus-last analysis these criteria are recomputed from truncated input.
- Same frozen v2.1/2.2 computation, 500 posterior draws, seed 17, pair support 0.8 across versions. No parameter is tuned using new results.
- Correlation by patient-level paired IDs; Spearman, top-quintile agreement only if unique cutoff (ties produce NA), 1,000 patient-bootstrap rho replicates; within-patient all-waves vs all-but-last **patient** wave.
- Locked promotion criterion: Spearman ≥0.8 V0 vs V2 in **every cohort with at least 20 matched patients**, and at least two adequately powered cohorts. Within-patient rank stability must be reported, even if weak. The two-cohort requirement is conservative anti-vacuity; not a claim of independent external replication.

**Why V1 differs from requested naive numeric-floor fit:** Old `clone_inference.py` treats a numerical value as quantitative. A below-LoD floor is instead an upper bound/censored datum. Fitting all raw floors as if observations would violate the rule that sub-LoD measurements do not prove clone expansion/decline, and entails up to ~153k potentially spurious within-patient variant pairs. We therefore preserve the complete floor audit, *but only analyze quantitatively identifiable trajectories* with unchanged inference. A new censor-aware clone inference could be tested in a separately specified project, not silently fitted here.

## Data QC and attrition

18,081 original rows from 497 raw patient IDs; 866 LoD-**missing** rows (GSE178936=116, GSE288742=750), **10 additional** LoD values >50% / uninformative at ultra-low read depth, and 1,294 LBC rows with depth <500 (overlapping categories, do not sum as mutually exclusive). **12,790** rows are below a known LoD; **4,403** pass both known-LoD/above-LoD and depth criteria. The source files remain **1%-ascertained variant universes**; this is not an all-locus observation matrix.

| Version | Rows used for frozen inference | Patients | Variants with ≥3 reliable timepoints | Patients with prior-wave features |
|---|---:|---:|---:|---:|
| V0 | 1,483 | 265 | 423 | 265 (many posterior features undefined with two points) |
| V1 reliable | 2,836 | 362 | 743 | 244 |
| V2 pre-index reliable | 1,904 | 295 | 489 | 186 |

**Paired V0 vs V2:** 233 patients overall (49 GSE178936; **4 GSE288742, not powered**; 180 SardiNIA). Report differential analyzability and selection bias in the manuscript. No zero/absent variant is inferred from a missing row.

## Core feature rank robustness — ORIGINAL time axis

Spearman rho for V0 vs V2, paired patient IDs:

| feature                            | LBC1936      | SardiNIA      | LBC1921      |
|:-----------------------------------|:-------------|:--------------|:-------------|
| posterior_max_growth               | 0.764 (n=49) | 0.664 (n=180) | 0.000 (n=4)  |
| posterior_median_growth_sd         | 0.784 (n=49) | 0.649 (n=180) | -0.800 (n=4) |
| max_delta_vaf_per_year             | 0.914 (n=49) | 0.920 (n=180) | -1.000 (n=4) |
| n_supported_variant_pairs          | 0.700 (n=49) | 0.347 (n=180) | NA (n=4)     |
| fraction_evaluated_pairs_supported | 0.729 (n=17) | 0.299 (n=53)  | NA (n=0)     |
| n_multivariant_clones              | 0.700 (n=49) | 0.352 (n=180) | NA (n=4)     |
| clone_vaf_entropy                  | 0.355 (n=49) | 0.639 (n=180) | NA (n=4)     |
| clone_growth_dispersion            | 0.357 (n=49) | 0.510 (n=180) | NA (n=4)     |
| n_variants_crossed_vaf_ge_20pct    | 1.000 (n=49) | 0.931 (n=180) | NA (n=4)     |

The exact `rho≥0.8` rule yields **2 of 9 internally rank-stable features**:

1. `max_delta_vaf_per_year`: rho 0.914 (GSE178936 n=49) and 0.920 (SardiNIA n=180). 95% bootstrap CIs approximately [0.792, 0.983] and [0.861, 0.963]. **Candidate to prioritize for external analytical replication**, not clinical risk.
2. `n_variants_crossed_vaf_ge_20pct`: rho 1.000 and 0.931 on the same cohorts. **Sparse/ordinal warning**: only six original GSE178936 patients and 21 original SardiNIA patients have any such crossing; the 80th-percentile cutoff is tied, so top-quintile concordance is **NA**, not 100%. This remains provisional and must not be turned into a clinical threshold.

The seven remaining core features **fail** the internal rank threshold or have insufficient estimable pairs. In particular clone candidate-pair counts, entropy and growth dispersion are assay/censoring-sensitive and should be moved to `exploratory_v27`. This is a negative but important reproducible scientific result.

### Within-patient stability (V2, all vs before final patient wave)

| Feature | Spearman | Patients with both values |
|---|---:|---:|
| `posterior_max_growth` | 0.702 | 186 |
| `posterior_median_growth_sd` | 0.753 | 186 |
| `max_delta_vaf_per_year` | 0.826 | 186 |
| `n_supported_variant_pairs` | 0.346 | 186 |
| `fraction_evaluated_pairs_supported` | 0.260 | 57 |
| `n_multivariant_clones` | 0.348 | 186 |
| `clone_vaf_entropy` | 0.687 | 186 |
| `clone_growth_dispersion` | 0.666 | 186 |
| `n_variants_crossed_vaf_ge_20pct` | 0.760 | 186 |

Within-patient stability is **not the same** as V0–V2 cross-ascertainment rank agreement. A feature can be sensitive to newly observed mutations and therefore move over time even when its population ranks are reproducible. Sparse data create extra missingness; interpret these as technical limitations, not as a biological event.

## Time axis check

GSE288742/LBC1921 source TSV provides wave but not individual age or sample date. The historic code used `(wave-wave_first)*3`. A separately versioned sensitivity run uses **approximate cohort mean ages** wave 1=79.1, wave 3=86.6, wave 4=90.1, wave 5=92.1. Do NOT use these as individual dates or describe them as verified personal sampling times. The LBC1921 original-wave intervals are not necessarily equal; cohort articles describe approximately three-yearly follow-up and additional access routes for full phenotypic data.

On the **previous V0 GSE288742 trajectories (n=38)**, median observed horizon changes **12 -> 13 years**; median absolute change in estimated log-VAF slope is ~0.00206/year. Exact source-specific wave dates, if obtainable, should replace the approximate sensitivity mapping in a new preregistered analysis. v2.7 rank governance is identical under original and approximate mappings because GSE288742 has only four matched reliable patients.

## Why this is not publication-ready clinical validation

- The nine feature choices were designed on the same source cohorts; the v2.7 threshold was specified for this audit but internal robustness remains a **development analysis**, not external confirmation.
- A patient-level Spearman rank agreement is not analytical repeatability, clinical benefit or early leukemia detection. No AML/MDS events are attached.
- The source 1% ascertainment and missing LoD are major selection mechanisms. GSE288742 has insufficient reliably quantifiable paired patients.
- SardiNIA's 0.2% LoD and nominal depth are not per-call assay validation; no alt-read counts are available in the used supplement.
- Candidate clone co-movement inferred from VAF does not identify a physical single-cell clone.
- Underlying posterior/noise module is *unchanged* and **does not treat censored measurements probabilistically**. Censored rows are excluded, not modeled.
- The 20% crossing feature is rare and top-quintile non-identifiable. Continuous raw VAF slope has internal sensitivity but no prospective/outcome validation.
- Cohort mean-wave timing is approximate; exact per-patient ages/dates are not in this source TSV.

## Governance decisions

- **Preserve** v2.3 spec/code and v2.4–v2.6 outputs; do not retune point forecasters.
- Publish **all nine** feature results and time/missingness sensitivity, not just the two rank-stable features.
- `max_delta_vaf_per_year` is the primary **continuous measurement candidate** for orthogonal replication; `n_variants_crossed_vaf_ge_20pct` is a sparse secondary internal result only.
- Move seven unstable features to exploratory in the NEW `src/model_spec_v27.py` spec, without deleting source computations.
- Before any human-use research: secure exact assay LoD and per-locus AO/DP, accurate collection dates, orthogonal variant validation and an independent cohort. For a non-interventional IRB-approved shadow-mode study, follow the linked companion plan. No clinical-management influence.

**References (methodological/observational context):**

- Nature Medicine Lothian serial clonal hematopoiesis study, 2022: https://www.nature.com/articles/s41591-022-01883-3
- Public longitudinal CH follow-up and access pointers: https://pmc.ncbi.nlm.nih.gov/articles/PMC11908133/
- STROBE observational reporting: https://www.equator-network.org/reporting-guidelines/strobe/
- TRIPOD+AI for future *outcome-linked* prediction-model evaluation: https://www.bmj.com/content/385/bmj-2023-078378

Outputs and SHA256 input provenance are in `feature_robustness_v27/v27_manifest.json`.
