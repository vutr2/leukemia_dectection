# LeukemiPrediag — v2.8 proposal: scaling to more data (Claude, 2026-10-25)

Status: **PROPOSAL, no model code changed.** Needs ChatGPT review and owner approval (CLAUDE.md §16A) before implementation.
Frozen: point forecasting, v2.7 feature definitions and classification, no outcome claims, no outcome training.

## 1. Drive check (2026-10-25)
- No project file on Drive changed after the v2.7.1 submission (2026-10-09 09:45). Files modified since then are unrelated (IELTS handouts, a spending sheet, a PDF).
- No ChatGPT review disposition for v2.7.1 is on Drive; v2.7.1 tooling not merged; `feature_robustness_v27/` output folder still empty.

## 2. New evidence (frozen v2.7 code, unmodified; local CPU, Python 3.12 pinned venv; not Colab)

### E1 — Features depend on row order and on which other patients are in the file (proposed MAJOR)
`clone_inference.py` uses three dataset-wide RNG streams (`seed`, `seed+1009`, `seed+2027`) consumed in `groupby(..., sort=False)` order. A patient's Monte Carlo features therefore change when rows are reordered or when any other patient is added/removed.

Test: V0 (1,483 rows, 265 patients), `compute_features(draws=500, seed=17)`; script `review/v28_probe/order_test.py`.

| feature | reverse patient order: patients changed / max abs diff | drop 10 other patients: changed / max diff | max diff as % of between-patient SD |
|---|---|---|---|
| posterior_max_growth | 265/265, 0.0077 | 255/255, 0.0092 | ~11–14% |
| posterior_median_growth_sd | 265/265, 0.0062 | 255/255, 0.0082 | ~46–61% |
| clone_growth_dispersion | 82, 0.0071 | 77, 0.0048 | ~14–21% |
| n_supported_variant_pairs | 1 patient flips by 1 | 2 patients flip by 1 | discrete |
| max_delta_vaf_per_year, n_variants_crossed_vaf_ge_20pct, clone counts/entropy | 0 | 0 | deterministic |

The drop test also changes the pooled cohort-noise estimate (a legitimate data dependence); the reverse-order test isolates pure RNG-order dependence. Consequence: when a cohort is added, existing patients' exploratory features shift; per-cohort caching is unsafe; Monte Carlo error in `posterior_median_growth_sd` is large relative to its signal. The two confirmatory_v27 features are unaffected.

### E2 — Clone pipeline cost grows ~quadratically with variants per patient
`pairwise_clone_support` loops over all within-patient pairs and re-filters the patient frame per pair; `_complete_link_groups` rescans all cluster pairs after every merge. Script `review/v28_probe/scale_test.py` (40 background patients + one patient with V variants, 5 waves, 500 draws):

| V variants | pairs | runtime (s) |
|---|---|---|
| 10 | 45 | 1.1 |
| 40 | 780 | 3.6 |
| 80 | 3,160 | 11.2 |
| 160 | 12,720 | 42.2 |

The unfiltered table already holds patients with up to 331 variants (p95 ≈ 45). Lower-LoD assays (error-corrected sequencing) will increase variants per patient further.

### E3 — Cohort rules are hard-coded by cohort name
`feature_robustness_v27.py` line 108: `depth_qc = ~cohort.isin(['GSE178936','GSE288742']) | depth >= 500`. **Any new cohort passes depth QC automatically, whatever its depth.** The approximate time axis is hard-coded to `GSE288742`.

## 3. Proposed work (each step a separate reviewable commit)
**P0 — safe to scale (engineering, numbers must be preserved or deviations documented)**
1. Cohort registry + data contract: per-cohort config (assay type, LoD rule, depth gate, time-axis source, VAF units) and a schema validator that refuses unknown cohorts. Replaces the hard-coded names in E3. Tests: an unregistered cohort raises; v2.7 outputs reproduce exactly.
2. Faster clone pipeline with an **exact-equivalence test**: pre-index variants once, vectorize pair statistics, incremental agglomeration. Must reproduce v2.7 outputs bit-for-bit under the old RNG order before anything else changes.
3. Per-patient RNG streams (seed derived from `seed` + patient_id + variant ids) → features invariant to row order and to other patients. This changes exploratory values within Monte Carlo error, so it ships as v2.8 with a full re-run of the v2.7 robustness audit and a documented before/after table; never as a silent change. Optionally raise draws or use the closed form where available (slope of a Gaussian-perturbed log-VAF regression is itself Gaussian).
4. Freeze cohort-noise estimates per data release in a manifest (or document that they are re-estimated), so the added-data effect is explicit.
5. Parquet inputs + content-addressed cache keyed by (input sha256, code fingerprint, seed), valid only after step 3.

**P1 — scientific robustness with more data (pre-specified, exploratory until validated)**
6. Pre-register a locked validation protocol for `max_delta_vaf_per_year` (addresses v2.7.1 F1) and apply it once to an independent serial cohort.
7. Assay-aware observation model: binomial/beta-binomial on alt/total reads when counts exist; censored likelihood below LoD instead of dropping rows. Lets ECS (≈0.1–0.5%) and WES/panel (≈1–2%) data be pooled. Must beat the current baseline under patient-held-out + LOCO before adoption.
8. Revisit hierarchical gene-level growth shrinkage (rejected in v2.4 at current N) under the same acceptance gate once N grows.
9. Outcome/survival head only with independent outcome-linked data; never trained on current cohorts.

**P2 — housekeeping**: review + merge v2.7.1 tooling, upload v2.7 outputs, resolve Drive duplicates, restore GitHub access for CI.

## 4. Limitations
Runtimes are from one local CPU container, not Colab. The synthetic scaling data are not biological evidence. No model code was changed; the frozen v2.7 modules were run as published.
