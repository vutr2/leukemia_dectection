# LeukemiPrediag v2.7.1 — Review Package for ChatGPT (CLAUDE.md §16A)

**Implementer:** Claude. **Reviewer:** ChatGPT (independent). **Date:** 2026-10-09.
**Review status:** SUBMITTED — awaiting review. Not approved; not merged.
**Scope:** research software reproducibility only. No model, feature definition, threshold, classification, forecaster or outcome label was added or changed. Point forecasting stays frozen.

## 1. Goal and acceptance criteria (stated before running)

The approved task, from the CLAUDE.md v2.7 handoff, was: verify that notebook 20 runs cleanly and that the exact QC, provenance and rank decisions of v2.7 reproduce.

Acceptance criteria:

1. The v2.7 pipeline, `finalize_v27.py` and the original v2.7 unit tests run without error from a clean environment.
2. Every number locked in `V27_MODEL_AUDIT.md` is reproduced:
   - dataset sizes;
   - LoD attrition;
   - 21 V0-vs-V2 Spearman results with their paired n;
   - 9 within-patient results;
   - bootstrap CIs;
   - the feature classification.
   Spearman values must match to the audit's 3-decimal reporting (tolerance 0.0005); counts must match exactly.
3. Two runs with the same seed give identical outputs.
4. The frozen v2.x modules are semantically unchanged.
5. New tests cover the anti-leakage, LoD and governance rules.

## 2. Change set

| Item | Value |
|---|---|
| Repository | local git repo reconstructed from Drive (Drive has no git history) |
| Baseline commit | `a639a87`: v2.7 sources as published on Drive |
| Commits under review | `19bdd2af28a23a75bde0bbb07d88647cb93d05ae` (tooling + tests), `4ebaf110c1272aa7e9a737947065449c356bc413` (fix: never report stale results), `02598078b4764e3d9ca7135709ae2745985905c2` (notebook runtime note). Code HEAD = `0259807` |
| Git bundle | `lp_v271.bundle` in the review zip (`git clone lp_v271.bundle`); contains the baseline, all commits and these docs |
| Combined diff | `review/v271_combined.diff` = `git diff a639a87 0259807` over code paths (10 files, +822/−0; sha256 `dc15b504…f76ac7`) |
| Patch series | `review/patches/0001…0003-*.patch` (apply with `git am` on the baseline) |

New files only. **No existing file is modified.**

| File | Purpose |
|---|---|
| `src/repro_v27.py` | Verifier with these parts: <br>• input SHA256 check <br>• version-stable AST fingerprints of frozen modules <br>• audit-value checks <br>• same-seed determinism check <br>• environment-deviation table <br>• descriptive V0/V2 overlap diagnostic |
| `tests/test_frozen_modules.py` | Fails if any frozen module changes semantically. Comments and docstrings are ignored. |
| `tests/test_leakage_v27.py` | 11 synthetic tests of the published v2.7 rules (list below). |
| `tests/test_repro_tools.py` | The verifier must fail on drift, a changed classification, or a NaN mismatch. |
| `tests/test_reference_outputs.py` | Opt-in integration checks: input hashes, audit reproduction, determinism. |
| `tests/frozen_fingerprints.json`, `tests/conftest.py` | Locked fingerprints and shared test setup. |
| `scripts/reproduce_v27.sh` | One command: two full runs, both test suites, verification, overlap diagnostic. |
| `notebooks/21_reproduce_v27.ipynb` | Colab wrapper around the script. Asserts exit code 0 before reading any report. |
| `requirements-lock.txt` | Pinned environment used for the reference run. |

The 11 rules covered by `tests/test_leakage_v27.py`:

- V2 membership does not change when only the last-wave VAF changes.
- The pre-index rule uses the patient's last wave, not the variant's last wave.
- Rows below the LoD, rows with missing LoD and rows with an implausible LoD are never quantifiable.
- The LBC depth gate does not apply to SardiNIA.
- The approximate time axis changes only GSE288742 and keeps the original column.
- An unmapped wave raises an error.
- Patient truncation drops only the last wave.
- The policy needs 2 powered cohorts.
- The 0.80 threshold is inclusive.
- A cohort with 49 overlapping patients but only 17 paired observations fails.

## 3. Data contract

| Input | Bytes | SHA256 |
|---|---|---|
| `data/processed/v2/real_serial_filtered.csv` | 174,490 | `c24c85c8…aec4918` |
| `data/processed/v2/real_serial_unfiltered.csv` | 4,141,258 | `0825640b…813cb` |

**Required columns:**

- Both tables: `patient_id, cohort, variant_id, time_years, wave, vaf, gene, depth`.
- Unfiltered table only: also `detection_limit, detection_limit_source`.

**Outputs:**

- v2.7 outputs, unchanged in format: `run_a/` and `run_b/`.
- New:
  - `repro_report.json`: one row per check with expected, observed and passed.
  - `v0_v2_overlap_diagnostic.csv`.
  - `logs/`: environment, pipeline, finalize, unittest, pytest and verify logs.

## 4. Commands executed (from the repo root)

```bash
# reference environment (pinned)
uv venv -p python3.12 .venv && uv pip install -p .venv/bin/python numpy==2.1.3 pandas==2.2.3 scipy==1.14.1 pytest==8.3.4
bash scripts/reproduce_v27.sh .venv/bin/python outputs/v27_repro_pinned

# second environment: system Python 3.13 / numpy 2.5.3 / pandas 3.0.5 / scipy 1.18.1,
# run by executing notebooks/21_reproduce_v27.ipynb in a Jupyter kernel (nbclient)
```

The reproduction script itself runs these steps:

1. `feature_robustness_v27.py --draws 500 --seed 17`, run twice.
2. `finalize_v27.py` on each run.
3. `python -m unittest src/test_feature_robustness_v27.py`.
4. `pytest tests`.
5. `repro_v27.py --compare-with run_b`.

## 5. Results (all pass)

| Check | Reference env (py3.12, pandas 2.2.3) | Second env (py3.13, pandas 3.0.5) |
|---|---|---|
| Original v2.7 unit tests | 4/4 OK | 4/4 OK |
| v2.7.1 pytest suite | 27/27 passed | 27/27 passed |
| Verifier checks | **77/77 passed** | **77/77 passed** |
| — input SHA256 | 2/2 | 2/2 |
| — frozen-module AST fingerprints | 6/6 | 6/6 |
| — manifest seed/draws/thresholds, sizes, LoD attrition | 10/10 | 10/10 |
| — V0-vs-V2 rho and paired n vs audit | 21/21 | 21/21 |
| — within-patient rho and n vs audit | 9/9 | 9/9 |
| — bootstrap CIs vs audit | 2/2 | 2/2 |
| — classification (manifest and CSV, both axes) | 4/4 | 4/4 |
| — run A vs run B identical (22 CSVs) | 23/23 | 23/23 |
| Runtime, two full runs | 7.0 min CPU | ~9 min CPU (run concurrently) |

**Reproduced values:**

- **Sizes and attrition:**
  - V0: 1,483 rows / 265 patients / 423 variants;
  - V1: 2,836 / 362 / 743;
  - V2: 1,904 / 295 / 489;
  - 866 rows with missing LoD (116 / 750 / 0);
  - 10 invalid LoD values;
  - 1,294 low-depth LBC rows;
  - 12,790 rows below LoD;
  - 4,403 quantifiable rows.
- **`max_delta_vaf_per_year`:**
  - rho 0.914 in GSE178936 (n = 49) and 0.920 in SardiNIA (n = 180);
  - 95% CIs [0.792, 0.983] and [0.861, 0.963].
- **Classification:** `confirmatory_v27` = {`max_delta_vaf_per_year`, `n_variants_crossed_vaf_ge_20pct`} under both time axes.

**Environment sensitivity:** comparing pandas 3.0.5 with the pinned 2.2.3, only the patient feature tables differ, by at most 6.0e-16 (floating-point rounding). All statistics, CIs and decisions are identical (`review/env_sensitivity_pandas3_vs_pinned.csv`).

## 6. Findings for the reviewer

The severities below are my proposals; the reviewer assigns the final ones. Nothing below was changed in the v2.7 code, outputs or decisions.

**F1 — proposed MAJOR (scientific interpretation, not code): the main "rank-stable" result is largely true by construction.**

The descriptive diagnostic (post hoc) compares the V0 and V2 inputs:

| Cohort | Share of V2 rows also in V0 | Share of V0 rows also in V2 | Patients with an *identical* `max_delta_vaf_per_year` in V0 and V2 |
|---|---|---|---|
| GSE178936 | 66% | 46% | 31/49 (63%) |
| SardiNIA | 59% | 100% | 124/180 (69%) |

The feature is an endpoint slope (last − first) / span for the maximum variant. When that variant's first and last above-floor points are the same in both versions, the value is identical. Among patients whose value actually differs, Spearman is 0.43 (n = 18) in GSE178936 and 0.74 (n = 56) in SardiNIA. These subsets are post hoc and small, so no CI is claimed.

The locked v2.7 classification is unaffected and must not be retuned. Requested action for the audit and any manuscript:

- report this overlap alongside rho 0.914 / 0.920;
- do not describe the feature as "robust to LoD handling" without this caveat.

**F2 — proposed MAJOR (provenance): v2.7 output artifacts were never on Drive.**

- `data/processed/v2/feature_robustness_v27/` on Drive is empty: no manifest, no feature tables, no SHA256 manifest.
- The V27 audit's "outputs and SHA256 input provenance are in `v27_manifest.json`" was therefore not verifiable on Drive.
- This package supplies a reproduced manifest (`review/v27_manifest_reproduced.json`) and the full output procedure.

**F3 — MINOR: duplicate v2.7 files on Drive.** There are two copies each of:

- `feature_robustness_v27.py`, `finalize_v27.py`, `model_spec_v27.py`;
- `20_feature_robustness_v27.ipynb`;
- `RESEARCH_TO_CLINICAL_VALIDATION_PLAN.md`.

The copies are the same size, but Colab path resolution between same-named files is ambiguous.

**F4 — MINOR: "all-but-last wave" means different things across versions.**

- V0's prior run drops V0's own last *filtered* timepoint.
- V1/V2's prior runs drop the patient's last *raw* wave.

Within-patient rho is therefore not like-for-like across versions. The V2 numbers in the audit are internally consistent.

**F5 — MINOR / cosmetic: an internal provenance column is mislabelled in one run.** In the `approx_age` run, the prior-wave `prepare_versions(..., approx_time=False)` re-creates `time_years_v26_original` from already-approximated times. The column is internal only and never written out.

**F6 — MINOR: one original v2.7 test is vacuous.**

- `test_time_axis_is_versioned` uses only GSE178936, so the approximate mapping is never exercised.
- `tests/test_leakage_v27.py::test_approx_time_axis_only_touches_gse288742_and_keeps_original` covers this case.

**F7 — limitation of this package: the baseline is reconstructed.**

- Drive has no git repository and the connector exports source as text, so the baseline was reconstructed.
- Four source files and notebook 20 match the Drive byte sizes exactly.
- Three files differ only in comment/docstring whitespace:
  - `clone_inference.py`: −11 B, which equals its 11 sentence breaks (consistent with double spaces after full stops);
  - `progression_features.py`: −13 B;
  - `feature_robustness_v27.py`: −5 B.
- Two pieces of evidence for semantic identity:
  - (a) all 77 audit values reproduce exactly;
  - (b) the AST-fingerprint test will fail if the Drive originals differ in any executable code.
- **Requested reviewer check:** run `pytest tests/test_frozen_modules.py` against the Drive originals.

**F8 — limitation: not executed in Colab.** It was executed in two local Python environments only. Notebooks 20 and 21 have not been run in Colab by me. Per §16A, I do not claim a Colab run.

## 7. Scientific limitations (unchanged from v2.7; restated)

- Selected-locus (1%-ascertained) source data, not an all-locus matrix.
- SardiNIA's 0.2% LoD is operational and not validated.
- Missing or poor LBC LoD values.
- GSE288742 has 4 matched patients (underpowered) and only an approximate time axis.
- Co-movement is not phylogeny.
- No outcomes.

Internal rank stability is not external replication, analytical validation or clinical evidence. Nothing here supports AML/MDS prediction.

## 8. Requested disposition

Please review the diff and the artifacts, reproduce if possible, and return **APPROVE / REQUEST CHANGES / BLOCKED — INSUFFICIENT EVIDENCE**. In particular, rule on:

- F1: how the caveat should enter the V27 audit;
- F2: whether the reproduced manifest should become the canonical v2.7 artifact.

I will fix all BLOCKER and MAJOR findings and resubmit before asking the owner to merge.
