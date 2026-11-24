# LeukemiPrediag v2.8 — Diagram pipeline: Review Package for ChatGPT (CLAUDE.md §16A)

**Implementer:** Claude. **Reviewer:** ChatGPT (independent). **Date:** 2026-10-26.
**Review status:** SUBMITTED — awaiting review. Not approved; not merged.
**Branch:** `v2.8-diagram-pipeline`, baseline `50d91a3` (main).
**Scope:**
- The code is restructured to follow the owner's architecture diagram, stages 1→11.
- No frozen module was edited.
- No feature definition, threshold or classification was changed.
- No outcome label was used.
- Point forecasting stays frozen.
- Research use only.

## 1. Goal and acceptance criteria (stated before running)

The owner asked that the code follow the architecture diagram he supplied. Acceptance criteria:

1. Every box in the diagram maps to a named function. Each stage appears in a run manifest with an honest status: `done`, `partial`, `not_run`, `fusion_only` or `gated`.
2. With default settings, on the real tables:
   - stages 2–3 equal v2.7 `prepare_versions` exactly;
   - the stage 4–7a features equal v2.7 `compute_features` exactly.
3. The frozen modules are unchanged (AST fingerprints), and `fusion_model.py` (v2.3) is used unmodified.
4. Stages that need outcomes (the ablation and the progression model) cannot run on the development cohorts. This is enforced in code, not by convention.
5. Nothing is imputed or invented:
   - missing CBC values stay NaN;
   - a patient without an embeddable variant gets a zero vector **and** `has_molecular = 0`.
6. A new assay source must be declared in a registry before its data are processed.

## 2. Diagram → code mapping

| Stage | Diagram box | Code | Status on current data |
|---|---|---|---|
| 1 | Longitudinal sequencing data | `assay_registry.validate_tables` (input contract + registry check) | done: 18,081 rows, 497 patients, 3 sources |
| 2 | Quality control and preprocessing | `assay_registry.apply_qc` / `flag_measurements` / `qc_audit` | done: 4,403 quantifiable, 12,790 sub-LoD, 866 missing LoD, 10 invalid LoD, 1,294 low depth |
| 3 | Variant-level time series | `V2_PREINDEX_RELIABLE` from `apply_qc` | done: 1,904 rows, 295 patients, 489 variants |
| 4 | Probabilistic growth estimation | `clone_inference.trajectory_posteriors` (frozen) | done: 489 trajectories |
| 5 | Pairwise variant support | `clone_inference.pairwise_clone_support` (frozen) | done: 291 pairs evaluated |
| 6 | Candidate clone inference | `clone_inference.infer_clone_assignments` (frozen) | done: 489 variants assigned |
| 7a | Clone-level growth / fitness | `clone_posteriors`, `patient_clone_features`, `progression_features.patient_progression_features` (all frozen) | done: 486 clones; 295 patients × 9 dynamics features |
| 7b | BioNeMo / ESM-2 molecular features | `molecular_features.clone_molecular_features`; `compute_embedding_table` → existing `embedding.py` | with `--embeddings`: done, 2,561 dims, 179/295 patients have a vector. Without: `not_run` |
| 7c | Clinical and genetic baseline | `clinical_baseline.build_static_baseline` | partial: genetic features 100%, age 80% (SardiNIA only), rdw/mcv/n_cytopenias/ccus 0% |
| 8 | Feature fusion and ablation | fusion: `fusion_ablation.assemble_ladder`; ablation: `run_ablation` (gated) | fusion_only |
| 9 | M0 / M1 / M2 | `fusion_model.build_feature_bundle` (v2.3, unmodified) | M0 = 5 features, M1 = 14, M2 = 2,575 (with embeddings) |
| 10 | Research outputs: clonal fitness and uncertainty | `pipeline_v28.research_outputs` | done: clonal_fitness, variant_growth, patient_dynamics |
| 11 | Future progression model | `fusion_ablation.train_progression_model` (gated) | gated |

Design decisions that the diagram does not show:

- **ESM-2 inference (7b).** It is not run inside the pipeline because it needs a GPU and the model weights. The pipeline consumes the embedding table that `embedding.py` produces.
- **Clone-aware aggregation (7b).** This is the arrow from clone inference into 7b:
  - each clone's vector is the mean of its members' ESM-2 deltas, weighted by each member's VAF at the clone's last time point;
  - each patient gets a clone-size-weighted mean, plus the dominant clone's vector, plus a coverage flag.
- **Arrows 8→9→11.** Labels can enter only through `OutcomeGate`, which requires all of the following:
  - a named outcome cohort that is **not** GSE178936, GSE288742 or SardiNIA;
  - a label definition;
  - a pre-registered protocol id;
  - a declaration that the cohort is independent of feature development;
  - owner approval.
- **Arrow 9→10.** An ablation summary appears in stage 10 only when the gate is open.

## 3. Change set

| Commit | Content |
|---|---|
| `aa77b75` | Adds `src/fusion_model.py` exactly as on Drive. It was on Drive but missing from the repo. 20,204 bytes; AST fingerprint `6e7cd4ff…cdc` is locked in a test. |
| `699dcdb` | Stage 2: `src/assay_registry.py` |
| `45b4534` | Stage 7b: `src/molecular_features.py` |
| `03c76df` | Stage 7c: `src/clinical_baseline.py` |
| `0b59e27` | Stages 8–11: `src/fusion_ablation.py` |
| `f5d3475` | Orchestrator and CLI: `src/pipeline_v28.py` |
| `26c2793` | `tests/test_v28_pipeline.py` (39 tests) and `requirements-lock.txt` |

- **Combined diff:** `review/v28_combined.diff` = `git diff 50d91a3 HEAD -- src tests requirements-lock.txt`.
  - 8 files, +1,845 / −0.
  - sha256 `e36498d62947363015c33e7c087700e7af1ac220653d0e659f32370d93634f37`.
- **Patch series:** `review/v28_patches/0001…0007`. Apply with `git am` on `50d91a3`.
- **Modified files:** none, apart from `requirements-lock.txt`, which gains scikit-learn 1.5.2, joblib 1.6.0, threadpoolctl 3.7.0 and cloudpickle 3.1.2 (all needed by `fusion_model.py`).

## 4. Inputs and outputs

**Inputs (sha256)**

| File | sha256 |
|---|---|
| `data/processed/v2/real_serial_filtered.csv` | `c24c85c87b4eedca4e9ff51eec7e2869ee6f968956dc9089b63ef1a04aec4918` |
| `data/processed/v2/real_serial_unfiltered.csv` | `0825640be93ed574b716e3129ec3834c3e07c14eef4abb8cd140a6d295a813cb` |
| `data/processed/v2_molecular/esm2_delta_table.pkl` | `0272776ef18786ba8cf8269bb559e311fc5c3489d27bac3f0c8e7ea48ff30c14` |

`esm2_delta_table.pkl` contains only the `mut_key` and `delta` columns, taken from the owner's Drive file `lbc_sardinia_with_delta.pkl`. Its other columns (`s`, `s_z`, `cohort`, …) were not used. `data/` is gitignored, so this table is not in the commits.

**Outputs, written to `--outdir`**

| File | Content |
|---|---|
| `stage2_qc_provenance.csv` | per-measurement QC flags |
| `stage3_trajectories.csv` | the analysis set |
| `stage4_growth_posteriors.csv` | per-variant growth posteriors |
| `stage5_pair_support.csv` | pairwise co-movement support |
| `stage6_clone_assignments.csv` | candidate clone assignments |
| `stage7a_clone_fitness.csv` | clone-level growth / fitness |
| `stage7b_molecular_clones.csv`, `stage7b_molecular_coverage.csv`, `stage7b_molecular_matrix.npz` | only with embeddings |
| `stage7c_static_baseline.csv` | static baseline table |
| `stage9_model_availability.csv` | which of M0/M1/M2 can be built |
| `stage10_*.csv` | research outputs |
| `assay_registry.csv` | the registry used |
| `pipeline_manifest.json` | input hashes, seed, draws, per-stage status and counts |

## 5. Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt
.venv/bin/python -m pytest tests -v -rs
(cd src && ../.venv/bin/python -m unittest -v test_feature_robustness_v27)
cd src
../.venv/bin/python pipeline_v28.py --filtered ../data/processed/v2/real_serial_filtered.csv \
    --unfiltered ../data/processed/v2/real_serial_unfiltered.csv --outdir OUT_A
../.venv/bin/python pipeline_v28.py --filtered ../data/processed/v2/real_serial_filtered.csv \
    --unfiltered ../data/processed/v2/real_serial_unfiltered.csv \
    --embeddings ../data/processed/v2_molecular/esm2_delta_table.pkl --outdir OUT_B
```

## 6. Results (logs in `review/v28_logs/`)

**Test runs**

- **pytest:** 64 passed, 2 skipped (`pytest_full.log`).
  - The 2 skips are the existing opt-in reference-run tests, which need `LP_V27_OUTDIR`.
  - 39 of the passing tests are new.
- **unittest (`test_feature_robustness_v27`):** 4 OK (`unittest_v27.log`).

**Equivalence on the real tables (both pass)**

- `apply_qc(v27_registry(...))` equals `prepare_versions(...)` for all 4 data sets, with the approximate time axis on and off.
- `dynamics_stage` equals `compute_features`: 295 × 11, identical frames. Checked at 100 draws, seed 17.

**Governance and correctness tests (all pass)**

- An unregistered source is rejected.
- A new registered source gets its own depth gate.
- A depth gate without per-call depth is refused.
- The gate blocks each development cohort and each missing condition.
- `run_ablation`, `train_progression_model` and `run()` raise `OutcomeGateError` when the gate is closed.
- The ablation runs on synthetic data when the gate is open.
- Missing labels are never filled.
- Clinical validation rejects out-of-range values, duplicate rows, and label-like columns such as `aml_diagnosis`.
- Clone weighting is checked against hand-computed values: weights 0.75 / 0.25, size-weighted mean, dominant clone, and zeros with flag 0.

**Other checks**

- **Determinism:** two separate CLI runs with seed 17 and 500 draws give byte-identical files for all 9 shared stage tables (`determinism_two_runs.txt`).
- **Frozen modules:** all 6 AST fingerprints are unchanged, and `fusion_model.py` equals the locked fingerprint.
- **Runtime:** about 11 s per full run (500 draws) in the sandbox.

## 7. Limitations and open risks

1. **Execution environment.** Nothing was run in Colab. Everything ran in Claude's Linux sandbox (Python 3.12.3, the pinned lock file).
2. **ESM-2 computation untested.** `compute_embedding_table` → `embedding.py` was not executed because torch and the weights are not available. `embedding.py` lives on Drive and is not in this repo, so it must be on `PYTHONPATH` for that function. The embedding table was produced earlier by the owner. I have not verified how it was made: model size, layer, protein isoform.
3. **Pickle loading.** `load_embedding_table` reads pickles, so only trusted files should be passed. A switch to `.npz` or parquet is recommended.
4. **M2 dimensionality.** M2 has 2,575 features for 295 patients, and only 179 of them have a molecular vector. Overfitting risk is high. The AdaptivePCA inside `fusion_model` acts within folds, but any M2 result must be compared with `random_embedding_null`.
5. **M0 is genetic only.** No cohort has CBC or clinical data, so M0 here has 5 features. It is not the CHRS-style baseline the diagram intends.
6. **Age missingness follows cohort.** Age exists only for SardiNIA; LBC wave mean ages are never used as individual ages. Any model must handle this inside the folds.
7. **No outcome results.** No AML/MDS labels exist, and stages 8 (ablation) and 11 remain gated. Nothing in this package supports a leukemia, AML or MDS claim.
8. **fusion_model.py provenance.** It was reconstructed from Drive base64 and the byte size matches. Drive has no history, so I cannot prove this is the version behind earlier results.
9. **E1 order dependence unchanged.** The order dependence measured in `V28_SCALING_PROPOSAL.md` is behaviour of the frozen clone inference and is not addressed here.
10. **fusion_model.py is not frozen.** It is not in `FROZEN_MODULES`. Its fingerprint is locked in the new test file and `tests/frozen_fingerprints.json` was not edited. The reviewer should decide whether to freeze it.

## 8. Questions for the reviewer

1. Are the five `OutcomeGate` conditions sufficient, or should the gate also check label timing (outcome after index)?
2. Should `fusion_model.py` join `FROZEN_MODULES`?
3. Is the clone weighting acceptable? It uses VAF at the clone's last time and the median member VAF as clone size. The alternative is posterior-based weights.
4. Should the embedding-table format move from pickle to `.npz`?

## 9. Explicitly not changed

- `clone_inference.py`, `progression_features.py`, `model_spec.py`, `model_spec_v27.py`, `feature_robustness_v27.py`, `finalize_v27.py`, `fusion_model.py`.
- `embedding.py` on Drive (not touched, and not copied into the repo).
- All thresholds: LoD range, depth 500, prior VAF 1%, ≥ 3 timepoints, pair support 0.80.
- The feature classification.
- The point forecaster.

## 10. Proposed CLAUDE.md "AI HANDOFF LOG" entry (for the owner to paste)

> **2026-10-26 — Claude — v2.8 diagram pipeline (SUBMITTED, not merged).**
> Restructured the code to follow the owner's architecture diagram (stages 1–11) on branch `v2.8-diagram-pipeline`. Commits `aa77b75…3ccec34`, new files only.
>
> New modules:
> - `assay_registry.py` (stage 2);
> - `molecular_features.py` (7b, clone-aware ESM-2);
> - `clinical_baseline.py` (7c);
> - `fusion_ablation.py` (8–11, with `OutcomeGate`);
> - `pipeline_v28.py` (CLI and manifest).
>
> Verified on the real tables:
> - QC equals v2.7 `prepare_versions`;
> - dynamics equal v2.7 `compute_features`;
> - frozen modules unchanged;
> - 64 pytest tests pass (2 opt-in skips) and 4 unittest tests pass.
>
> Not done, by design or by missing data:
> - ESM-2 computation not run (no GPU);
> - no CBC data;
> - ablation and progression model gated, since no independent outcome cohort exists.
>
> Not run in Colab. Review package: `review/V28_DIAGRAM_PIPELINE_REVIEW.md`.
