# Running LeukemiPrediag on your own computer

This folder is a copy of the Google Drive project
`leukemia-prediagnose/blood-cancer-early-detection` (snapshot 2026-11-24), plus a
few small scripts that let the Colab notebooks run on a normal computer.
No model code, data or results were changed. Research use only; not a diagnostic device.

## What is included

| Item | Status |
|---|---|
| Code (`src/`, `notebooks/`, `configs/`), docs, audits, pitch files | included, byte-identical to Drive |
| Models (`models/*.joblib`, `fitness_regressor_meta.json`, `beataml_result.json`) | included |
| `models/older_versions/` | two older Drive copies that had the same file names (2026-08-28) |
| `data/raw`, `data/processed`, `data/external` | included (57 MB) |
| 19 files over 10 MB (cBioPortal mRNA / methylation / CNA matrices, `am_subset.tsv`, one TCGA mutation file; about 1.9 GB) | **not included**: download from Drive if needed (links below) |

`DRIVE_MANIFEST.tsv` lists every Drive file with its size, SHA-256, Drive ID, link and status
(257 included, 7 byte-identical duplicates kept once, 19 not included).
None of the 19 large files is read by the v2 pipeline (notebooks 12-20, v2.7, v2.8).
`python run_local.py check` prints them with their Drive links.

Keep the folder layout: `.../leukemia-prediagnose/blood-cancer-early-detection`.
The notebooks look for the project by that name.

## Setup (once)

Python 3.12 is recommended (3.10-3.13 work). Install it from python.org if needed.

* **macOS / Linux**: open a terminal in this folder and run `bash setup_local.sh`
* **Windows**: double-click `setup_local.bat`

The script creates `.venv`, installs pinned packages (`requirements-notebooks-lock.txt`),
writes the local notebook copies and checks every file against the manifest.
It should end with `Drive snapshot files intact: 257/257`.

Then activate the environment in each new terminal:

* macOS / Linux: `source .venv/bin/activate`
* Windows: `.venv\Scripts\activate`

## Commands

```
python run_local.py check        # verify files against the Drive snapshot
python run_local.py v27          # v2.7 feature-robustness run + unit tests (about 2.5 min)
python run_local.py tests        # every test suite on the current branch
python run_local.py notebooks    # (re)create notebooks/*.local.ipynb
python run_local.py v28          # v2.8 diagram pipeline (v2.8 branch only)
python run_local.py v28 --embeddings   # same, with the ESM-2 delta table (stage 7b)
```

Outputs go to `outputs/` and never overwrite the Drive results in `data/`.

### Notebooks

```
python -m jupyter lab notebooks
```

Open the `*.local.ipynb` copies, not the originals. Run them in the CLAUDE.md order:
12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20.
The local copies differ from the originals only in three ways: Google Drive mounting is skipped,
`/content/drive/MyDrive` points at the folder that holds `leukemia-prediagnose/`, and
`import pandas as pd` is added where a notebook used `pd` without importing it (notebook 13).
If you move the project folder, run `python run_local.py notebooks` again.

* Do **not** run 10/11 (fusion training / inference) until outcome-linked data exist (CLAUDE.md §11).
* The notebooks write into `data/processed/...` like they do on Drive. After re-running them,
  `check` reports those outputs as `CHANGED`; that is expected (see the verification table).
* Legacy v1 notebooks 01-07 and ESM-2 / BioNeMo embedding (03, 09) need `torch` and
  `transformers` (see the Drive `requirements.txt`) and ideally a GPU. They were not tested locally.

## Branches (git)

| Branch | Contents | Status |
|---|---|---|
| `main` | Drive snapshot + local run tooling | approved state |
| `review/v2.7.1` | + `src/repro_v27.py`, `tests/`, `scripts/reproduce_v27.sh`, notebook 21 | submitted, awaiting review |
| `v2.8-diagram-pipeline` | + v2.8 modules (`assay_registry`, `clinical_baseline`, `molecular_features`, `fusion_ablation`, `pipeline_v28`) and tests | submitted, awaiting review |

Switch with `git switch <branch>`. Only `main` reflects the approved Drive state.

## Verification (sandbox, Linux, Python 3.12.3)

These checks ran in Claude's Linux sandbox on a fresh copy of this folder. They did not run
in Colab or on your computer, and `setup_local.bat` (Windows) was not executed.

| Check | Result |
|---|---|
| `setup_local.sh` | about 25 s; 18 local notebooks; `257/257` files intact |
| Notebooks 12-20 (`*.local.ipynb`) | all ran; 165 output files byte-identical to Drive; `real_patient_features` and v2.5 forecasts differ by at most 4e-15; two JSON files differ only in key names (same values) |
| Notebook 12 | regenerates `real_serial_filtered.csv` byte-identical |
| `run_local.py v27` | OK; 4 unit tests pass (2 min 28 s) |
| `review/v2.7.1`: `bash scripts/reproduce_v27.sh .venv/bin/python outputs/v27_repro` | pytest 27 passed; verifier 77 checks, 0 failed; unit tests OK (4 min 43 s) |
| `v2.8-diagram-pipeline`: `run_local.py tests` | 64 passed, 2 skipped (need `LP_V27_OUTDIR` / two runs); unit tests OK |
| `run_local.py v28` | stages 1-7a, 9, 10 done; 7b not run; 8 fusion only; 11 gated (10 s) |
| `run_local.py v28 --embeddings` | 7b done: 2,561 dims, 179 of 295 patients; M0 5, M1 14, M2 2,575 features |

## Files added for local use

`run_local.py`, `setup_local.sh`, `setup_local.bat`, `scripts/check_data.py`,
`scripts/make_local_notebooks.py`, `requirements-lock.txt` (core pins),
`requirements-notebooks-lock.txt` (core + Jupyter, all platforms), `DRIVE_MANIFEST.tsv`,
`.gitignore`, `.gitattributes`, this file.
