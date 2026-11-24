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
```

Known user execution state:

- Notebook 12: completed
- Notebook 13: completed
- Notebook 14: available; execution status may need confirmation
- Notebook 15: available; execution status may need confirmation
- Notebook 16: available; execution status may need confirmation

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

### Molecular
BioNeMo/ESM-2 and mutation-type representations.

Evaluate separately so the project can determine whether molecular representation truly adds incremental value.

---

## 15. Next model-development priorities

Priority order:

1. Validate temporal forecasting and uncertainty.
2. Improve growth estimation with conservative hierarchical / empirical-Bayes shrinkage.
3. Add predictive intervals for future VAF, not only point predictions.
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
