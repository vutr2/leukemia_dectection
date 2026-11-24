# Blood Cancer Early Detection (CHIP → pre-leukaemia)

Binary classifier over peripheral-blood somatic mutations (gene + protein change + VAF)
predicting signs of malignant transformation, with ESM2 protein-language-model
embeddings of each mutation.

## Quickstart

```bash
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order. They work **today** on a synthetic cohort, so you can
debug the pipeline before any download finishes. Drop real files into `data/raw/`
and the synthetic fallback switches itself off.

| notebook | does |
|---|---|
| `01_data_exploration.ipynb` | load MAF + clinical, cohort sanity checks |
| `02_prepare_mutations.ipynb` | per-patient aggregation, labelling, UniProt sequences |
| `03_get_embeddings.ipynb` | ESM2 wild-type vs mutant delta embeddings |
| `04_train_classifier.ipynb` | ablation, cross-validation, screening metrics |
| `05_inference_demo.ipynb` | mutations in → probability out |

On Colab: set `ROOT` in the bootstrap cell to your clone or Drive path. Notebook 03
wants a GPU; the rest run fine on CPU.

## Two design decisions you should understand

**1. Mutation delta, not whole-protein embedding.** DNMT3A is 912 residues. A
mean-pooled embedding of the mutant is nearly identical to the wild type, so the
signal disappears. We take the residue-level representation at the mutated site
and subtract:

```
delta = ESM2(mutant)[i] − ESM2(wildtype)[i]        # 1280-d
```

Long proteins (TET2, 2002 aa) exceed ESM2's 1022-token context, so sequences are
windowed symmetrically around the mutation first.

**2. Labels are the hard part, not the model.** `assign_labels` offers three
strategies and only one is defensible:

- `longitudinal` — same person sampled while healthy, outcome recorded at follow-up. **This is the actual task.**
- `diagnosis` — AML/MDS sample = 1. Leaky: the label is a function of the very mutations you feed in as features. You will get AUC 0.95 and it will mean nothing.
- `chip_risk` — high-risk gene + VAF rule = 1. Circular: the model re-learns your rule.

The leaky strategies exist so the pipeline runs before the right data arrives.
Never report a number produced by them.

Public longitudinal CHIP→AML cohorts are scarce and mostly sit behind dbGaP —
Abelson et al. (Nature, 2018), Desai et al. (Nat Med, 2018), Bolton et al.
(Nat Genet, 2020) are the reference cohorts. Getting access to one of these is
the real critical path for this project; the code is not.

## Metrics

Accuracy is not reported anywhere on purpose. At a realistic ~1% positive rate,
"always predict 0" scores 99%. The pipeline reports **sensitivity at 98%
specificity**, AUPRC, and AUC, and the ablation compares gene identity + VAF
alone against gene + VAF + ESM2. If ESM2 does not clear the fold-to-fold noise
band, that is a finding worth knowing before writing any application.

## Layout

```
src/
  mutation_utils.py   HGVSp parsing, CH gene panels
  data_processing.py  MAF loading, VAF, filtering, aggregation, labelling
  sequences.py        UniProt fetch + disk cache
  embedding.py        ESM2 wrapper, windowing, delta features
  model.py            feature assembly, baselines, CV, ablation
  inference.py        single-patient predictor
  synthetic.py        fake cohort so nothing blocks on downloads
```

## Not a diagnostic

Research code. Outputs are risk scores on retrospective research cohorts and
carry no clinical validity.
