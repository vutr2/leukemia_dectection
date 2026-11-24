"""
beataml.py
Load the BeatAML (aml_ohsu_2022) cohort for the secondary-vs-de-novo AML task.

Why this task and not early detection: every BeatAML sample is collected at or
after AML diagnosis, so the cohort cannot support a pre-diagnosis question.
What it CAN support is a real, unsolved clinical question - was this AML
preceded by MDS/MPN? - with a label that comes from clinical history and is
therefore independent of the mutations used as features.

Clinical context: WHO/ICC/ELN frameworks recognize myelodysplasia-related
gene mutations as diagnostically/prognostically important in AML. Treatment
selection, including whether CPX-351 is appropriate, depends on the full disease
context and is NOT determined by this mutation-list classifier. The published
gene-level categories still motivate a representation question: can variant-level
features add information beyond gene identity without being mistaken for a
treatment recommendation?
"""

from __future__ import annotations
import os
from typing import Optional, Tuple

import pandas as pd

from .data_processing import load_maf, clean_mutations, aggregate_by_patient

# ELN 2022 / WHO 2022 secondary-type ("myelodysplasia-related") gene set.
SECONDARY_TYPE_GENES = [
    "SRSF2", "SF3B1", "U2AF1", "ZRSR2", "ASXL1", "EZH2", "BCOR", "STAG2",
]


def load_beataml(
    data_dir: str = "data/raw/aml_ohsu_2022",
    ambiguous: str = "drop",
    genes=None,
    min_vaf: float = 0.02,
    min_depth: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (mutations, patients) for patients sampled at Initial Diagnosis.

    label = 1 : IS_TRANSFORMED, i.e. documented antecedent MDS/MPN
    label = 0 : IS_DENOVO

    `ambiguous` controls the 86 patients who are neither (most likely
    therapy-related AML):
        "drop"     - exclude them (default; two clean, opposed groups)
        "positive" - treat as secondary (t-AML is secondary biologically, but
                     the mechanism differs, so this dilutes the signal)
    Run both. If the conclusion flips between them, that is a finding to report,
    not to hide.
    """
    sample_f = os.path.join(data_dir, "data_clinical_sample.txt")
    maf_f = os.path.join(data_dir, "data_mutations.txt")
    for f in (sample_f, maf_f):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing {f}. Download aml_ohsu_2022 from cBioPortal.")

    cs = pd.read_csv(sample_f, sep="\t", comment="#", low_memory=False)

    sel = cs[(cs.DISEASE_STAGE_AT_SPECIMEN_COLLECTION == "Initial Diagnosis")
             & cs.IS_TRANSFORMED.isin(["TRUE", "FALSE"])
             & cs.IS_DENOVO.isin(["TRUE", "FALSE"])].drop_duplicates("PATIENT_ID").copy()

    if ambiguous == "drop":
        sel = sel[(sel.IS_TRANSFORMED == "TRUE") | (sel.IS_DENOVO == "TRUE")]
    elif ambiguous != "positive":
        raise ValueError("ambiguous must be 'drop' or 'positive'")

    sel["label"] = (sel.IS_DENOVO != "TRUE").astype(int)
    print(f"[beataml] {len(sel)} patients at Initial Diagnosis | "
          f"{int(sel.label.sum())} secondary ({sel.label.mean():.1%}) "
          f"| ambiguous={ambiguous}")

    maf = load_maf(maf_f, cohort="BEATAML")
    maf = maf[maf.Tumor_Sample_Barcode.isin(sel.SAMPLE_ID)].copy()

    # Use the real PATIENT_ID. The barcode-splitting heuristic in
    # standardize_columns is TCGA-specific and would silently merge patients.
    s2p = dict(zip(sel.SAMPLE_ID, sel.PATIENT_ID))
    s2l = dict(zip(sel.SAMPLE_ID, sel.label))
    maf["patient_id"] = maf.Tumor_Sample_Barcode.map(s2p)

    mut = clean_mutations(maf, genes=genes, min_vaf=min_vaf, min_depth=min_depth)
    mut["label"] = mut.sample_id.map(s2l)

    patients = aggregate_by_patient(mut)
    patients["label"] = patients.patient_id.map(
        mut.groupby("patient_id")["label"].first()).astype(int)

    dropped = len(sel) - len(patients)
    if dropped:
        print(f"[beataml] {dropped} patients had no mutation passing filters "
              f"and are excluded. Note this in your limitations: it is a "
              f"selection step, not a neutral one.")
    return mut, patients


def eln2022_rule(patients: pd.DataFrame,
                 genes=SECONDARY_TYPE_GENES) -> pd.Series:
    """
    The current clinical rule: any secondary-type gene mutated -> AML-MR.
    This is the benchmark to beat, not a strawman - it is in the guidelines.
    """
    g = set(genes)
    return patients["genes"].apply(lambda gs: int(any(x in g for x in gs)))
