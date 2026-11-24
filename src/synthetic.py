"""
synthetic.py
Generate a fake cBioPortal-style MAF + clinical table + matching sequences.

Purpose: let every notebook run end-to-end today, so you are debugging the
pipeline and not the download. Delete the calls to this module the moment real
data lands. Numbers produced here are NOT results and must never appear in an
application.
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .mutation_utils import CH_GENES, HIGH_RISK_GENES

AA = list("ACDEFGHIKLMNPQRSTVWY")

# Real canonical lengths, so windowing in embedding.py gets exercised properly
# (TET2 at 2002 aa is over ESM2's 1022-token limit — that path needs testing).
PROTEIN_LEN = {
    "DNMT3A": 912, "TET2": 2002, "ASXL1": 1541, "TP53": 393, "JAK2": 1132,
    "SF3B1": 1304, "SRSF2": 221, "U2AF1": 240, "ZRSR2": 482, "IDH1": 414,
    "IDH2": 452, "RUNX1": 453, "PPM1D": 605, "CBL": 906, "GNB1": 340,
    "GNAS": 1037, "KRAS": 189, "NRAS": 189, "FLT3": 993, "NPM1": 294,
    "CEBPA": 358, "KIT": 976, "WT1": 449, "EZH2": 746, "STAG2": 1231,
    "BCOR": 1755, "BCORL1": 1785, "PHF6": 365, "ETV6": 452, "SETBP1": 1596,
    "CSF3R": 836, "CALR": 417, "MPL": 635, "RAD21": 631, "SMC3": 1217,
    "NF1": 2839, "PTPN11": 593, "BRAF": 766, "CUX1": 1505, "SETD2": 2564,
    "KDM6A": 1401, "CREBBP": 2442, "EP300": 2414, "PRPF8": 2335,
    "LUC7L2": 392, "DDX41": 622,
}


def make_synthetic_sequences(seed: int = 0) -> Dict[str, str]:
    """Random but length-correct 'protein' per gene, used as the wild type."""
    rng = np.random.default_rng(seed)
    return {g: "".join(rng.choice(AA, size=PROTEIN_LEN.get(g, 500)))
            for g in CH_GENES}


def make_synthetic_cohort(
    n_patients: int = 600,
    positive_rate: float = 0.18,
    seed: int = 42,
    difficulty: float = 0.65,
    cohort_confound: float = 0.0,
    return_sequences: bool = False,
) -> Tuple:
    """
    Returns (maf_df, clinical_df) or (maf_df, clinical_df, sequences).

    `cohort_confound` in [0, 1] simulates the classic study-design failure:
    cases downloaded from an AML study, controls from a separate CH study, each
    sequenced on its own panel. At 1.0 the two cohorts use disjoint gene panels
    and cohort predicts label perfectly. Notebook 01's leakage check should fire
    loudly. Use it to confirm your detector actually works before trusting a
    clean report on real data.

    `difficulty` in [0, 1] controls how much cases and controls overlap.
      0.0  -> cases obviously separate; any model hits AUC ~1.0
      0.65 -> default; a decent baseline lands around AUC 0.75-0.85
      1.0  -> no signal at all, AUC should be ~0.5. Use this as a negative
              control: if the pipeline still reports 0.9 here, you have a leak.

    Wild-type residues are read from `sequences`, so mutations are internally
    consistent and notebook 03 sees a 0% reference-mismatch rate. Against real
    UniProt sequences these fake mutations would mismatch ~95% of the time,
    which makes that diagnostic useless exactly when you want to trust it.
    """
    rng = np.random.default_rng(seed)
    sequences = make_synthetic_sequences(seed)
    d = float(np.clip(difficulty, 0.0, 1.0))
    c = float(np.clip(cohort_confound, 0.0, 1.0))

    # Split the panel in two so a confounded design has cohort-specific genes.
    half = len(CH_GENES) // 2
    panel_case, panel_ctrl = CH_GENES[:half], CH_GENES[half:]

    # Separation between cases and controls shrinks as difficulty rises.
    lam_pos, lam_neg = 2.4 - 1.3 * d, 1.1
    beta_pos, beta_neg = 6 + 8 * d, 14
    p_highrisk = 0.60 * (1 - d)          # 0 at difficulty=1 -> a true null control

    rows, clinical = [], []
    for i in range(n_patients):
        pid = f"PT-{i:04d}"
        pos = rng.random() < positive_rate

        n_mut = rng.poisson(lam_pos if pos else lam_neg) + 1

        if c > 0 and rng.random() < c:
            cohort = "AML_STUDY" if pos else "CH_STUDY"
            pool = panel_case if pos else panel_ctrl
        else:
            cohort = rng.choice(["AML_STUDY", "CH_STUDY"]) if c > 0 else "SYNTHETIC"
            pool = HIGH_RISK_GENES if (pos and rng.random() < p_highrisk) else CH_GENES

        genes = rng.choice(pool, size=min(n_mut, len(pool)), replace=False)

        for g in genes:
            seq = sequences[g]
            p = int(rng.integers(2, len(seq)))
            wt = seq[p - 1]                       # consistent with the sequence
            mut = rng.choice([a for a in AA if a != wt])
            vaf = float(np.clip(rng.beta(3, beta_pos if pos else beta_neg), 0.01, 0.95))
            depth = int(rng.integers(200, 1200))
            alt = int(round(vaf * depth))
            rows.append({
                "Hugo_Symbol": g,
                "HGVSp_Short": f"p.{wt}{p}{mut}",
                "Variant_Classification": "Missense_Mutation",
                "Tumor_Sample_Barcode": f"{pid}-01",
                "t_alt_count": alt,
                "t_ref_count": depth - alt,
                "cohort": cohort,
            })

        clinical.append({
            "patient_id": pid,
            "age": int(rng.normal(68, 9)),
            "sex": rng.choice(["M", "F"]),
            "years_to_followup": round(float(rng.uniform(2, 10)), 1),
            "progressed_to_aml": int(pos),
            "diagnosis": "AML" if pos else "CH",
        })

    maf, clin = pd.DataFrame(rows), pd.DataFrame(clinical)
    print(f"[synthetic] {len(maf):,} mutations, {len(clin):,} patients, "
          f"{clin['progressed_to_aml'].mean():.1%} positive, "
          f"difficulty={d}, cohort_confound={c}")
    return (maf, clin, sequences) if return_sequences else (maf, clin)
