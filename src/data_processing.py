"""
data_processing.py
Load cBioPortal / MAF-style mutation tables -> tidy per-mutation table -> labels.
"""

from __future__ import annotations
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .mutation_utils import (
    parse_protein_change, mutation_key, CH_GENES, HIGH_RISK_GENES,
)

# Canonical name -> candidate source columns, in priority order.
# A single MAF can carry several of these at once (BeatAML has 68 columns), so
# we pick the FIRST one present rather than renaming them all, which would
# produce duplicate column names and make df["gene"] return a DataFrame.
CANONICAL_SOURCES = {
    "patient_id":    ["patient_id", "PATIENT_ID"],
    "sample_id":     ["Tumor_Sample_Barcode", "sample_id", "Sample_ID", "SAMPLE_ID"],
    "cohort":        ["cohort", "COHORT"],
    "gene":          ["Hugo_Symbol", "gene", "Gene", "GENE"],
    "protein_change": ["HGVSp_Short", "Protein_Change", "protein_change",
                       "amino_acid_change", "HGVSp"],
    "variant_class": ["Variant_Classification", "variant_class"],
    "alt_count":     ["t_alt_count", "alt_count", "TumorAltCount"],
    "ref_count":     ["t_ref_count", "ref_count", "TumorRefCount"],
    "vaf":           ["VAF", "vaf", "i_TumorVAF_WU", "tumor_vaf"],
}


def load_maf(
    path: str,
    columns: Optional[dict] = None,
    cohort: Optional[str] = None,
    comment: str = "#",
    sep: str = "\t",
) -> pd.DataFrame:
    """
    Read a data_mutations_extended.txt style file.

    columns: optional {source_name: canonical_name} overrides for datasets whose
             headers are not in COLUMN_ALIASES.
    cohort:  tag every row with its source. Do not skip this when merging TCGA +
             BeatAML + MSK — different panels and sequencing depths produce a
             batch effect that a classifier will happily learn instead of biology.
    """
    df = pd.read_csv(path, sep=sep, comment=comment, low_memory=False)
    if columns:
        df = df.rename(columns=columns)
    if cohort is not None:
        df["cohort"] = cohort
    print(f"[load_maf] {path}: {df.shape[0]:,} rows x {df.shape[1]} cols"
          + (f"  cohort={cohort}" if cohort else ""))
    return df


def standardize_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Pick one source column per canonical name. Never produces duplicates."""
    cols = list(df.columns)
    out, chosen = {}, {}
    for canon, candidates in CANONICAL_SOURCES.items():
        for c in candidates:
            if c in cols:
                col = df[c]
                if isinstance(col, pd.DataFrame):   # duplicated header in the file
                    col = col.iloc[:, 0]
                out[canon] = col.reset_index(drop=True)
                chosen[canon] = c
                break

    if "sample_id" not in out:
        raise KeyError(
            "No sample identifier found. Expected one of "
            f"{CANONICAL_SOURCES['sample_id']}. Got: {cols[:25]}"
        )
    if "gene" not in out:
        raise KeyError(f"No gene column found. Got: {cols[:25]}")

    res = pd.DataFrame(out)
    if "patient_id" not in res.columns:
        # cBioPortal barcodes: TCGA-AB-2802-03 -> patient TCGA-AB-2802.
        # This heuristic is TCGA-specific and WRONG for most other studies.
        res["patient_id"] = res["sample_id"].astype(str).str.rsplit("-", n=1).str[0]
        n_s, n_p = res["sample_id"].nunique(), res["patient_id"].nunique()
        if n_p < 0.5 * n_s:
            print(f"\n  !! Derived only {n_p} patients from {n_s} samples by "
                  f"splitting the barcode.\n     That is almost certainly wrong. "
                  f"Merge the real PATIENT_ID from data_clinical_sample.txt into\n"
                  f"     the MAF before calling clean_mutations, or the same "
                  f"person lands in both\n     train and test folds.\n")

    if verbose:
        print("[standardize] " + ", ".join(f"{v}->{k}" for k, v in chosen.items()))
        unused = [c for c in cols if c not in chosen.values()]
        if unused:
            print(f"[standardize] ignored {len(unused)} other columns")
    return res


def compute_vaf(df: pd.DataFrame) -> pd.DataFrame:
    """VAF = alt / (alt + ref) when not already provided."""
    out = df.copy()
    if "vaf" not in out.columns or out["vaf"].isna().all():
        if {"alt_count", "ref_count"}.issubset(out.columns):
            depth = out["alt_count"].fillna(0) + out["ref_count"].fillna(0)
            out["vaf"] = np.where(depth > 0, out["alt_count"] / depth, np.nan)
        else:
            warnings.warn("No VAF and no allele counts -> vaf set to NaN.")
            out["vaf"] = np.nan
    # Some sources store VAF as a percentage.
    if out["vaf"].max(skipna=True) is not np.nan and out["vaf"].max(skipna=True) > 1.5:
        out["vaf"] = out["vaf"] / 100.0
    out["vaf"] = out["vaf"].clip(0, 1)
    return out


def clean_mutations(
    df: pd.DataFrame,
    genes: Optional[Iterable[str]] = CH_GENES,
    min_vaf: float = 0.02,
    min_depth: int = 0,
    keep_kinds: Iterable[str] = ("missense", "nonsense", "frameshift",
                                "splice", "indel"),
) -> pd.DataFrame:
    """Standardise, filter to a gene panel, parse protein changes, drop noise."""
    out = standardize_columns(df)
    out = compute_vaf(out)

    out["gene"] = out["gene"].astype(str).str.upper().str.strip()
    if genes is not None:
        before = len(out)
        out = out[out["gene"].isin(set(genes))]
        print(f"[clean] gene panel filter: {before:,} -> {len(out):,} rows")

    parsed = out["protein_change"].apply(parse_protein_change)
    out["aa_pos"] = parsed.apply(lambda p: p["pos"] if p else np.nan)
    out["aa_wt"] = parsed.apply(lambda p: p["wt"] if p else None)
    out["aa_mut"] = parsed.apply(lambda p: p["mut"] if p else None)
    out["kind"] = parsed.apply(lambda p: p["kind"] if p else "unparsed")

    before = len(out)
    out = out[out["kind"].isin(set(keep_kinds))]
    out = out[out["vaf"].fillna(0) >= min_vaf]
    print(f"[clean] kind + VAF>={min_vaf} filter: {before:,} -> {len(out):,} rows")

    if min_depth > 0 and {"alt_count", "ref_count"}.issubset(out.columns):
        depth = out["alt_count"].fillna(0) + out["ref_count"].fillna(0)
        before = len(out)
        out = out[depth >= min_depth]
        print(f"[clean] depth>={min_depth} filter: {before:,} -> {len(out):,} rows")

    out["mut_key"] = [mutation_key(g, c) for g, c in
                      zip(out["gene"], out["protein_change"])]
    return out.reset_index(drop=True)


def aggregate_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: mutation list + simple clonal-burden summaries."""
    g = df.groupby("patient_id")
    agg = pd.DataFrame({
        "n_mutations": g.size(),
        "max_vaf": g["vaf"].max(),
        "mean_vaf": g["vaf"].mean(),
        "sum_vaf": g["vaf"].sum(),
        "n_genes": g["gene"].nunique(),
        "genes": g["gene"].apply(list),
        "mut_keys": g["mut_key"].apply(list),
        "vafs": g["vaf"].apply(list),
    }).reset_index()
    if "cohort" in df.columns:
        agg = agg.merge(df.groupby("patient_id")["cohort"].first().reset_index(),
                        on="patient_id", how="left")
    print(f"[aggregate] {len(agg):,} patients")
    return agg


# --- Labelling ---------------------------------------------------------------

_LEAK_WARNING = """
!! LABEL LEAKAGE WARNING !!
Strategy '{s}' derives the label from the same mutations you are feeding in as
features. The model will score very well and mean nothing: it is re-learning
your own labelling rule. Use this only as a pipeline smoke test.
The only non-circular label is 'longitudinal': the same person sampled at t0
(still healthy) with a recorded outcome at t1 (progressed / did not).
"""


def assign_labels(
    patients: pd.DataFrame,
    strategy: str = "longitudinal",
    clinical: Optional[pd.DataFrame] = None,
    diagnosis_col: str = "diagnosis",
    positive_diagnoses: Iterable[str] = ("AML", "MDS", "MPN", "CMML"),
    outcome_col: str = "progressed_to_aml",
    high_risk_genes: Iterable[str] = HIGH_RISK_GENES,
    high_risk_vaf: float = 0.10,
) -> pd.DataFrame:
    """
    Attach a binary `label` column.

    strategy:
      'longitudinal' - join an outcome column from `clinical` (RECOMMENDED)
      'diagnosis'    - AML/MDS sample = 1, CH-only sample = 0 (leaky, see warning)
      'chip_risk'    - high-risk genotype rule = 1 (circular, smoke test only)
    """
    out = patients.copy()

    if strategy == "longitudinal":
        if clinical is None or outcome_col not in clinical.columns:
            raise ValueError(
                f"strategy='longitudinal' needs a clinical table with '{outcome_col}'."
            )
        out = out.merge(clinical[["patient_id", outcome_col]], on="patient_id", how="inner")
        out["label"] = out[outcome_col].astype(int)

    elif strategy == "diagnosis":
        print(_LEAK_WARNING.format(s=strategy))
        if clinical is None or diagnosis_col not in clinical.columns:
            raise ValueError(f"strategy='diagnosis' needs clinical['{diagnosis_col}'].")
        out = out.merge(clinical[["patient_id", diagnosis_col]], on="patient_id", how="inner")
        pos = {d.upper() for d in positive_diagnoses}
        out["label"] = out[diagnosis_col].astype(str).str.upper().isin(pos).astype(int)

    elif strategy == "chip_risk":
        print(_LEAK_WARNING.format(s=strategy))
        hr = set(high_risk_genes)
        out["label"] = (
            out["genes"].apply(lambda gs: any(g in hr for g in gs))
            & (out["max_vaf"] >= high_risk_vaf)
        ).astype(int)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    n1 = int(out["label"].sum())
    print(f"[label:{strategy}] n={len(out):,}  positives={n1:,} ({n1/max(len(out),1):.1%})")
    return out
