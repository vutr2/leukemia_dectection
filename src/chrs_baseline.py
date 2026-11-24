"""CHRS-style static reference baseline for LeukemiPrediag v2.

Research/reproduction utility only. This module encodes the published variable
weights from Weeks et al., NEJM Evidence (2023) so future longitudinal models
can be compared against a credible static baseline rather than a strawman.

It is NOT a medical-device implementation and must not be used for patient care.
Exact cohort variable definitions and high-risk mutation mapping must be checked
against the source paper / supplement before any formal validation report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CHRS_HIGH_RISK_GENES = (
    "SRSF2", "SF3B1", "ZRSR2", "JAK2", "TP53",
    "IDH1", "IDH2", "FLT3", "RUNX1",
)


def _parse_genes(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return [str(v).upper() for v in x]
    if isinstance(x, str):
        z = x.strip()
        if not z:
            return []
        if z.startswith("[") and z.endswith("]"):
            try:
                import ast
                v = ast.literal_eval(z)
                if isinstance(v, (list, tuple, set)):
                    return [str(q).upper() for q in v]
            except Exception:
                pass
        sep = ";" if ";" in z else ","
        return [q.strip().upper() for q in z.split(sep) if q.strip()]
    return [str(x).upper()]


def annotate_chrs_genetics(
    df: pd.DataFrame,
    *,
    genes_col: str = "genes",
    mutation_count_col: str = "n_mutations",
) -> pd.DataFrame:
    """Add transparent CHRS genetic indicators from patient-level gene lists.

    `single_dnmt3a` means exactly one CH mutation and that mutation is DNMT3A.
    `high_risk_mutation` follows the published CHRS gene category.
    """
    if genes_col not in df.columns or mutation_count_col not in df.columns:
        raise KeyError(f"Need {genes_col!r} and {mutation_count_col!r}")
    out = df.copy()
    parsed = out[genes_col].map(_parse_genes)
    nmut = pd.to_numeric(out[mutation_count_col], errors="coerce")
    out["single_dnmt3a"] = [
        int((n == 1) and (set(gs) == {"DNMT3A"})) if np.isfinite(n) else np.nan
        for n, gs in zip(nmut, parsed)
    ]
    risk = set(CHRS_HIGH_RISK_GENES)
    out["high_risk_mutation"] = [int(bool(set(gs) & risk)) for gs in parsed]
    return out


def chrs_score(
    df: pd.DataFrame,
    *,
    single_dnmt3a_col: str = "single_dnmt3a",
    high_risk_mutation_col: str = "high_risk_mutation",
    mutation_count_col: str = "mutation_count",
    max_vaf_col: str = "max_vaf",
    rdw_col: str = "rdw",
    mcv_col: str = "mcv",
    ccus_col: str = "ccus",
    age_col: str = "age",
) -> pd.DataFrame:
    """Return a copy with a transparent CHRS-style score and risk group.

    Required inputs are patient-level variables available at the index time.
    Missing values are *not* silently imputed: rows with missing required inputs
    receive NaN score so leakage-prone or undocumented imputation cannot sneak
    into evaluation.
    """
    out = df.copy()
    cols = [single_dnmt3a_col, high_risk_mutation_col, mutation_count_col,
            max_vaf_col, rdw_col, mcv_col, ccus_col, age_col]
    missing = [c for c in cols if c not in out.columns]
    if missing:
        raise KeyError(f"Missing CHRS baseline fields: {missing}")

    complete = out[cols].notna().all(axis=1)
    score = pd.Series(np.nan, index=out.index, dtype=float)

    d = out.loc[complete]
    s = (
        np.where(d[single_dnmt3a_col].astype(bool), 0.5, 1.0)
        + np.where(d[high_risk_mutation_col].astype(bool), 2.5, 1.0)
        + np.where(d[mutation_count_col] >= 2, 2.0, 1.0)
        + np.where(d[max_vaf_col] >= 0.20, 2.0, 1.0)
        + np.where(d[rdw_col] >= 15.0, 2.5, 1.0)
        + np.where(d[mcv_col] >= 100.0, 2.5, 1.0)
        + np.where(d[ccus_col].astype(bool), 1.5, 1.0)
        + np.where(d[age_col] >= 65.0, 1.5, 1.0)
    )
    score.loc[complete] = s.astype(float)
    out["chrs_score"] = score

    group = pd.Series(pd.NA, index=out.index, dtype="string")
    group.loc[score <= 9.5] = "low"
    group.loc[(score >= 10.0) & (score <= 12.0)] = "intermediate"
    group.loc[score >= 12.5] = "high"
    out["chrs_group"] = group
    return out
