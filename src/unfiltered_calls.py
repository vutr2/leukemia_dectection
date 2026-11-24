"""Build a trajectory-unfiltered serial table for LeukemiPrediag v2.5.

This recovers sub-1% and zero-VAF measurements that are present in the supplied
raw cohort tables. It does not invent rows for patient/variant/time combinations
that are absent from the source files.

Important: the LBC source files themselves are labelled 1PCT_VAF, so this is not
an ascertainment-free callset. It recovers below-floor measurements for variants
represented in those files, but variants that never entered the 1% source universe
may still be absent.

Research use only. Not a diagnostic device.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE_FLOOR = 0.01
SARDINIA_OPERATIONAL_LIMIT = 0.002


def _lbc(path: str | Path, cohort: str, years_per_wave: float = 3.0) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", low_memory=False)
    req = ["participant_id", "chromosome", "position", "reference", "mutation",
           "wave", "AF", "DP", "AO", "PreferredSymbol", "protein_substitution"]
    missing = [c for c in req if c not in d.columns]
    if missing:
        raise KeyError(f"{cohort} missing required columns: {missing}")

    x = pd.DataFrame({
        "patient_id": d["participant_id"].astype(str),
        "cohort": cohort,
        "variant_id": (
            d["chromosome"].astype(str) + ":" + d["position"].astype(str) + ":" +
            d["reference"].astype(str) + ">" + d["mutation"].astype(str)
        ),
        "wave": pd.to_numeric(d["wave"], errors="coerce"),
        "vaf": pd.to_numeric(d["AF"], errors="coerce"),
        "depth": pd.to_numeric(d["DP"], errors="coerce"),
        "alt_reads": pd.to_numeric(d["AO"], errors="coerce"),
        "gene": d["PreferredSymbol"].astype(str).str.upper().str.strip(),
        "protein_change": d["protein_substitution"],
        "detection_limit": pd.to_numeric(d["X95MDAF"], errors="coerce") if "X95MDAF" in d.columns else np.nan,
    })
    x["age"] = np.nan
    x["time_years"] = x["wave"] * float(years_per_wave)
    x["time_years"] -= x.groupby("patient_id")["time_years"].transform("min")
    x["detection_limit_source"] = "X95MDAF in supplied LBC table"
    x["assay_measurement"] = "AO/DP with AF; source file variant universe labelled 1PCT_VAF"
    return x


def _sardinia(path: str | Path) -> pd.DataFrame:
    d = pd.read_excel(path, header=3)
    d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
    d = d.dropna(subset=["Sample ID", "VAF", "Age", "Gene", "Chromosome", "Start", "WT", "MT"])
    x = pd.DataFrame({
        "patient_id": d["Sample ID"].astype(str),
        "cohort": "SardiNIA",
        "variant_id": (
            d["Chromosome"].astype(str) + ":" + d["Start"].astype(str) + ":" +
            d["WT"].astype(str) + ">" + d["MT"].astype(str)
        ),
        "wave": pd.to_numeric(d["Study Phase"], errors="coerce"),
        "age": pd.to_numeric(d["Age"], errors="coerce"),
        "vaf": pd.to_numeric(d["VAF"], errors="coerce"),
        "depth": np.nan,
        "alt_reads": np.nan,
        "gene": d["Gene"].astype(str).str.upper().str.strip(),
        "protein_change": d["Protein"],
        "detection_limit": float(SARDINIA_OPERATIONAL_LIMIT),
    })
    x["time_years"] = x["age"] - x.groupby("patient_id")["age"].transform("min")
    x["detection_limit_source"] = (
        "Fabre et al. Nature 2022 operational detection definition VAF>=0.2%; "
        "not a per-call field in Supplementary Table 2"
    )
    x["assay_measurement"] = "reported VAF; site re-queried across serial samples; no per-call counts in table"
    return x


def finalize(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x = x[np.isfinite(pd.to_numeric(x["vaf"], errors="coerce")) &
          np.isfinite(pd.to_numeric(x["time_years"], errors="coerce"))].copy()
    key = ["patient_id", "cohort", "variant_id", "time_years"]
    if x.duplicated(key).any():
        raise ValueError("Source produced duplicate patient x variant x timepoint rows; inspect before collapsing")

    v = pd.to_numeric(x["vaf"], errors="coerce")
    lim = pd.to_numeric(x["detection_limit"], errors="coerce")
    alt = pd.to_numeric(x["alt_reads"], errors="coerce")
    x["observed_alt_signal"] = np.where(alt.notna(), (alt > 0).astype(int), (v > 0).astype(int))
    x["below_pipeline_1pct"] = (v < PIPELINE_FLOOR).astype(int)
    x["below_detection_limit"] = np.where(lim.notna(), (v < lim).astype(int), np.nan)
    x["detected"] = np.where(lim.notna(), (v >= lim).astype(int), x["observed_alt_signal"])
    x["censored"] = np.where(lim.notna(), (v < lim).astype(int), (v <= 0).astype(int))
    x["pipeline_filter_floor"] = PIPELINE_FLOOR

    cols = [
        "patient_id", "cohort", "variant_id", "time_years", "wave", "age",
        "vaf", "alt_reads", "depth", "detected", "censored", "observed_alt_signal",
        "detection_limit", "detection_limit_source", "below_detection_limit",
        "below_pipeline_1pct", "pipeline_filter_floor", "gene", "protein_change",
        "assay_measurement",
    ]
    return x[cols].sort_values(["cohort", "patient_id", "variant_id", "time_years"]).reset_index(drop=True)


def build(lbc178936: str | Path, lbc288742: str | Path, sardinia: str | Path) -> pd.DataFrame:
    x = pd.concat([
        _lbc(lbc178936, "GSE178936"),
        _lbc(lbc288742, "GSE288742"),
        _sardinia(sardinia),
    ], ignore_index=True, sort=False)
    return finalize(x)
