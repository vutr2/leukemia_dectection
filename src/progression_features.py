"""LeukemiPrediag v2.2 progression-feature engineering.

This module enriches the Stage-B longitudinal clone outputs without creating or
inventing leukemia outcome labels.  It converts serial VAF measurements,
uncertainty-aware trajectory posteriors, pairwise co-movement support and
candidate-clone summaries into a patient-level feature matrix that can later be
used by the M1/M2 fusion models on a genuinely outcome-linked cohort.

Key feature families
--------------------
1. Clone size / VAF burden: continuous VAF summaries plus observed threshold
   indicators. Threshold features are covariates, not diagnostic rules.
2. Kinetics: log-VAF growth, raw dVAF/dt, posterior growth uncertainty and
   positive-growth doubling time.
3. Experimental acceleration: quadratic log-VAF curvature only when >=4 unique
   timepoints are available. This is deliberately marked experimental because
   sparse trajectories make second derivatives noisy.
4. Clone architecture: candidate-clone count, size, diversity, dominant-clone
   fraction and growth dispersion.
5. Co-mutation architecture: observed gene-pair burden and the subset of pairs
   with longitudinal same-clone support.
6. Configurable gene-presence flags. These indicate presence only and are not
   hard-coded risk labels.

Research use only. Not a diagnostic device.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_GENE_FLAGS: Tuple[str, ...] = (
    "DNMT3A", "TET2", "ASXL1", "TP53", "FLT3", "SRSF2", "SF3B1",
    "U2AF1", "JAK2", "IDH1", "IDH2", "RUNX1", "STAG2", "BCOR",
    "EZH2", "ZRSR2", "PPM1D", "NPM1",
)


@dataclass(frozen=True)
class ProgressionFeatureConfig:
    patient_col: str = "patient_id"
    cohort_col: str = "cohort"
    variant_col: str = "variant_id"
    time_col: str = "time_years"
    vaf_col: str = "vaf"
    gene_col: str = "gene"
    protein_col: str = "protein_change"
    log_eps: float = 1e-5
    min_acceleration_points: int = 4
    pair_support_threshold: float = 0.80
    fast_growth_threshold: float = 0.05
    vaf_thresholds: Tuple[float, ...] = (0.10, 0.20)
    gene_flags: Tuple[str, ...] = DEFAULT_GENE_FLAGS


def _threshold_tag(thr: float) -> str:
    """Stable human-readable tag, e.g. 0.20 -> 20pct."""
    return f"{int(round(float(thr) * 100)):02d}pct"


def _safe_entropy(values: np.ndarray) -> float:
    x = np.asarray(values, float)
    x = x[np.isfinite(x) & (x > 0)]
    if not len(x) or x.sum() <= 0:
        return np.nan
    p = x / x.sum()
    return float(-(p * np.log(p)).sum())


def _positive_doubling_time(growth: float) -> float:
    return float(np.log(2.0) / growth) if np.isfinite(growth) and growth > 0 else np.nan


def _variant_kinetics_one(g: pd.DataFrame, cfg: ProgressionFeatureConfig) -> dict:
    z = (
        g[[cfg.time_col, cfg.vaf_col]]
        .dropna()
        .groupby(cfg.time_col, as_index=False)[cfg.vaf_col]
        .mean()
        .sort_values(cfg.time_col)
    )
    if len(z) < 2:
        return {}

    t = z[cfg.time_col].to_numpy(float)
    v = np.clip(z[cfg.vaf_col].to_numpy(float), cfg.log_eps, None)
    span = float(np.ptp(t))
    if span <= 0:
        return {}

    y = np.log(v)
    t0 = t - t.mean()
    slope = float(np.polyfit(t0, y, 1)[0])
    raw_slope = float((v[-1] - v[0]) / span)

    acceleration = np.nan
    end_log_growth = np.nan
    if len(z) >= cfg.min_acceleration_points:
        # y = a2*t^2 + a1*t + a0 on centered time.
        a2, a1, _ = np.polyfit(t0, y, 2)
        acceleration = float(2.0 * a2)  # d^2 log(VAF) / dt^2
        end_log_growth = float(a1 + 2.0 * a2 * t0[-1])

    out = {
        "n_timepoints_observed": int(len(z)),
        "time_span_years_observed": span,
        "first_vaf_observed": float(v[0]),
        "last_vaf_observed": float(v[-1]),
        "max_vaf_observed": float(v.max()),
        "delta_vaf_per_year": raw_slope,
        "observed_log_vaf_growth": slope,
        "observed_doubling_time_years": _positive_doubling_time(slope),
        "log_vaf_acceleration": acceleration,
        "end_log_vaf_growth": end_log_growth,
    }

    for thr in cfg.vaf_thresholds:
        tag = _threshold_tag(thr)
        out[f"ever_vaf_ge_{tag}"] = int(np.any(v >= thr))
        out[f"crossed_vaf_ge_{tag}"] = int((v[0] < thr) and np.any(v[1:] >= thr))
    return out


def variant_kinetics(
    serial: pd.DataFrame,
    cfg: ProgressionFeatureConfig = ProgressionFeatureConfig(),
) -> pd.DataFrame:
    """Observed first/second-order kinetics for each patient-variant trajectory."""
    required = [cfg.patient_col, cfg.cohort_col, cfg.variant_col, cfg.time_col, cfg.vaf_col]
    missing = [c for c in required if c not in serial.columns]
    if missing:
        raise KeyError(f"serial missing required columns: {missing}")

    rows = []
    for (pid, cohort, vid), g in serial.groupby(
        [cfg.patient_col, cfg.cohort_col, cfg.variant_col], sort=False
    ):
        d = _variant_kinetics_one(g, cfg)
        if not d:
            continue
        row = {
            cfg.patient_col: pid,
            cfg.cohort_col: cohort,
            cfg.variant_col: vid,
        }
        for c in [cfg.gene_col, cfg.protein_col]:
            if c in g.columns:
                nonnull = g[c].dropna()
                row[c] = nonnull.iloc[0] if len(nonnull) else np.nan
        row.update(d)
        rows.append(row)
    return pd.DataFrame(rows)


def _variant_annotations(serial: pd.DataFrame, cfg: ProgressionFeatureConfig) -> pd.DataFrame:
    cols = [cfg.patient_col, cfg.variant_col]
    rows = []
    for (pid, vid), g in serial.groupby(cols, sort=False):
        row = {cfg.patient_col: pid, cfg.variant_col: vid}
        if cfg.gene_col in g.columns:
            vals = g[cfg.gene_col].dropna()
            row[cfg.gene_col] = vals.iloc[0] if len(vals) else np.nan
        if cfg.protein_col in g.columns:
            vals = g[cfg.protein_col].dropna()
            row[cfg.protein_col] = vals.iloc[0] if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def supported_comutation_pairs(
    serial: pd.DataFrame,
    pair_support: pd.DataFrame,
    cfg: ProgressionFeatureConfig = ProgressionFeatureConfig(),
) -> pd.DataFrame:
    """Annotate high-support variant pairs with genes/protein changes.

    This is candidate same-clone evidence from longitudinal VAF, not proof that
    the two variants reside in the same biological cell.
    """
    if pair_support is None or pair_support.empty:
        return pd.DataFrame()
    ann = _variant_annotations(serial, cfg)
    a = ann.rename(columns={
        cfg.variant_col: "variant_a",
        cfg.gene_col: "gene_a",
        cfg.protein_col: "protein_a",
    })
    b = ann.rename(columns={
        cfg.variant_col: "variant_b",
        cfg.gene_col: "gene_b",
        cfg.protein_col: "protein_b",
    })
    keep_a = [cfg.patient_col, "variant_a"] + [c for c in ["gene_a", "protein_a"] if c in a.columns]
    keep_b = [cfg.patient_col, "variant_b"] + [c for c in ["gene_b", "protein_b"] if c in b.columns]

    d = (pair_support.merge(a[keep_a], on=[cfg.patient_col, "variant_a"], how="left")
                     .merge(b[keep_b], on=[cfg.patient_col, "variant_b"], how="left"))
    d = d[pd.to_numeric(d["support"], errors="coerce") >= cfg.pair_support_threshold].copy()
    if "gene_a" in d.columns and "gene_b" in d.columns:
        d["gene_pair"] = [
            "|".join(sorted([str(x), str(y)])) if pd.notna(x) and pd.notna(y) else np.nan
            for x, y in zip(d["gene_a"], d["gene_b"])
        ]
    return d


def _patient_gene_table(serial: pd.DataFrame, cfg: ProgressionFeatureConfig) -> pd.DataFrame:
    if cfg.gene_col not in serial.columns:
        return serial[[cfg.patient_col, cfg.cohort_col]].drop_duplicates().copy()
    rows = []
    for (pid, cohort), g in serial.groupby([cfg.patient_col, cfg.cohort_col], sort=False):
        genes = sorted(set(str(x) for x in g[cfg.gene_col].dropna() if str(x)))
        row = {
            cfg.patient_col: pid,
            cfg.cohort_col: cohort,
            "n_genes_observed": int(len(genes)),
            "n_observed_gene_pairs": int(len(genes) * (len(genes) - 1) // 2),
            "genes": ";".join(genes),
        }
        gene_set = set(genes)
        for gene in cfg.gene_flags:
            row[f"has_gene_{gene}"] = int(gene in gene_set)
        rows.append(row)
    return pd.DataFrame(rows)


def patient_progression_features(
    serial: pd.DataFrame,
    *,
    trajectory_posteriors: Optional[pd.DataFrame] = None,
    pair_support: Optional[pd.DataFrame] = None,
    assignments: Optional[pd.DataFrame] = None,
    clone_posteriors: Optional[pd.DataFrame] = None,
    patient_clone_features: Optional[pd.DataFrame] = None,
    cfg: ProgressionFeatureConfig = ProgressionFeatureConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one-row-per-patient v2.2 research feature matrix.

    Returns
    -------
    patient_features, variant_kinetics_table, supported_pair_table
    """
    vk = variant_kinetics(serial, cfg)
    pair_hi = supported_comutation_pairs(serial, pair_support, cfg) if pair_support is not None else pd.DataFrame()

    base = serial[[cfg.patient_col, cfg.cohort_col]].drop_duplicates().copy()

    # Observed VAF + first/second-order kinetics.
    agg = vk.groupby([cfg.patient_col, cfg.cohort_col], as_index=False).agg(
        n_variants_observed=(cfg.variant_col, "nunique"),
        max_vaf_observed=("max_vaf_observed", "max"),
        max_delta_vaf_per_year=("delta_vaf_per_year", "max"),
        max_observed_log_vaf_growth=("observed_log_vaf_growth", "max"),
        median_observed_log_vaf_growth=("observed_log_vaf_growth", "median"),
        fastest_observed_doubling_time_years=("observed_doubling_time_years", "min"),
        median_observed_doubling_time_years=("observed_doubling_time_years", "median"),
        n_acceleration_estimable=("log_vaf_acceleration", lambda x: int(pd.Series(x).notna().sum())),
        max_log_vaf_acceleration=("log_vaf_acceleration", "max"),
        median_log_vaf_acceleration=("log_vaf_acceleration", "median"),
        max_end_log_vaf_growth=("end_log_vaf_growth", "max"),
    )
    for thr in cfg.vaf_thresholds:
        tag = _threshold_tag(thr)
        per_patient = vk.groupby([cfg.patient_col, cfg.cohort_col]).agg(
            **{
                f"n_variants_ever_vaf_ge_{tag}": (f"ever_vaf_ge_{tag}", "sum"),
                f"n_variants_crossed_vaf_ge_{tag}": (f"crossed_vaf_ge_{tag}", "sum"),
            }
        ).reset_index()
        per_patient[f"any_vaf_ge_{tag}"] = (per_patient[f"n_variants_ever_vaf_ge_{tag}"] > 0).astype(int)
        agg = agg.merge(per_patient, on=[cfg.patient_col, cfg.cohort_col], how="left")

    out = base.merge(agg, on=[cfg.patient_col, cfg.cohort_col], how="left", validate="one_to_one")

    # Gene burden and presence flags.
    out = out.merge(_patient_gene_table(serial, cfg), on=[cfg.patient_col, cfg.cohort_col], how="left", validate="one_to_one")

    # Posterior trajectory features from v2.1.
    if trajectory_posteriors is not None and len(trajectory_posteriors):
        tr = trajectory_posteriors.copy()
        tr["posterior_doubling_time_years"] = pd.to_numeric(tr["growth_mean"], errors="coerce").map(_positive_doubling_time)
        ta = tr.groupby([cfg.patient_col, cfg.cohort_col], as_index=False).agg(
            posterior_max_growth=("growth_mean", "max"),
            posterior_median_growth=("growth_mean", "median"),
            posterior_median_growth_sd=("growth_sd", "median"),
            posterior_max_p_growing=("p_growing", "max"),
            posterior_max_p_fast_growth=("p_fast_growth", "max"),
            fastest_posterior_doubling_time_years=("posterior_doubling_time_years", "min"),
            median_posterior_doubling_time_years=("posterior_doubling_time_years", "median"),
        )
        out = out.merge(ta, on=[cfg.patient_col, cfg.cohort_col], how="left", validate="one_to_one")

    # Pairwise architecture.
    if pair_support is not None:
        p = pair_support.copy()
        pa = p.groupby([cfg.patient_col, cfg.cohort_col], as_index=False).agg(
            n_variant_pairs_evaluated=("support", "size"),
            max_pair_support=("support", "max"),
            mean_pair_support=("support", "mean"),
        )
        if len(pair_hi):
            hi = pair_hi.groupby([cfg.patient_col, cfg.cohort_col], as_index=False).agg(
                n_supported_variant_pairs=("support", "size"),
                mean_supported_pair_support=("support", "mean"),
            )
            if "gene_pair" in pair_hi.columns:
                gp = pair_hi.groupby([cfg.patient_col, cfg.cohort_col], as_index=False).agg(
                    n_supported_gene_pairs=("gene_pair", lambda x: int(pd.Series(x).dropna().nunique()))
                )
                hi = hi.merge(gp, on=[cfg.patient_col, cfg.cohort_col], how="left")
            pa = pa.merge(hi, on=[cfg.patient_col, cfg.cohort_col], how="left")
        pa["n_supported_variant_pairs"] = pa.get("n_supported_variant_pairs", 0)
        pa["n_supported_variant_pairs"] = pd.to_numeric(pa["n_supported_variant_pairs"], errors="coerce").fillna(0)
        pa["fraction_evaluated_pairs_supported"] = pa["n_supported_variant_pairs"] / pa["n_variant_pairs_evaluated"].clip(lower=1)
        if "n_supported_gene_pairs" in pa.columns:
            pa["n_supported_gene_pairs"] = pa["n_supported_gene_pairs"].fillna(0)
        out = out.merge(pa, on=[cfg.patient_col, cfg.cohort_col], how="left", validate="one_to_one")
        for col in ["n_variant_pairs_evaluated", "n_supported_variant_pairs", "n_supported_gene_pairs"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        if "fraction_evaluated_pairs_supported" in out.columns:
            out["fraction_evaluated_pairs_supported"] = out["fraction_evaluated_pairs_supported"].fillna(0.0)

    # Candidate-clone architecture and diversity.
    if clone_posteriors is not None and len(clone_posteriors):
        clone_rows = []
        for (pid, cohort), c in clone_posteriors.groupby([cfg.patient_col, cfg.cohort_col], sort=False):
            growth = pd.to_numeric(c["clone_growth_mean"], errors="coerce")
            vaf = pd.to_numeric(c["index_vaf_proxy"], errors="coerce").clip(lower=0)
            dt = growth.map(_positive_doubling_time)
            total_vaf = float(vaf.sum())
            dominant_fraction = float(vaf.max() / total_vaf) if total_vaf > 0 else np.nan
            entropy = _safe_entropy(vaf.to_numpy(float))
            row = {
                cfg.patient_col: pid,
                cfg.cohort_col: cohort,
                "n_candidate_clones_v22": int(len(c)),
                "max_variants_per_candidate_clone": int(pd.to_numeric(c["n_variants"], errors="coerce").max()),
                "clone_growth_dispersion": float(growth.std(ddof=1)) if growth.notna().sum() > 1 else 0.0,
                "clone_growth_range": float(growth.max() - growth.min()) if growth.notna().any() else np.nan,
                "n_fast_growth_clones_mean": int((growth > cfg.fast_growth_threshold).sum()),
                "fastest_clone_doubling_time_years": float(dt.min()) if dt.notna().any() else np.nan,
                "median_clone_doubling_time_years": float(dt.median()) if dt.notna().any() else np.nan,
                "dominant_clone_fraction_proxy": dominant_fraction,
                "clone_vaf_entropy": entropy,
                "effective_clone_number_proxy": float(np.exp(entropy)) if np.isfinite(entropy) else np.nan,
                "any_clone_member_vaf_ge_0p20": int((pd.to_numeric(c["max_member_vaf"], errors="coerce") >= 0.20).any()),
                "n_clones_member_vaf_ge_0p20": int((pd.to_numeric(c["max_member_vaf"], errors="coerce") >= 0.20).sum()),
            }
            clone_rows.append(row)
        out = out.merge(pd.DataFrame(clone_rows), on=[cfg.patient_col, cfg.cohort_col], how="left", validate="one_to_one")

    # Reuse v2.1 patient clone features without recomputing them.
    if patient_clone_features is not None and len(patient_clone_features):
        pc = patient_clone_features.copy()
        duplicate = [c for c in pc.columns if c in out.columns and c not in {cfg.patient_col, cfg.cohort_col}]
        pc = pc.drop(columns=duplicate, errors="ignore")
        out = out.merge(pc, on=[cfg.patient_col, cfg.cohort_col], how="left", validate="one_to_one")

    return out, vk, pair_hi


def feature_manifest(
    patient_features: pd.DataFrame,
    cfg: ProgressionFeatureConfig = ProgressionFeatureConfig(),
) -> Dict[str, object]:
    """Machine-readable description of the v2.2 feature matrix."""
    return {
        "version": "LeukemiPrediag-v2.2",
        "research_use_only": True,
        "not_a_diagnostic_device": True,
        "does_not_create_outcome_labels": True,
        "config": asdict(cfg),
        "n_patients": int(len(patient_features)),
        "feature_columns": [
            c for c in patient_features.columns
            if c not in {cfg.patient_col, cfg.cohort_col, "genes"}
        ],
        "experimental_features": [
            "max_log_vaf_acceleration",
            "median_log_vaf_acceleration",
            "max_end_log_vaf_growth",
        ],
        "interpretation_notes": {
            "vaf_thresholds": "Covariates only; not standalone diagnostic rules.",
            "doubling_time": "Defined only for positive log-VAF growth estimates.",
            "acceleration": "Exploratory second derivative; requires >=4 timepoints and is noise-sensitive.",
            "supported_comutation_pairs": "Longitudinal co-movement evidence; not proof of single-cell co-occurrence.",
            "gene_flags": "Presence indicators only; not hard-coded risk labels.",
        },
    }
