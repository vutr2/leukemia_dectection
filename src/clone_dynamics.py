"""Conservative clone-dynamics features for LeukemiPrediag v2.

This first version does NOT claim to reconstruct a true phylogeny from VAF alone.
It builds trajectory-level and patient-level features, proposes co-moving variant
groups when serial measurements support them, and exposes uncertainty so future
HMM/Bayesian clone models have a clean input contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DynamicsConfig:
    patient_col: str = "patient_id"
    variant_col: str = "variant_id"
    time_col: str = "time_years"
    vaf_col: str = "vaf"
    min_points_per_variant: int = 3
    min_shared_points: int = 3
    corr_threshold: float = 0.90
    log_eps: float = 1e-5


def _slope(t: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(t) < 2 or np.ptp(t) <= 0:
        return np.nan, np.nan
    x = np.asarray(t, float)
    z = np.log(np.clip(np.asarray(y, float), 0, None) + 1e-5)
    b, a = np.polyfit(x, z, 1)
    pred = a + b * x
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))
    return float(b), rmse


def trajectory_table(df: pd.DataFrame, cfg: DynamicsConfig = DynamicsConfig()) -> pd.DataFrame:
    """One row per patient/variant with slope and trajectory QC."""
    req = [cfg.patient_col, cfg.variant_col, cfg.time_col, cfg.vaf_col]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"Missing dynamics fields: {miss}")

    rows = []
    for (pid, vid), g in df.groupby([cfg.patient_col, cfg.variant_col], sort=False):
        g = g[[cfg.time_col, cfg.vaf_col]].dropna().sort_values(cfg.time_col)
        # collapse technical duplicates at a timepoint
        g = g.groupby(cfg.time_col, as_index=False)[cfg.vaf_col].mean()
        if len(g) < cfg.min_points_per_variant:
            continue
        s, rmse = _slope(g[cfg.time_col].to_numpy(), g[cfg.vaf_col].to_numpy())
        rows.append({
            cfg.patient_col: pid,
            cfg.variant_col: vid,
            "n_timepoints": len(g),
            "first_vaf": float(g[cfg.vaf_col].iloc[0]),
            "last_vaf": float(g[cfg.vaf_col].iloc[-1]),
            "max_vaf": float(g[cfg.vaf_col].max()),
            "log_vaf_slope": s,
            "trajectory_rmse": rmse,
            "time_span_years": float(g[cfg.time_col].max() - g[cfg.time_col].min()),
        })
    return pd.DataFrame(rows)


def pairwise_comovement(df: pd.DataFrame, cfg: DynamicsConfig = DynamicsConfig()) -> pd.DataFrame:
    """Pairwise within-patient trajectory correlation on shared sample times.

    High correlation is evidence of co-movement, not proof of same-clone ancestry.
    """
    rows: List[dict] = []
    for pid, pg in df.groupby(cfg.patient_col, sort=False):
        piv = pg.pivot_table(index=cfg.time_col, columns=cfg.variant_col,
                             values=cfg.vaf_col, aggfunc="mean")
        vids = list(piv.columns)
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                a, b = vids[i], vids[j]
                z = piv[[a, b]].dropna()
                if len(z) < cfg.min_shared_points:
                    continue
                # correlation on log VAF is more aligned with exponential growth
                x = np.log(np.clip(z[a].to_numpy(float), 0, None) + cfg.log_eps)
                y = np.log(np.clip(z[b].to_numpy(float), 0, None) + cfg.log_eps)
                corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else np.nan
                rows.append({
                    cfg.patient_col: pid,
                    "variant_a": a,
                    "variant_b": b,
                    "shared_timepoints": len(z),
                    "log_vaf_corr": corr,
                    "candidate_coclone": bool(np.isfinite(corr) and corr >= cfg.corr_threshold),
                })
    return pd.DataFrame(rows)


def patient_dynamics_features(
    serial_df: pd.DataFrame,
    cfg: DynamicsConfig = DynamicsConfig(),
) -> pd.DataFrame:
    """Create inspectable patient-level dynamic features for baseline comparisons."""
    traj = trajectory_table(serial_df, cfg)
    pairs = pairwise_comovement(serial_df, cfg)
    if traj.empty:
        return pd.DataFrame(columns=[cfg.patient_col])

    rows = []
    for pid, g in traj.groupby(cfg.patient_col):
        slopes = g["log_vaf_slope"].dropna().to_numpy(float)
        p = pairs[pairs[cfg.patient_col] == pid] if not pairs.empty else pd.DataFrame()
        finite_corr = p["log_vaf_corr"].dropna() if len(p) else pd.Series(dtype=float)
        rows.append({
            cfg.patient_col: pid,
            "n_tracked_variants": int(len(g)),
            "n_growing_variants": int((g["log_vaf_slope"] > 0).sum()),
            "max_log_vaf_slope": float(np.max(slopes)) if len(slopes) else np.nan,
            "median_log_vaf_slope": float(np.median(slopes)) if len(slopes) else np.nan,
            "slope_dispersion": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
            "max_observed_vaf": float(g["max_vaf"].max()),
            "median_trajectory_rmse": float(g["trajectory_rmse"].median()),
            "n_candidate_coclone_pairs": int(p["candidate_coclone"].sum()) if len(p) else 0,
            "mean_pairwise_log_vaf_corr": float(finite_corr.mean()) if len(finite_corr) else np.nan,
            "structure_identifiable": bool(len(p) > 0),
        })
    return pd.DataFrame(rows)
