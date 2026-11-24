"""Probabilistic clone inference for LeukemiPrediag v2.

This module upgrades the conservative trajectory/correlation layer into an
uncertainty-aware *candidate* clone model.  It does not claim to recover a true
phylogeny from VAF alone.

Core ideas
----------
1. Estimate cohort-level log-VAF noise from residuals around simple trajectory
   fits.  For cohorts with per-call depth, sequencing/binomial uncertainty is
   also included.  SardiNIA uses the empirical cohort noise only because the
   public supplement does not provide per-call depth.
2. For each variant trajectory, sample plausible log-VAF paths and derive a
   posterior over growth rate.
3. For each within-patient variant pair, compute posterior *co-movement support*
   based on correlation, ratio stability, and slope similarity.
4. Form conservative candidate clone groups using complete-link clustering:
   every cross-pair between two groups must exceed the support threshold before
   the groups can merge.
5. Estimate a clone-level common growth slope with variant-specific intercepts.

Outputs are research features + uncertainty summaries for the v2 fusion model.
They are not clinical predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CloneInferenceConfig:
    patient_col: str = "patient_id"
    cohort_col: str = "cohort"
    variant_col: str = "variant_id"
    time_col: str = "time_years"
    vaf_col: str = "vaf"
    depth_col: str = "depth"
    min_timepoints: int = 3
    n_draws: int = 500
    seed: int = 17
    log_eps: float = 1e-5
    min_log_sigma: float = 0.05
    # SardiNIA's public supplement has no per-call depth; the adapter carries a
    # nominal depth only to keep QC filtering inert. Do not treat it as read depth.
    cohorts_without_call_depth: Tuple[str, ...] = ("SardiNIA",)
    corr_threshold: float = 0.80
    ratio_log_sd_threshold: float = 0.30
    slope_diff_threshold: float = 0.05  # per year, log-VAF scale
    pair_support_threshold: float = 0.80
    fast_growth_threshold: float = 0.05  # per year, log-VAF scale


def _linear_slope_matrix(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorized OLS slope. Y shape = (draws, n_timepoints)."""
    x = np.asarray(x, float)
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    if den <= 0:
        return np.full(Y.shape[0], np.nan)
    return (Y @ xc) / den


def _robust_sigma(x: np.ndarray, floor: float) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float(floor)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 0:
        s = np.std(x, ddof=1) if len(x) > 1 else floor
    return float(max(s, floor))


def estimate_cohort_log_noise(
    serial: pd.DataFrame,
    cfg: CloneInferenceConfig = CloneInferenceConfig(),
) -> Dict[str, float]:
    """Empirical log-VAF residual noise by cohort.

    This deliberately estimates a conservative measurement/trajectory noise floor
    from the same cohort. It is not a biological fitness prior.
    """
    req = [cfg.patient_col, cfg.cohort_col, cfg.variant_col, cfg.time_col, cfg.vaf_col]
    miss = [c for c in req if c not in serial.columns]
    if miss:
        raise KeyError(f"serial missing required columns: {miss}")

    residuals: Dict[str, List[float]] = {}
    for (cohort, _pid, _vid), g in serial.groupby(
        [cfg.cohort_col, cfg.patient_col, cfg.variant_col], sort=False
    ):
        g = g[[cfg.time_col, cfg.vaf_col]].dropna().sort_values(cfg.time_col)
        g = g.groupby(cfg.time_col, as_index=False)[cfg.vaf_col].mean()
        if len(g) < cfg.min_timepoints or np.ptp(g[cfg.time_col].to_numpy(float)) <= 0:
            continue
        x = g[cfg.time_col].to_numpy(float)
        y = np.log(np.clip(g[cfg.vaf_col].to_numpy(float), cfg.log_eps, None))
        b, a = np.polyfit(x, y, 1)
        r = y - (a + b * x)
        residuals.setdefault(str(cohort), []).extend(r.tolist())

    return {
        cohort: _robust_sigma(np.asarray(rs), cfg.min_log_sigma)
        for cohort, rs in residuals.items()
    }


def _obs_log_sigma(
    g: pd.DataFrame,
    cohort_noise: Mapping[str, float],
    cfg: CloneInferenceConfig,
) -> np.ndarray:
    cohort = str(g[cfg.cohort_col].iloc[0])
    base = float(cohort_noise.get(cohort, cfg.min_log_sigma))
    sig = np.full(len(g), base, dtype=float)

    if cfg.depth_col in g.columns and cohort not in cfg.cohorts_without_call_depth:
        depth = pd.to_numeric(g[cfg.depth_col], errors="coerce").to_numpy(float)
        v = np.clip(pd.to_numeric(g[cfg.vaf_col], errors="coerce").to_numpy(float),
                    cfg.log_eps, 1 - cfg.log_eps)
        # Delta-method SE for log(VAF) under binomial sampling.
        seq = np.sqrt(np.maximum((1.0 - v) / np.maximum(depth * v, 1e-9), 0.0))
        seq[~np.isfinite(seq)] = 0.0
        sig = np.sqrt(base * base + seq * seq)
    return np.maximum(sig, cfg.min_log_sigma)


def _sample_log_vaf(
    g: pd.DataFrame,
    cohort_noise: Mapping[str, float],
    cfg: CloneInferenceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    mu = np.log(np.clip(pd.to_numeric(g[cfg.vaf_col], errors="coerce").to_numpy(float),
                        cfg.log_eps, None))
    sigma = _obs_log_sigma(g, cohort_noise, cfg)
    return rng.normal(mu[None, :], sigma[None, :], size=(cfg.n_draws, len(g)))


def trajectory_posteriors(
    serial: pd.DataFrame,
    cohort_noise: Optional[Mapping[str, float]] = None,
    cfg: CloneInferenceConfig = CloneInferenceConfig(),
) -> pd.DataFrame:
    """Posterior growth-rate summaries for every patient/variant trajectory."""
    if cohort_noise is None:
        cohort_noise = estimate_cohort_log_noise(serial, cfg)
    rng = np.random.default_rng(cfg.seed)
    rows: List[dict] = []

    for (pid, cohort, vid), g in serial.groupby(
        [cfg.patient_col, cfg.cohort_col, cfg.variant_col], sort=False
    ):
        keep = [cfg.time_col, cfg.vaf_col]
        if cfg.depth_col in g.columns:
            keep.append(cfg.depth_col)
        g = g[keep].dropna(subset=[cfg.time_col, cfg.vaf_col]).sort_values(cfg.time_col)
        agg = {cfg.vaf_col: "mean"}
        if cfg.depth_col in g.columns:
            agg[cfg.depth_col] = "median"
        g = g.groupby(cfg.time_col, as_index=False).agg(agg)
        if len(g) < cfg.min_timepoints:
            continue
        g[cfg.cohort_col] = cohort
        Y = _sample_log_vaf(g, cohort_noise, cfg, rng)
        slopes = _linear_slope_matrix(g[cfg.time_col].to_numpy(float), Y)
        slopes = slopes[np.isfinite(slopes)]
        if not len(slopes):
            continue
        rows.append({
            cfg.patient_col: pid,
            cfg.cohort_col: cohort,
            cfg.variant_col: vid,
            "n_timepoints": int(len(g)),
            "growth_mean": float(np.mean(slopes)),
            "growth_median": float(np.median(slopes)),
            "growth_sd": float(np.std(slopes, ddof=1)),
            "growth_ci_low": float(np.percentile(slopes, 2.5)),
            "growth_ci_high": float(np.percentile(slopes, 97.5)),
            "p_growing": float(np.mean(slopes > 0.0)),
            "p_fast_growth": float(np.mean(slopes > cfg.fast_growth_threshold)),
            "max_vaf": float(g[cfg.vaf_col].max()),
            "last_vaf": float(g.sort_values(cfg.time_col)[cfg.vaf_col].iloc[-1]),
            "time_span_years": float(np.ptp(g[cfg.time_col].to_numpy(float))),
        })
    return pd.DataFrame(rows)


def _pair_draw_support(
    a: pd.DataFrame,
    b: pd.DataFrame,
    cohort_noise: Mapping[str, float],
    cfg: CloneInferenceConfig,
    rng: np.random.Generator,
) -> Optional[dict]:
    ca = a[[cfg.time_col, cfg.vaf_col] + ([cfg.depth_col] if cfg.depth_col in a else [])].copy()
    cb = b[[cfg.time_col, cfg.vaf_col] + ([cfg.depth_col] if cfg.depth_col in b else [])].copy()
    ca = ca.rename(columns={cfg.vaf_col: "vaf_a", cfg.depth_col: "depth_a"})
    cb = cb.rename(columns={cfg.vaf_col: "vaf_b", cfg.depth_col: "depth_b"})
    z = ca.merge(cb, on=cfg.time_col, how="inner").sort_values(cfg.time_col)
    if len(z) < cfg.min_timepoints:
        return None

    cohort = str(a[cfg.cohort_col].iloc[0])
    ga = pd.DataFrame({cfg.time_col: z[cfg.time_col], cfg.vaf_col: z["vaf_a"]})
    gb = pd.DataFrame({cfg.time_col: z[cfg.time_col], cfg.vaf_col: z["vaf_b"]})
    if "depth_a" in z:
        ga[cfg.depth_col] = z["depth_a"]
    if "depth_b" in z:
        gb[cfg.depth_col] = z["depth_b"]
    ga[cfg.cohort_col] = cohort
    gb[cfg.cohort_col] = cohort

    Ya = _sample_log_vaf(ga, cohort_noise, cfg, rng)
    Yb = _sample_log_vaf(gb, cohort_noise, cfg, rng)
    x = z[cfg.time_col].to_numpy(float)
    sa = _linear_slope_matrix(x, Ya)
    sb = _linear_slope_matrix(x, Yb)

    ca0 = Ya - Ya.mean(axis=1, keepdims=True)
    cb0 = Yb - Yb.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(ca0 * ca0, axis=1) * np.sum(cb0 * cb0, axis=1))
    corr = np.divide(np.sum(ca0 * cb0, axis=1), den,
                     out=np.zeros_like(den), where=den > 0)
    ratio_sd = np.std(Ya - Yb, axis=1, ddof=1)
    slope_diff = np.abs(sa - sb)

    c_corr = corr >= cfg.corr_threshold
    c_ratio = ratio_sd <= cfg.ratio_log_sd_threshold
    c_slope = slope_diff <= cfg.slope_diff_threshold
    # Support requires at least two independent forms of co-movement evidence.
    support = np.mean((c_corr.astype(int) + c_ratio.astype(int) + c_slope.astype(int)) >= 2)

    return {
        "shared_timepoints": int(len(z)),
        "support": float(support),
        "p_corr": float(np.mean(c_corr)),
        "p_ratio_stable": float(np.mean(c_ratio)),
        "p_slope_similar": float(np.mean(c_slope)),
        "corr_median": float(np.median(corr)),
        "ratio_log_sd_median": float(np.median(ratio_sd)),
        "slope_diff_median": float(np.median(slope_diff)),
    }


def pairwise_clone_support(
    serial: pd.DataFrame,
    cohort_noise: Optional[Mapping[str, float]] = None,
    cfg: CloneInferenceConfig = CloneInferenceConfig(),
) -> pd.DataFrame:
    """Posterior co-movement support for every within-patient variant pair."""
    if cohort_noise is None:
        cohort_noise = estimate_cohort_log_noise(serial, cfg)
    rng = np.random.default_rng(cfg.seed + 1009)
    rows: List[dict] = []

    for (pid, cohort), pg in serial.groupby([cfg.patient_col, cfg.cohort_col], sort=False):
        vids = list(dict.fromkeys(pg[cfg.variant_col].astype(str)))
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                a = pg[pg[cfg.variant_col].astype(str) == vids[i]].copy()
                b = pg[pg[cfg.variant_col].astype(str) == vids[j]].copy()
                r = _pair_draw_support(a, b, cohort_noise, cfg, rng)
                if r is None:
                    continue
                r.update({
                    cfg.patient_col: pid,
                    cfg.cohort_col: cohort,
                    "variant_a": vids[i],
                    "variant_b": vids[j],
                    "candidate_same_clone": bool(r["support"] >= cfg.pair_support_threshold),
                })
                rows.append(r)
    return pd.DataFrame(rows)


def _complete_link_groups(variants: Sequence[str], pair_support: Mapping[Tuple[str, str], float],
                          threshold: float) -> List[List[str]]:
    """Conservative agglomeration: every cross-pair must clear threshold."""
    clusters = [[str(v)] for v in variants]

    def ps(a: str, b: str) -> float:
        if a == b:
            return 1.0
        k = tuple(sorted((str(a), str(b))))
        return float(pair_support.get(k, 0.0))

    while True:
        best = None
        best_mean = -1.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                vals = [ps(a, b) for a in clusters[i] for b in clusters[j]]
                if vals and min(vals) >= threshold:
                    m = float(np.mean(vals))
                    if m > best_mean:
                        best = (i, j)
                        best_mean = m
        if best is None:
            break
        i, j = best
        merged = sorted(clusters[i] + clusters[j])
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    return sorted(clusters, key=lambda c: (-len(c), c))


def infer_clone_assignments(
    serial: pd.DataFrame,
    pair_table: pd.DataFrame,
    cfg: CloneInferenceConfig = CloneInferenceConfig(),
) -> pd.DataFrame:
    """Assign conservative candidate clone groups per patient."""
    rows: List[dict] = []
    for (pid, cohort), pg in serial.groupby([cfg.patient_col, cfg.cohort_col], sort=False):
        vids = sorted(pg[cfg.variant_col].astype(str).unique())
        pp = pair_table[pair_table[cfg.patient_col] == pid] if len(pair_table) else pd.DataFrame()
        lookup: Dict[Tuple[str, str], float] = {}
        if len(pp):
            for _, r in pp.iterrows():
                lookup[tuple(sorted((str(r["variant_a"]), str(r["variant_b"]))))] = float(r["support"])
        groups = _complete_link_groups(vids, lookup, cfg.pair_support_threshold)
        for k, members in enumerate(groups, 1):
            vals = []
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    vals.append(lookup.get(tuple(sorted((members[i], members[j]))), 0.0))
            coherence = float(min(vals)) if vals else np.nan
            mean_support = float(np.mean(vals)) if vals else np.nan
            for v in members:
                rows.append({
                    cfg.patient_col: pid,
                    cfg.cohort_col: cohort,
                    cfg.variant_col: v,
                    "clone_id": f"{pid}::clone_{k}",
                    "clone_index": k,
                    "clone_size_variants": len(members),
                    "clone_min_pair_support": coherence,
                    "clone_mean_pair_support": mean_support,
                })
    return pd.DataFrame(rows)


def _clone_common_slope_draws(
    g: pd.DataFrame,
    cohort_noise: Mapping[str, float],
    cfg: CloneInferenceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Common slope with variant-specific intercepts via within-variant centering."""
    pieces = []
    denom = 0.0
    for _vid, vg in g.groupby(cfg.variant_col, sort=False):
        cols = [cfg.time_col, cfg.vaf_col]
        if cfg.depth_col in vg.columns:
            cols.append(cfg.depth_col)
        z = vg[cols].dropna(subset=[cfg.time_col, cfg.vaf_col]).sort_values(cfg.time_col)
        agg = {cfg.vaf_col: "mean"}
        if cfg.depth_col in z.columns:
            agg[cfg.depth_col] = "median"
        z = z.groupby(cfg.time_col, as_index=False).agg(agg)
        if len(z) < cfg.min_timepoints:
            continue
        z[cfg.cohort_col] = str(g[cfg.cohort_col].iloc[0])
        Y = _sample_log_vaf(z, cohort_noise, cfg, rng)
        x = z[cfg.time_col].to_numpy(float)
        xc = x - x.mean()
        denom += float(np.sum(xc * xc))
        Yc = Y - Y.mean(axis=1, keepdims=True)
        pieces.append(Yc @ xc)
    if not pieces or denom <= 0:
        return np.full(cfg.n_draws, np.nan)
    return np.sum(np.vstack(pieces), axis=0) / denom


def clone_posteriors(
    serial: pd.DataFrame,
    assignments: pd.DataFrame,
    cohort_noise: Optional[Mapping[str, float]] = None,
    cfg: CloneInferenceConfig = CloneInferenceConfig(),
) -> pd.DataFrame:
    """Clone-level common growth posterior and index VAF proxy."""
    if cohort_noise is None:
        cohort_noise = estimate_cohort_log_noise(serial, cfg)
    rng = np.random.default_rng(cfg.seed + 2027)
    d = serial.merge(
        assignments[[cfg.patient_col, cfg.variant_col, "clone_id", "clone_size_variants",
                     "clone_min_pair_support", "clone_mean_pair_support"]],
        on=[cfg.patient_col, cfg.variant_col], how="inner", validate="many_to_one"
    )
    rows: List[dict] = []
    for (pid, cohort, cid), g in d.groupby([cfg.patient_col, cfg.cohort_col, "clone_id"], sort=False):
        slopes = _clone_common_slope_draws(g, cohort_noise, cfg, rng)
        slopes = slopes[np.isfinite(slopes)]
        if not len(slopes):
            continue
        last_t = g[cfg.time_col].max()
        last = g[np.isclose(g[cfg.time_col], last_t)]
        rows.append({
            cfg.patient_col: pid,
            cfg.cohort_col: cohort,
            "clone_id": cid,
            "n_variants": int(g[cfg.variant_col].nunique()),
            "index_vaf_proxy": float(last[cfg.vaf_col].median()),
            "max_member_vaf": float(g[cfg.vaf_col].max()),
            "clone_growth_mean": float(np.mean(slopes)),
            "clone_growth_median": float(np.median(slopes)),
            "clone_growth_sd": float(np.std(slopes, ddof=1)),
            "clone_growth_ci_low": float(np.percentile(slopes, 2.5)),
            "clone_growth_ci_high": float(np.percentile(slopes, 97.5)),
            "clone_p_growing": float(np.mean(slopes > 0.0)),
            "clone_p_fast_growth": float(np.mean(slopes > cfg.fast_growth_threshold)),
            "clone_min_pair_support": float(g["clone_min_pair_support"].dropna().iloc[0])
                if g["clone_min_pair_support"].notna().any() else np.nan,
            "clone_mean_pair_support": float(g["clone_mean_pair_support"].dropna().iloc[0])
                if g["clone_mean_pair_support"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def patient_clone_features(
    serial: pd.DataFrame,
    assignments: pd.DataFrame,
    clones: pd.DataFrame,
    cfg: CloneInferenceConfig = CloneInferenceConfig(),
) -> pd.DataFrame:
    """Patient-level clone-aware features ready for fusion_model.py."""
    rows: List[dict] = []
    for (pid, cohort), pg in serial.groupby([cfg.patient_col, cfg.cohort_col], sort=False):
        a = assignments[assignments[cfg.patient_col] == pid]
        c = clones[clones[cfg.patient_col] == pid]
        n_variants = int(pg[cfg.variant_col].nunique())
        if c.empty:
            rows.append({cfg.patient_col: pid, cfg.cohort_col: cohort})
            continue
        dominant = c.sort_values("index_vaf_proxy", ascending=False).iloc[0]
        multi = c[c["n_variants"] > 1]
        grouped_variants = int(a[a["clone_size_variants"] > 1][cfg.variant_col].nunique()) if len(a) else 0
        rows.append({
            cfg.patient_col: pid,
            cfg.cohort_col: cohort,
            "n_inferred_clones": int(len(c)),
            "n_multivariant_clones": int(len(multi)),
            "fraction_variants_grouped": grouped_variants / max(n_variants, 1),
            "max_clone_growth": float(c["clone_growth_mean"].max()),
            "median_clone_growth": float(c["clone_growth_mean"].median()),
            "max_clone_p_growing": float(c["clone_p_growing"].max()),
            "max_clone_p_fast_growth": float(c["clone_p_fast_growth"].max()),
            "dominant_clone_vaf": float(dominant["index_vaf_proxy"]),
            "dominant_clone_growth": float(dominant["clone_growth_mean"]),
            "dominant_clone_p_growing": float(dominant["clone_p_growing"]),
            "mean_clone_coherence": float(multi["clone_mean_pair_support"].mean()) if len(multi) else np.nan,
            "min_clone_coherence": float(multi["clone_min_pair_support"].min()) if len(multi) else np.nan,
            "structure_supported": int(len(multi) > 0),
        })
    return pd.DataFrame(rows)


def run_clone_inference(
    serial: pd.DataFrame,
    cfg: CloneInferenceConfig = CloneInferenceConfig(),
) -> Dict[str, object]:
    """End-to-end uncertainty-aware clone inference."""
    noise = estimate_cohort_log_noise(serial, cfg)
    traj = trajectory_posteriors(serial, noise, cfg)
    pairs = pairwise_clone_support(serial, noise, cfg)
    assignments = infer_clone_assignments(serial, pairs, cfg)
    clones = clone_posteriors(serial, assignments, noise, cfg)
    patient = patient_clone_features(serial, assignments, clones, cfg)

    audit = {
        "config": asdict(cfg),
        "cohort_log_noise": noise,
        "n_patients": int(serial[cfg.patient_col].nunique()),
        "n_variants": int(serial.groupby([cfg.patient_col, cfg.variant_col]).ngroups),
        "n_pairs_evaluated": int(len(pairs)),
        "n_high_support_pairs": int(pairs["candidate_same_clone"].sum()) if len(pairs) else 0,
        "n_candidate_clones": int(len(clones)),
        "n_multivariant_clones": int((clones["n_variants"] > 1).sum()) if len(clones) else 0,
        "n_patients_with_supported_structure": int(patient.get("structure_supported", pd.Series(dtype=int)).fillna(0).sum()),
    }
    return {
        "cohort_noise": noise,
        "trajectory_posteriors": traj,
        "pair_support": pairs,
        "assignments": assignments,
        "clone_posteriors": clones,
        "patient_features": patient,
        "audit": audit,
    }
