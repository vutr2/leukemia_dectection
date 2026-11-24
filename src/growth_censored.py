"""LeukemiPrediag v2.5 — censoring-aware longitudinal VAF forecaster.

Separate experimental layer; does not modify v2.3 temporal_validation.py or
v2.4 growth_shrinkage.py.

Observation model:
* GSE178936 / GSE288742: binomial count information (AO/DP) is used directly,
  including zero and sub-1% measurements.
* SardiNIA: Supplementary Table 2 has VAF but no per-timepoint counts; observations
  below the documented 0.2% operational detection definition are treated as
  left-censored on the logit-VAF scale.

Latent state is deliberately simple: logit p(t) = intercept + slope * time.
The slope prior is fixed at SD=0.1/year (ridge penalty 50) rather than tuned on
held-out patients. Research use only. Not a diagnostic device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.special import expit, log_ndtr
from sklearn.model_selection import KFold

EPS = 1e-5
VAF_STRATUM_THRESHOLD = 0.02  # frozen exploratory v2.4 threshold; not retuned here
ALPHA_GRID = np.linspace(0.0, 1.0, 21)
MU_GRID = np.linspace(-0.10, 0.20, 61)
LAMBDA_GRID = (50.0,)  # fixed N(0, 0.1^2) slope prior: penalty = 50*slope^2
MODELS = ["carry_forward", "global_alpha", "drift_global", "alpha_vaf_stratum", "censored_growth"]


@dataclass(frozen=True)
class CensoredConfig:
    min_total_timepoints: int = 3
    sardinia_logit_sigma: float = 0.35
    slope_bound: float = 1.0
    logit_bound: float = 14.0


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _prepare_group(g: pd.DataFrame) -> pd.DataFrame:
    cols = ["time_years", "vaf", "depth", "alt_reads", "detection_limit", "censored"]
    z = g[[c for c in cols if c in g.columns]].copy()
    for c in cols:
        if c not in z.columns:
            z[c] = np.nan
    z = z.dropna(subset=["time_years", "vaf"]).sort_values("time_years")
    if z["time_years"].duplicated().any():
        z = z.groupby("time_years", as_index=False).agg({
            "vaf": "mean", "depth": "mean", "alt_reads": "mean",
            "detection_limit": "median", "censored": "max",
        })
    return z


def build_trajectory_table(serial: pd.DataFrame, cfg: CensoredConfig = CensoredConfig()) -> pd.DataFrame:
    required = ["patient_id", "cohort", "variant_id", "time_years", "vaf"]
    missing = [c for c in required if c not in serial.columns]
    if missing:
        raise KeyError(f"serial missing required columns: {missing}")
    rows = []
    for (pid, coh, vid), g in serial.groupby(["patient_id", "cohort", "variant_id"], sort=False):
        z = _prepare_group(g)
        if len(z) < cfg.min_total_timepoints:
            continue
        tr, te = z.iloc[:-1].copy(), z.iloc[-1]
        if len(tr) < 2:
            continue
        t = tr.time_years.to_numpy(float)
        tt = float(te.time_years)
        if np.ptp(t) <= 0 or tt <= t.max():
            continue
        v = np.clip(tr.vaf.to_numpy(float), 0.0, 1.0)
        lv = np.log(np.clip(v, EPS, 1.0))
        b_ols, _ = np.polyfit(t, lv, 1)
        gene = g["gene"].dropna().iloc[0] if "gene" in g.columns and g["gene"].notna().any() else np.nan
        rows.append({
            "patient_id": str(pid), "cohort": str(coh), "variant_id": str(vid), "gene": gene,
            "n_train": int(len(tr)), "horizon": float(tt - t.max()),
            "last_vaf": float(np.clip(v[-1], EPS, 1.0)),
            "last_lv": float(np.log(np.clip(v[-1], EPS, 1.0))),
            "true_vaf": float(np.clip(float(te.vaf), EPS, 1.0)),
            "s_ols": float(b_ols), "train_frame": tr.reset_index(drop=True),
        })
    return pd.DataFrame(rows)


def _initial_params(tr: pd.DataFrame) -> np.ndarray:
    t = tr.time_years.to_numpy(float)
    x = t - t.max()
    vals = []
    for _, r in tr.iterrows():
        n, k = r.get("depth", np.nan), r.get("alt_reads", np.nan)
        if np.isfinite(n) and n > 0 and np.isfinite(k):
            p = (float(k) + 0.5) / (float(n) + 1.0)
        else:
            v = float(r.vaf)
            lim = float(r.detection_limit) if np.isfinite(r.detection_limit) else 0.002
            p = v if v > 0 else lim / 2.0
        vals.append(float(_logit(p)))
    vals = np.asarray(vals)
    if len(vals) >= 2 and np.ptp(x) > 0:
        s, a = np.polyfit(x, vals, 1)
    else:
        s, a = 0.0, vals[-1]
    return np.array([float(a), float(np.clip(s, -0.5, 0.5))])


def fit_trajectory_forecast(tr: pd.DataFrame, horizon: float, ridge_lambda: float = 50.0,
                            cfg: CensoredConfig = CensoredConfig()) -> Tuple[float, bool]:
    """Hybrid binomial/censored-Gaussian IRLS fit for one training trajectory."""
    theta = _initial_params(tr)
    t = tr.time_years.to_numpy(float)
    x = t - t.max()
    X = np.column_stack([np.ones(len(x)), x])
    sigma = float(cfg.sardinia_logit_sigma)
    ok = True
    for _ in range(30):
        mu = X @ theta
        w = np.zeros(len(tr), float)
        z = np.zeros(len(tr), float)
        for j, (_, r) in enumerate(tr.iterrows()):
            n, k = r.get("depth", np.nan), r.get("alt_reads", np.nan)
            if np.isfinite(n) and n > 0 and np.isfinite(k) and 0 <= k <= n:
                p = float(np.clip(expit(mu[j]), 1e-7, 1 - 1e-7))
                info = max(float(n) * p * (1.0 - p), 1e-3)
                w[j] = info
                z[j] = mu[j] + (float(k) - float(n) * p) / info
                continue

            lim = float(r.detection_limit) if np.isfinite(r.detection_limit) else np.nan
            cens = bool(r.get("censored", False))
            v = float(r.vaf)
            w[j] = 1.0 / (sigma * sigma)
            if cens and np.isfinite(lim) and lim > 0:
                L = float(_logit(lim))
                a = (L - mu[j]) / sigma
                logphi = -0.5 * a * a - 0.5 * np.log(2 * np.pi)
                ratio = float(np.exp(logphi - log_ndtr(a))) if a > -35 else float(-a)
                z[j] = mu[j] - sigma * ratio
            elif v > 0:
                z[j] = float(_logit(v))
            elif np.isfinite(lim) and lim > 0:
                L = float(_logit(lim))
                a = (L - mu[j]) / sigma
                logphi = -0.5 * a * a - 0.5 * np.log(2 * np.pi)
                ratio = float(np.exp(logphi - log_ndtr(a))) if a > -35 else float(-a)
                z[j] = mu[j] - sigma * ratio
            else:
                w[j] = 0.0
                z[j] = mu[j]
        if not np.isfinite(w).all() or not np.isfinite(z).all() or w.sum() <= 0:
            ok = False
            break
        A = X.T @ (w[:, None] * X)
        b = X.T @ (w * z)
        A[1, 1] += 2.0 * float(ridge_lambda)
        try:
            new = np.linalg.solve(A + 1e-8 * np.eye(2), b)
        except np.linalg.LinAlgError:
            ok = False
            break
        new[0] = np.clip(new[0], -cfg.logit_bound, cfg.logit_bound)
        new[1] = np.clip(new[1], -cfg.slope_bound, cfg.slope_bound)
        if np.max(np.abs(new - theta)) < 1e-7:
            theta = new
            break
        theta = new
    future_logit = float(theta[0] + theta[1] * float(horizon))
    pred = float(np.clip(expit(future_logit), EPS, 1.0))
    return pred, ok


def _lambda_tag(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def add_censored_candidates(tab: pd.DataFrame, cfg: CensoredConfig = CensoredConfig(),
                            lambda_grid: Sequence[float] = LAMBDA_GRID) -> pd.DataFrame:
    out = tab.copy()
    success = {float(l): [] for l in lambda_grid}
    preds = {float(l): [] for l in lambda_grid}
    for _, r in out.iterrows():
        for lam in lambda_grid:
            p, ok = fit_trajectory_forecast(r.train_frame, r.horizon, float(lam), cfg)
            preds[float(lam)].append(p)
            success[float(lam)].append(ok)
    for lam in lambda_grid:
        tag = _lambda_tag(lam)
        out[f"pred_censored_{tag}"] = preds[float(lam)]
        out[f"logpred_censored_{tag}"] = np.log(np.clip(out[f"pred_censored_{tag}"], EPS, 1.0))
        out[f"fit_ok_{tag}"] = success[float(lam)]
    return out


def _ale(logpred, true_vaf):
    return np.abs(np.asarray(logpred, float) - np.log(np.clip(np.asarray(true_vaf, float), EPS, 1.0)))


def _tune_alpha(train: pd.DataFrame, mask=None) -> float:
    d = train if mask is None else train[mask]
    if len(d) < 15:
        d = train
    best, loss = 0.0, np.inf
    for a in ALPHA_GRID:
        lp = d.last_lv + float(a) * d.s_ols * d.horizon
        cur = float(_ale(lp, d.true_vaf).mean())
        if cur < loss - 1e-12:
            best, loss = float(a), cur
    return best


def _tune_mu(train: pd.DataFrame) -> float:
    best, loss = 0.0, np.inf
    for mu in MU_GRID:
        lp = train.last_lv + float(mu) * train.horizon
        cur = float(_ale(lp, train.true_vaf).mean())
        if cur < loss - 1e-12:
            best, loss = float(mu), cur
    return best


def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                lambda_grid: Sequence[float] = LAMBDA_GRID) -> pd.DataFrame:
    P = pd.DataFrame(index=test.index)
    P["carry_forward"] = test.last_lv
    a = _tune_alpha(train)
    P["global_alpha"] = test.last_lv + a * test.s_ols * test.horizon
    mu = _tune_mu(train)
    P["drift_global"] = test.last_lv + mu * test.horizon

    lo = train.last_vaf < VAF_STRATUM_THRESHOLD
    a_lo = _tune_alpha(train, lo)
    a_hi = _tune_alpha(train, ~lo)
    ak = np.where(test.last_vaf < VAF_STRATUM_THRESHOLD, a_lo, a_hi)
    P["alpha_vaf_stratum"] = test.last_lv + ak * test.s_ols * test.horizon

    # v2.5 pre-specified fixed slope prior; with a singleton grid no tuning occurs.
    lam = float(lambda_grid[0])
    P["censored_growth"] = test[f"logpred_censored_{_lambda_tag(lam)}"].to_numpy(float)
    P.attrs = {"alpha": a, "mu": mu, "alpha_below_2pct": a_lo,
               "alpha_above_2pct": a_hi, "ridge_lambda": lam}
    return P


def crossfit_all(tab: pd.DataFrame, n_splits: int = 5, seed: int = 17,
                 lambda_grid: Sequence[float] = LAMBDA_GRID):
    pts = np.asarray(sorted(tab.patient_id.unique()))
    preds, params = [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(pts), 1):
        trm = tab.patient_id.isin(set(pts[tr]))
        tem = tab.patient_id.isin(set(pts[te]))
        P = fit_predict(tab[trm], tab[tem], lambda_grid=lambda_grid)
        P["fold"] = fold
        preds.append(P)
        params.append({"fold": fold, **P.attrs})
    return pd.concat(preds).loc[tab.index], params


def leave_one_cohort_out(tab: pd.DataFrame, lambda_grid: Sequence[float] = LAMBDA_GRID):
    out = []
    for c in tab.cohort.unique():
        tr, te = tab[tab.cohort != c], tab[tab.cohort == c]
        out.append(fit_predict(tr, te, lambda_grid=lambda_grid))
    return pd.concat(out).loc[tab.index]


def errors(P: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({m: _ale(P[m], tab.true_vaf) for m in MODELS}, index=tab.index)


def cluster_bootstrap_delta(err_a, err_b, groups, n_boot: int = 4000, seed: int = 0):
    d = (pd.Series(err_a, index=groups.index) - pd.Series(err_b, index=groups.index)).groupby(groups).agg(["sum", "count"])
    s, n = d["sum"].to_numpy(), d["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    point = float(s.sum() / n.sum())
    return point, float(np.quantile(boots, .025)), float(np.quantile(boots, .975)), float((boots < 0).mean())
