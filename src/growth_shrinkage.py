"""LeukemiPrediag v2.4 — hierarchical / empirical-Bayes growth-shrinkage audit.

Outcome-free. Extends the v2.3 leave-last-timepoint forecast (src/temporal_validation.py)
with simpler AND more complex competitors, all evaluated under patient-held-out
cross-fitting. Every learned quantity (alpha, drift, tau^2, extra noise, gene means,
interval quantiles) is fitted on training patients only.

Forecast form for every model (log space, anchored at the last observed VAF):
    log VAF_hat(t_last + d) = log VAF_last + g_hat * d
Models differ only in g_hat:
    carry_forward     g = 0
    global_alpha      g = alpha * s_ols                       (v2.3 notebook 16)
    drift_global      g = mu                                  (population drift, no individual slope)
    drift_cohort      g = mu_cohort
    alpha_cohort      g = alpha_cohort * s_ols
    alpha_vaf_stratum g = alpha_k * s_ols, k = [last VAF < 2%] vs [>= 2%]   (detection-floor aware)
    eb_cohort         g = mu_c + w_i (s_wls - mu_c),  w_i = tau2 / (tau2 + v_i)
    eb_gene           as eb_cohort but prior mean = partially pooled gene mean within cohort
v_i is the sampling variance of the trajectory's training slope from a binomial
(depth-aware) + extra-binomial log-VAF noise model; tau2 and the extra noise are
method-of-moments estimates on training patients.

Research use only. Not a diagnostic device. Says nothing about leukemia outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

EPS = 1e-5
MODELS = ["carry_forward", "global_alpha", "drift_global", "drift_cohort",
          "alpha_cohort", "alpha_vaf_stratum", "eb_cohort", "eb_gene"]
# 2% = conventional CHIP VAF threshold and 2x the 1% floor of real_serial_filtered.csv.
# Chosen after inspecting v2.3 errors -> EXPLORATORY until confirmed on an independent cohort.
VAF_STRATUM_THRESHOLD = 0.02


# --------------------------------------------------------------------------- table
def build_trajectory_table(serial: pd.DataFrame, min_total: int = 3) -> pd.DataFrame:
    """One row per trajectory; training window = all but last timepoint.

    Eligibility mirrors temporal_validation.leave_last_timepoint_forecasts.
    """
    rows = []
    agg = {"vaf": "mean", "depth": "mean"}
    for (pid, coh, vid), g in serial.groupby(["patient_id", "cohort", "variant_id"], sort=False):
        z = g.dropna(subset=["time_years", "vaf"]).groupby("time_years", as_index=False).agg(agg)
        z = z.sort_values("time_years")
        if len(z) < min_total:
            continue
        tr, te = z.iloc[:-1], z.iloc[-1]
        t = tr.time_years.to_numpy(float)
        tt = float(te.time_years)
        if np.ptp(t) <= 0 or tt <= t.max():
            continue
        v = np.clip(tr.vaf.to_numpy(float), EPS, 1.0)
        dp = np.clip(tr.depth.fillna(tr.depth.median()).to_numpy(float), 50, None)
        lv = np.log(v)
        b_ols, a_ols = np.polyfit(t, lv, 1)
        binvar = (1 - v) / (dp * v)              # delta-method var of log VAF
        rows.append(dict(
            patient_id=str(pid), cohort=coh, variant_id=vid,
            gene=g.gene.iloc[0], kind=g.kind.iloc[0],
            n_train=len(tr), horizon=tt - t.max(),
            t_train=t, lv_train=lv, binvar_train=binvar,
            last_lv=lv[-1], last_vaf=v[-1], true_vaf=float(np.clip(te.vaf, EPS, 1.0)),
            s_ols=float(b_ols), resid_ss=float(np.sum((lv - (a_ols + b_ols * t)) ** 2)),
        ))
    return pd.DataFrame(rows)


def _wls_slope(t, y, w):
    W = w.sum()
    tb, yb = (w * t).sum() / W, (w * y).sum() / W
    sxx = (w * (t - tb) ** 2).sum()
    return (w * (t - tb) * (y - yb)).sum() / sxx, 1.0 / sxx


def add_noise_model(tab: pd.DataFrame, sigma2_extra: float) -> pd.DataFrame:
    s, v = [], []
    for t, y, bv in zip(tab.t_train, tab.lv_train, tab.binvar_train):
        w = 1.0 / (bv + sigma2_extra)
        b, var = _wls_slope(t, y, w)
        s.append(b); v.append(var)
    out = tab.copy()
    out["s_wls"], out["v_slope"] = s, v
    return out


def estimate_extra_noise(train: pd.DataFrame) -> float:
    """MoM extra-binomial log-VAF variance from residuals of >=3-point training fits."""
    m = train[train.n_train >= 3]
    if m.empty:
        return 0.02
    df = (m.n_train - 2).sum()
    resid_var = m.resid_ss.sum() / df
    mean_bin = np.mean(np.concatenate(m.binvar_train.to_numpy()))
    return float(max(resid_var - mean_bin, 1e-3))


# --------------------------------------------------------------------- utilities
def _ale(lpred, true):
    return np.abs(lpred - np.log(true))


def _tune_scalar(train, gfun, grid):
    best, bl = grid[0], np.inf
    for x in grid:
        loss = _ale(train.last_lv + gfun(train, x) * train.horizon, train.true_vaf).mean()
        if loss < bl - 1e-12:
            best, bl = x, loss
    return best


ALPHA_GRID = np.linspace(0, 1, 21)
MU_GRID = np.linspace(-0.10, 0.20, 61)


# ------------------------------------------------------------------------- fit
def fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                vaf_threshold: float = VAF_STRATUM_THRESHOLD) -> pd.DataFrame:
    """Fit every model on training trajectories; return log-forecasts for test."""
    s2 = estimate_extra_noise(train)
    train, test = add_noise_model(train, s2), add_noise_model(test, s2)
    P = pd.DataFrame(index=test.index)
    P["carry_forward"] = test.last_lv

    a = _tune_scalar(train, lambda d, x: x * d.s_ols, ALPHA_GRID)
    P["global_alpha"] = test.last_lv + a * test.s_ols * test.horizon

    mu = _tune_scalar(train, lambda d, x: x, MU_GRID)
    P["drift_global"] = test.last_lv + mu * test.horizon

    mu_c, a_c = {}, {}
    for c, d in train.groupby("cohort"):
        mu_c[c] = _tune_scalar(d, lambda d, x: x, MU_GRID) if len(d) >= 15 else mu
        a_c[c] = _tune_scalar(d, lambda d, x: x * d.s_ols, ALPHA_GRID) if len(d) >= 15 else a
    mc = test.cohort.map(mu_c).fillna(mu)
    ac = test.cohort.map(a_c).fillna(a)
    P["drift_cohort"] = test.last_lv + mc * test.horizon
    P["alpha_cohort"] = test.last_lv + ac * test.s_ols * test.horizon

    lo_tr = train.last_vaf < vaf_threshold
    a_lo = _tune_scalar(train[lo_tr], lambda d, x: x * d.s_ols, ALPHA_GRID) if lo_tr.sum() >= 15 else a
    a_hi = _tune_scalar(train[~lo_tr], lambda d, x: x * d.s_ols, ALPHA_GRID) if (~lo_tr).sum() >= 15 else a
    a_k = np.where(test.last_vaf < vaf_threshold, a_lo, a_hi)
    P["alpha_vaf_stratum"] = test.last_lv + a_k * test.s_ols * test.horizon

    # EB: prior N(m_c, tau2_c) on true slope; MoM on training slopes (precision-weighted mean)
    eb_m, eb_t = {}, {}
    for c, d in train.groupby("cohort"):
        w = 1 / d.v_slope
        m = float((w * d.s_wls).sum() / w.sum())
        tau2 = float(max(np.mean((d.s_wls - m) ** 2 - d.v_slope), 1e-6))
        eb_m[c], eb_t[c] = m, tau2
    gm = float(np.mean(list(eb_m.values())))
    gt = float(np.mean(list(eb_t.values())))
    m_i = test.cohort.map(eb_m).fillna(gm)
    t_i = test.cohort.map(eb_t).fillna(gt)
    w_i = t_i / (t_i + test.v_slope)
    P["eb_cohort"] = test.last_lv + (m_i + w_i * (test.s_wls - m_i)) * test.horizon

    # gene mean within cohort, partially pooled toward cohort mean (second level)
    gene_post = {}
    for (c, gname), d in train.groupby(["cohort", "gene"]):
        m_c, tau2_c = eb_m[c], eb_t[c]
        # gene-level variance of the mean of member slopes around cohort mean
        var_mean = (tau2_c + d.v_slope.mean()) / len(d)
        # between-gene variance: crude MoM over genes in cohort computed below
        gene_post[(c, gname)] = (d.s_wls.mean(), var_mean)
    tau2_gene = {}
    for c in eb_m:
        items = [(gm_, vm) for (cc, _), (gm_, vm) in gene_post.items() if cc == c]
        if len(items) >= 3:
            arr = np.array(items)
            tau2_gene[c] = float(max(np.mean((arr[:, 0] - eb_m[c]) ** 2 - arr[:, 1]), 0.0))
        else:
            tau2_gene[c] = 0.0
    prior = []
    for c, gname in zip(test.cohort, test.gene):
        base = eb_m.get(c, gm)
        if (c, gname) in gene_post and tau2_gene.get(c, 0) > 0:
            gmean, vm = gene_post[(c, gname)]
            k = tau2_gene[c] / (tau2_gene[c] + vm)
            prior.append(base + k * (gmean - base))
        else:
            prior.append(base)
    prior = pd.Series(prior, index=test.index)
    # residual within-gene slope variance ~ tau2_c - tau2_gene (floored)
    t_w = (t_i - test.cohort.map(tau2_gene).fillna(0)).clip(lower=1e-6)
    w_g = t_w / (t_w + test.v_slope)
    P["eb_gene"] = test.last_lv + (prior + w_g * (test.s_wls - prior)) * test.horizon

    P.attrs = dict(alpha=a, alpha_below_thr=a_lo, alpha_above_thr=a_hi, mu=mu, sigma2_extra=s2, mu_cohort=mu_c, alpha_cohort=a_c,
                   eb_mean=eb_m, eb_tau2=eb_t, tau2_gene=tau2_gene)
    return P


def crossfit_all(tab: pd.DataFrame, n_splits=5, seed=17, **kw):
    pts = np.asarray(sorted(tab.patient_id.unique()))
    preds, params = [], []
    for fold, (tr, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(pts), 1):
        trm, tem = tab.patient_id.isin(set(pts[tr])), tab.patient_id.isin(set(pts[te]))
        P = fit_predict(tab[trm], tab[tem], **kw)
        P["fold"] = fold
        preds.append(P); params.append(dict(fold=fold, **{k: v for k, v in P.attrs.items()}))
    P = pd.concat(preds).loc[tab.index]
    return P, params


def leave_one_cohort_out(tab: pd.DataFrame):
    out = []
    for c in tab.cohort.unique():
        tr, te = tab[tab.cohort != c], tab[tab.cohort == c].copy()
        # cohort-specific models fall back to pooled values for unseen cohort
        P = fit_predict(tr, te)
        out.append(P)
    return pd.concat(out).loc[tab.index]


def errors(P: pd.DataFrame, tab: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({m: _ale(P[m], tab.true_vaf) for m in MODELS if m in P}, index=tab.index)


def cluster_bootstrap_delta(err_a, err_b, groups, n_boot=4000, seed=0):
    """Patient-cluster bootstrap of mean(err_a - err_b). Negative = a better."""
    d = (err_a - err_b).groupby(groups).agg(["sum", "count"])
    s, n = d["sum"].to_numpy(), d["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(boots, .025)), float(np.quantile(boots, .975)), float((boots < 0).mean())
