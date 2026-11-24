"""LeukemiPrediag v2.6 — detection-aware scoring and ascertainment-aware evaluation.

Outcome-free. Re-evaluates v2.4/v2.5 forecasters on data/processed/v2/real_serial_unfiltered.csv
with two corrections:

1. Detection-aware error. The v2.5 metric |log(pred) - log(true)| clips zeros to 1e-5,
   so a held-out (or last) VAF of exactly 0 contributes an error set by the eps
   convention, not by biology. v2.6 floors truth AND forecast at the held-out
   observation's detection limit L (LBC: row X95MDAF, cohort median if missing;
   SardiNIA: operational 0.2%):
       err = | log max(pred, L) - log max(true, L) |
   Two values that are both below L are treated as indistinguishable (err = 0).
   Secondary metrics: MAE on the VAF scale; plug-in binomial log-likelihood of the
   held-out alt-read count (LBC rows with AO/DP only).

2. Prospective ascertainment. Variants enter the source tables because they reached
   ~1% at SOME wave (LBC files are 1PCT_VAF universes; SardiNIA re-queries a site at
   all timepoints once it is found at any one). If that wave is the held-out one,
   the trajectory's inclusion leaks future growth. The primary analysis set keeps
   only trajectories whose TRAINING window already contains VAF >= 1%, i.e. clones
   a user would actually know about at forecast time.

Models (log space, anchored at max(last VAF, L_last)):
    carry_forward, global_alpha, drift_global, alpha_vaf_stratum (2% threshold FROZEN
    from v2.4), censored_growth (v2.5 predictions, reused unchanged).
Slopes for alpha models use detection-floored log-VAF. All tuning on training
patients only, with the same detection-aware loss used for evaluation.

Research use only. Not a diagnostic device.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.model_selection import KFold

VAF_STRATUM_THRESHOLD = 0.02          # frozen from v2.4; NOT retuned
ASCERTAIN_FLOOR = 0.01
SARDINIA_LIMIT = 0.002
ALPHA_GRID = np.linspace(0, 1, 21)
MU_GRID = np.linspace(-0.10, 0.20, 61)
MODELS = ["carry_forward", "global_alpha", "drift_global", "alpha_vaf_stratum", "censored_growth"]


def _row_limits(s: pd.DataFrame) -> pd.Series:
    lim = pd.to_numeric(s["detection_limit"], errors="coerce")
    med = lim.groupby(s["cohort"]).transform("median")
    lim = lim.fillna(med).fillna(SARDINIA_LIMIT)
    return lim.clip(lower=1e-4, upper=0.5)


def build_table(serial: pd.DataFrame, min_total: int = 3) -> pd.DataFrame:
    s = serial.copy()
    s["L"] = _row_limits(s)
    rows = []
    for (pid, coh, vid), g in s.groupby(["patient_id", "cohort", "variant_id"], sort=False):
        z = (g.groupby("time_years", as_index=False)
              .agg(vaf=("vaf", "mean"), L=("L", "median"), alt=("alt_reads", "mean"), dp=("depth", "mean"))
              .sort_values("time_years"))
        if len(z) < min_total:
            continue
        tr, te = z.iloc[:-1], z.iloc[-1]
        t = tr.time_years.to_numpy(float)
        if np.ptp(t) <= 0 or te.time_years <= t.max():
            continue
        v = np.clip(tr.vaf.to_numpy(float), 0, 1)
        Lr = tr.L.to_numpy(float)
        lvf = np.log(np.maximum(v, Lr))                 # detection-floored log VAF
        slope = float(np.polyfit(t, lvf, 1)[0])
        rows.append(dict(
            patient_id=str(pid), cohort=coh, variant_id=vid,
            n_train=len(tr), horizon=float(te.time_years - t.max()),
            last_vaf=float(v[-1]), L_last=float(Lr[-1]), anchor_lv=float(lvf[-1]),
            train_max_vaf=float(v.max()), s_floor=slope,
            true_vaf=float(np.clip(te.vaf, 0, 1)), L_test=float(te.L),
            test_alt=float(te.alt) if pd.notna(te.alt) else np.nan,
            test_dp=float(te.dp) if pd.notna(te.dp) else np.nan,
        ))
    tab = pd.DataFrame(rows)
    tab["prospective"] = tab.train_max_vaf >= ASCERTAIN_FLOOR
    return tab


def det_err(lpred, tab):
    L = np.log(tab.L_test.to_numpy(float))
    lt = np.log(np.maximum(tab.true_vaf.to_numpy(float), tab.L_test.to_numpy(float)))
    return np.abs(np.maximum(np.asarray(lpred, float), L) - lt)


def _tune(train, gfun, grid):
    best, bl = grid[0], np.inf
    for x in grid:
        loss = det_err(train.anchor_lv + gfun(train, x) * train.horizon, train).mean()
        if loss < bl - 1e-12:
            best, bl = x, loss
    return best


def fit_predict(train, test):
    P = pd.DataFrame(index=test.index)
    P["carry_forward"] = test.anchor_lv
    a = _tune(train, lambda d, x: x * d.s_floor, ALPHA_GRID)
    P["global_alpha"] = test.anchor_lv + a * test.s_floor * test.horizon
    mu = _tune(train, lambda d, x: x, MU_GRID)
    P["drift_global"] = test.anchor_lv + mu * test.horizon
    lo = train.last_vaf < VAF_STRATUM_THRESHOLD
    a_lo = _tune(train[lo], lambda d, x: x * d.s_floor, ALPHA_GRID) if lo.sum() >= 15 else a
    a_hi = _tune(train[~lo], lambda d, x: x * d.s_floor, ALPHA_GRID) if (~lo).sum() >= 15 else a
    ak = np.where(test.last_vaf < VAF_STRATUM_THRESHOLD, a_lo, a_hi)
    P["alpha_vaf_stratum"] = test.anchor_lv + ak * test.s_floor * test.horizon
    if "lp_censored" in test:
        P["censored_growth"] = test.lp_censored
    P.attrs = dict(alpha=a, mu=mu, alpha_below_2pct=a_lo, alpha_above_2pct=a_hi)
    return P


def crossfit(tab, seed=17, n_splits=5):
    pts = np.asarray(sorted(tab.patient_id.unique()))
    out, pars = [], []
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(pts):
        a, b = tab.patient_id.isin(set(pts[tr])), tab.patient_id.isin(set(pts[te]))
        P = fit_predict(tab[a], tab[b]); out.append(P); pars.append(P.attrs)
    return pd.concat(out).loc[tab.index], pars


def loco(tab):
    return pd.concat([fit_predict(tab[tab.cohort != c], tab[tab.cohort == c])
                      for c in tab.cohort.unique()]).loc[tab.index]


def score(P, tab):
    E = pd.DataFrame({m: det_err(P[m], tab) for m in MODELS if m in P}, index=tab.index)
    V = pd.DataFrame({m: np.abs(np.exp(P[m]) - tab.true_vaf) for m in MODELS if m in P}, index=tab.index)
    ok = tab.test_dp.notna() & tab.test_alt.notna() & (tab.test_dp > 0)
    B = pd.DataFrame({m: -binom.logpmf(tab.test_alt[ok].round(), tab.test_dp[ok].round(),
                                       np.clip(np.exp(P[m][ok]), 1e-4, 0.9999))
                      for m in MODELS if m in P}, index=tab.index[ok])
    return E, V, B


def cluster_bootstrap_delta(a, b, groups, n_boot=4000, seed=0):
    d = (a - b).groupby(groups).agg(["sum", "count"])
    s, n = d["sum"].to_numpy(), d["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return [round(float(s.sum() / n.sum()), 5), round(float(np.quantile(boots, .025)), 5),
            round(float(np.quantile(boots, .975)), 5), round(float((boots < 0).mean()), 4)]
