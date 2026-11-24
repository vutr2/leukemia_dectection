"""Training-only cohort-aware split-conformal interval calibration for v2.4 forecasters.

This module does not alter point forecasts. It compares the original pooled
80% residual scale with cohort-specific residual quantiles learned only from
outer-training patients. Research use only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from growth_shrinkage import crossfit_all, fit_predict

MODELS = ["carry_forward", "global_alpha", "drift_global", "alpha_vaf_stratum"]


def evaluate_interval_calibration(tab: pd.DataFrame, *, n_splits: int = 5,
                                  seed: int = 17, target_coverage: float = 0.80,
                                  min_cohort_cal: int = 20):
    pts = np.asarray(sorted(tab.patient_id.unique()))
    rows = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(pts), 1):
        tr_ids, te_ids = set(pts[tr]), set(pts[te])
        trtab = tab[tab.patient_id.isin(tr_ids)].reset_index(drop=True)
        tetab = tab[tab.patient_id.isin(te_ids)].copy()

        inner_pred, _ = crossfit_all(trtab, seed=seed + 100 + fold)
        test_pred = fit_predict(trtab, tetab)

        for model in MODELS:
            r = np.abs(np.log(trtab.true_vaf.to_numpy(float)) - inner_pred[model].to_numpy(float))
            r = r / np.sqrt(trtab.horizon.to_numpy(float))
            cal = pd.DataFrame({"cohort": trtab.cohort.to_numpy(), "scaled_resid": r})
            q_pool = float(np.quantile(cal.scaled_resid, target_coverage))
            q_cohort = {}
            for cohort, g in cal.groupby("cohort"):
                q_cohort[cohort] = (
                    float(np.quantile(g.scaled_resid, target_coverage))
                    if len(g) >= min_cohort_cal else q_pool
                )

            truth = np.log(tetab.true_vaf.to_numpy(float))
            lp = test_pred[model].to_numpy(float)
            horizon_scale = np.sqrt(tetab.horizon.to_numpy(float))
            half_pool = q_pool * horizon_scale
            half_cohort = np.asarray([q_cohort.get(c, q_pool) for c in tetab.cohort]) * horizon_scale
            hp = np.abs(truth - lp) <= half_pool
            hc = np.abs(truth - lp) <= half_cohort

            for pos, (idx, cohort) in enumerate(zip(tetab.index, tetab.cohort)):
                rows.append({
                    "index": int(idx), "fold": fold, "cohort": cohort, "model": model,
                    "pooled_covered": int(hp[pos]), "cohort_covered": int(hc[pos]),
                    "pooled_log_width": float(2 * half_pool[pos]),
                    "cohort_log_width": float(2 * half_cohort[pos]),
                    "q_pool": q_pool, "q_cohort": float(q_cohort.get(cohort, q_pool)),
                })
    d = pd.DataFrame(rows)
    by = d.groupby(["model", "cohort"]).agg(
        pooled_coverage=("pooled_covered", "mean"),
        cohort_aware_coverage=("cohort_covered", "mean"),
        pooled_mean_log_width=("pooled_log_width", "mean"),
        cohort_aware_mean_log_width=("cohort_log_width", "mean"),
        n=("pooled_covered", "size"),
    ).reset_index()
    overall = d.groupby("model").agg(
        pooled_coverage=("pooled_covered", "mean"),
        cohort_aware_coverage=("cohort_covered", "mean"),
        pooled_mean_log_width=("pooled_log_width", "mean"),
        cohort_aware_mean_log_width=("cohort_log_width", "mean"),
        n=("pooled_covered", "size"),
    ).reset_index()
    return d, overall, by
