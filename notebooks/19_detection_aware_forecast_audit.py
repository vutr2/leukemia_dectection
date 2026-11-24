"""LeukemiPrediag v2.6 detection-aware / ascertainment-aware forecast audit.

Usage (from project root):
    python notebooks/19_detection_aware_forecast_audit.py \
        data/processed/v2/real_serial_unfiltered.csv \
        data/processed/v2/real_serial_filtered.csv \
        data/processed/v2/growth_censored_v25/v25_crossfit_forecasts.csv \
        data/processed/v2/forecast_scoring_v26
Research use only; outcome-free.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
_here = Path(__file__).resolve().parent
for _c in (_here / "src", _here.parent / "src"):
    if (_c / "forecast_scoring_v26.py").exists():
        sys.path.insert(0, str(_c)); break
from forecast_scoring_v26 import (build_table, crossfit, loco, score, cluster_bootstrap_delta,
                                  MODELS, ASCERTAIN_FLOOR)

unf, filt, v25, outdir = map(Path, sys.argv[1:5])
outdir.mkdir(parents=True, exist_ok=True)
serial = pd.read_csv(unf)
tab = build_table(serial)

# v2.5 censored predictions reused unchanged (per-trajectory fit, no cross-patient tuning)
f25 = pd.read_csv(v25)[["patient_id", "cohort", "variant_id", "logpred_censored_growth", "abs_log_err_carry_forward"]]
f25["patient_id"] = f25.patient_id.astype(str)
tab = tab.merge(f25, on=["patient_id", "cohort", "variant_id"], how="left").rename(
    columns={"logpred_censored_growth": "lp_censored"})

# trajectories already used to develop the 2% rule in v2.4
fs = pd.read_csv(filt)
cnt = fs.groupby(["patient_id", "cohort", "variant_id"]).time_years.nunique()
dev = set(cnt[cnt >= 3].index)
tab["in_v24_dev"] = [(p, c, v) in dev for p, c, v in zip(tab.patient_id, tab.cohort, tab.variant_id)]

res = {"n_trajectories": int(len(tab)), "n_patients": int(tab.patient_id.nunique()),
       "n_with_v25_censored_pred": int(tab.lp_censored.notna().sum())}

# --- diagnostics of the v2.5 metric and of ascertainment leakage
zero = (tab.true_vaf <= 0) | (tab.last_vaf <= 0)
e25 = tab.abs_log_err_carry_forward
res["v25_metric_share_of_cf_error_from_zero_rows"] = round(float(e25[zero].sum() / e25.sum()), 3)
res["frac_trajectories_with_zero_last_or_true"] = round(float(zero.mean()), 3)
res["frac_rows_below_row_detection_limit_by_cohort"] = (
    (serial.vaf < pd.to_numeric(serial.detection_limit, errors="coerce")).groupby(serial.cohort).mean().round(3).to_dict())
np_ = ~tab.prospective
res["ascertainment"] = {
    "n_prospective": int(tab.prospective.sum()), "n_not_prospective": int(np_.sum()),
    "not_prospective_frac_test_ge_1pct": round(float((tab.true_vaf[np_] >= ASCERTAIN_FLOOR).mean()), 3),
    "prospective_frac_test_ge_1pct": round(float((tab.true_vaf[tab.prospective] >= ASCERTAIN_FLOOR).mean()), 3),
    "by_cohort_n_prospective": tab[tab.prospective].groupby("cohort").size().to_dict(),
}

def run_set(name, mask, extra_subsets=()):
    T = tab[mask].copy()
    T = T[T.lp_censored.notna()] if "censored" in name else T
    P, pars = crossfit(T, seed=17)
    E, V, B = score(P, T)
    G = T.patient_id
    r = {"n": int(len(T)), "n_patients": int(G.nunique()),
         "fold_params": [{k: round(float(v), 4) for k, v in p.items()} for p in pars],
         "det_err": E.mean().round(5).to_dict(), "mae_vaf": V.mean().round(5).to_dict(),
         "binom_nll_lbc": B.mean().round(4).to_dict(), "n_binom": int(len(B)),
         "det_err_by_cohort": E.groupby(T.cohort).mean().round(4).to_dict(orient="index")}
    boot = {}
    for m in MODELS:
        for ref in ["carry_forward", "drift_global"]:
            if m != ref and m in E:
                boot[f"{m}_minus_{ref}"] = cluster_bootstrap_delta(E[m], E[ref], G)
    for c in T.cohort.unique():
        k = T.cohort == c
        boot[f"{c}:alpha_vaf_stratum_minus_carry_forward"] = cluster_bootstrap_delta(
            E.alpha_vaf_stratum[k], E.carry_forward[k], G[k])
        boot[f"{c}:censored_growth_minus_carry_forward"] = cluster_bootstrap_delta(
            E.censored_growth[k], E.carry_forward[k], G[k])
    r["bootstrap_det_err"] = boot
    for sname, smask in extra_subsets:
        k = smask.loc[T.index]
        r[f"subset:{sname}"] = {"n": int(k.sum()), "det_err": E[k].mean().round(5).to_dict(),
            "alpha_vaf_stratum_minus_carry_forward": cluster_bootstrap_delta(E.alpha_vaf_stratum[k], E.carry_forward[k], G[k]),
            "alpha_vaf_stratum_minus_drift_global": cluster_bootstrap_delta(E.alpha_vaf_stratum[k], E.drift_global[k], G[k]),
            "censored_growth_minus_carry_forward": cluster_bootstrap_delta(E.censored_growth[k], E.carry_forward[k], G[k])}
    rep = pd.DataFrame([score(crossfit(T, seed=s)[0], T)[0].mean() for s in range(20)])
    r["repeated_cv_20seeds_det_err"] = {m: [round(rep[m].mean(), 5), round(rep[m].min(), 5), round(rep[m].max(), 5)] for m in rep}
    r["repeated_cv_wins"] = rep.idxmin(axis=1).value_counts().to_dict()
    L = score(loco(T), T)[0]
    r["loco_det_err"] = L.mean().round(5).to_dict()
    r["loco_by_cohort"] = L.groupby(T.cohort).mean().round(4).to_dict(orient="index")
    out = pd.concat([T.drop(columns=["abs_log_err_carry_forward"]), P[[m for m in MODELS if m in P]].add_prefix("lp_"),
                     E.add_prefix("det_err_")], axis=1)
    out.to_csv(outdir / f"v26_{name}_forecasts.csv", index=False)
    return r

new_mask = ~tab.in_v24_dev
res["ALL_censored"] = run_set("ALL_censored", pd.Series(True, index=tab.index))
res["PRIMARY_prospective_censored"] = run_set(
    "PRIMARY_prospective_censored", tab.prospective,
    extra_subsets=[("not_in_v24_dev", new_mask), ("in_v24_dev", tab.in_v24_dev)])
json.dump(res, open(outdir / "v26_forecast_scoring_audit.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
