"""LeukemiPrediag v2.4 growth-shrinkage audit runner (outcome-free; research use only).

Usage (from project root):
    python notebooks/17_growth_shrinkage_audit.py \
        data/processed/v2/real_serial_filtered.csv data/processed/v2/growth_shrinkage_v24
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
_here = Path(__file__).resolve().parent
for _cand in (_here / "src", _here.parent / "src"):
    if (_cand / "growth_shrinkage.py").exists():
        sys.path.insert(0, str(_cand)); break
else:
    raise FileNotFoundError(f"src/growth_shrinkage.py not found near {_here}")
from growth_shrinkage import (build_trajectory_table, crossfit_all, leave_one_cohort_out,
                              errors, cluster_bootstrap_delta, MODELS)

serial_path, outdir = Path(sys.argv[1]), Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
serial = pd.read_csv(serial_path)
tab = build_trajectory_table(serial)
G = tab.patient_id
res = {"n_trajectories": len(tab), "n_patients": int(G.nunique())}

# 1. primary split (seed 17, identical to notebook 16)
P, params = crossfit_all(tab, seed=17)
E = errors(P, tab)
res["primary_seed17_mae_log"] = E.mean().round(5).to_dict()
res["primary_seed17_by_cohort"] = E.groupby(tab.cohort).mean().round(4).to_dict(orient="index")
res["primary_params"] = [{k: (v if not isinstance(v, dict) else {kk: round(float(vv), 4) for kk, vv in v.items()})
                          for k, v in p.items()} for p in params]

boot = {}
for ref in ["carry_forward", "global_alpha", "drift_global", "alpha_vaf_stratum"]:
    for m in MODELS:
        if m == ref:
            continue
        boot[f"{m}_minus_{ref}"] = cluster_bootstrap_delta(E[m], E[ref], G)
    for c in tab.cohort.unique():
        mk = tab.cohort == c
        for m in ["global_alpha", "drift_global", "alpha_vaf_stratum", "eb_cohort", "eb_gene"]:
            if m != ref:
                boot[f"{c}:{m}_minus_{ref}"] = cluster_bootstrap_delta(E[m][mk], E[ref][mk], G[mk])
res["bootstrap_delta_mean_lo_hi_pbetter"] = {k: [round(x, 4) for x in v] for k, v in boot.items()}

# 2. repeated CV (20 seeds) — split sensitivity
rep = []
for seed in range(20):
    Ps, _ = crossfit_all(tab, seed=seed)
    rep.append(errors(Ps, tab).mean())
rep = pd.DataFrame(rep)
res["repeated_cv_20seeds"] = {m: [round(rep[m].mean(), 5), round(rep[m].min(), 5), round(rep[m].max(), 5)] for m in rep}
res["repeated_cv_rank_wins"] = rep.idxmin(axis=1).value_counts().to_dict()

# 3. leave-one-cohort-out (transport)
L = errors(leave_one_cohort_out(tab), tab)
res["loco_by_cohort"] = L.groupby(tab.cohort).mean().round(4).to_dict(orient="index")
res["loco_overall"] = L.mean().round(5).to_dict()

# 3b. threshold sensitivity for the exploratory VAF-stratum model
thr = {}
for th in [0.012, 0.015, 0.02, 0.025, 0.03, 0.05]:
    Pt, _ = crossfit_all(tab, seed=17, vaf_threshold=th)
    e = errors(Pt, tab)["alpha_vaf_stratum"]
    thr[str(th)] = [round(float(e.mean()), 5)] + [round(x, 4) for x in cluster_bootstrap_delta(e, E["carry_forward"], G)]
res["vaf_threshold_sensitivity_mae_log_and_delta_vs_cf"] = thr

# 4. strata
res["by_kind"] = E.groupby(tab.kind.where(tab.kind.isin(["silent"]), "non_silent")).mean().round(4).to_dict(orient="index")
res["by_n_train"] = E.groupby(tab.n_train).mean().round(4).to_dict(orient="index")
res["signed_log_bias_cf_by_cohort"] = (tab.true_vaf.apply(np.log) - P.carry_forward).groupby(tab.cohort).mean().round(4).to_dict()

# 5. predictive intervals (80%) for carry_forward / drift / eb, split-conformal on training patients
from sklearn.model_selection import KFold
pts = np.asarray(sorted(tab.patient_id.unique()))
cov = {m: [] for m in ["carry_forward", "drift_global", "global_alpha", "alpha_vaf_stratum", "eb_cohort"]}
width = {m: [] for m in cov}
for tr, te in KFold(5, shuffle=True, random_state=17).split(pts):
    trm, tem = tab.patient_id.isin(set(pts[tr])), tab.patient_id.isin(set(pts[te]))
    # inner cross-fit on training patients to get honest residuals
    Pin, _ = crossfit_all(tab[trm].reset_index(drop=True), seed=1)
    trt = tab[trm].reset_index(drop=True)
    from growth_shrinkage import fit_predict
    Pte = fit_predict(tab[trm], tab[tem])
    for m in cov:
        r = np.abs(np.log(trt.true_vaf) - Pin[m]) / np.sqrt(trt.horizon)
        q = np.quantile(r, 0.80)
        half = q * np.sqrt(tab[tem].horizon)
        hit = np.abs(np.log(tab[tem].true_vaf) - Pte[m]) <= half
        cov[m].append(pd.Series(hit.values, index=tab[tem].index)); width[m].append(pd.Series(2 * half.values, index=tab[tem].index))
res["pi80_coverage_overall"] = {m: round(float(pd.concat(v).mean()), 3) for m, v in cov.items()}
res["pi80_coverage_by_cohort"] = {m: pd.concat(v).groupby(tab.cohort).mean().round(3).to_dict() for m, v in cov.items()}
res["pi80_mean_log_width"] = {m: round(float(pd.concat(v).mean()), 3) for m, v in width.items()}

out = pd.concat([tab[["patient_id", "cohort", "variant_id", "gene", "kind", "n_train", "horizon", "last_vaf", "true_vaf", "s_ols"]],
                 P[MODELS].add_prefix("logpred_"), E.add_prefix("abs_log_err_")], axis=1)
out.to_csv(outdir / "v24_crossfit_forecasts.csv", index=False)
json.dump(res, open(outdir / "v24_growth_shrinkage_audit.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
