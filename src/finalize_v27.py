"""Rebuild the v2.7 descriptive statistics from immutable patient feature tables.

No tuning and no model fitting: fixes top-quintile ambiguity and adds patient
bootstrap CIs without rerunning costly frozen clone inference.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from feature_robustness_v27 import comparisons,within_stability,policy
from model_spec import CORE_DYNAMICS_V23


def bootstrap_rho(a,b,seed=17,boots=1000):
    z=pd.DataFrame({'a':pd.to_numeric(a,errors='coerce'),'b':pd.to_numeric(b,errors='coerce')}).dropna()
    if len(z)<20 or z.a.nunique()<2 or z.b.nunique()<2:
        return None,None
    arr=z[['a','b']].to_numpy(float)
    rng=np.random.default_rng(seed)
    stats=[]
    for k in range(boots):
        sample=arr[rng.integers(0,len(arr),size=len(arr))]
        if len(np.unique(sample[:,0]))>=2 and len(np.unique(sample[:,1]))>=2:
            rho=float(spearmanr(sample[:,0],sample[:,1]).statistic)
            if np.isfinite(rho):stats.append(rho)
    if len(stats)<int(boots*.8):
        return None,None
    return (float(np.quantile(stats,.025)),float(np.quantile(stats,.975)))


def finalize(outdir):
    d=Path(outdir)
    manifest=json.loads((d/'v27_manifest.json').read_text())
    for axis in ['original','approx_age']:
        features={};prior={}
        for v in ['V0','V1_RELIABLE','V2_PREINDEX_RELIABLE']:
            features[v]=pd.read_csv(d/f'{axis}_{v}_patient_features.csv')
            prior[v]=pd.read_csv(d/f'{axis}_{v}_prior_patient_features.csv')
        comp,_=comparisons(features)
        stable=within_stability(features,prior)
        policy_tab=policy(comp,stable,tuple(sorted(features['V0'].cohort.unique())))
        ci=[]
        for cohort in sorted(features['V0'].cohort.unique()):
            x=features['V0'].merge(features['V2_PREINDEX_RELIABLE'],on=['patient_id','cohort'],suffixes=('_v0','_v2'))
            x=x[x.cohort==cohort]
            for ft in CORE_DYNAMICS_V23:
                lo,hi=bootstrap_rho(x[ft+'_v0'],x[ft+'_v2'])
                ci.append(dict(cohort=cohort,feature=ft,n_total=len(x),ci_95_low=lo,ci_95_high=hi))
        ci=pd.DataFrame(ci)
        ci.to_csv(d/f'{axis}_spearman_bootstrap_ci.csv',index=False)
        comp.to_csv(d/f'{axis}_spearman_and_quintile.csv',index=False)
        stable.to_csv(d/f'{axis}_within_patient.csv',index=False)
        policy_tab.to_csv(d/f'{axis}_feature_status.csv',index=False)
        manifest['runs'][axis]['classification']=policy_tab.set_index('feature').classification.to_dict()
        manifest['runs'][axis]['top_quintile_ties_censored']=int(comp.top_quintile_tie_ambiguous.sum())
    (d/'v27_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    import sys
    finalize(sys.argv[1])
