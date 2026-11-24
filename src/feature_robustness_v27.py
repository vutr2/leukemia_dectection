"""LeukemiPrediag v2.7: assay-aware measurement robustness, outcome-free.

This is a VERSIONED wrapper. Existing v2.1 clone_inference and v2.2
progression_features are imported unchanged. Point forecasting is FROZEN.

An LBC observation below its X95MDAF is *not* used as quantitative evidence
for growth, decline, correlation, or co-clonal structure. Records with unknown
limits are not assigned invented values. Three reliably detected timepoints are
required for the existing (non-censoring-aware) clone inference implementation.

The full LoD-floored table is retained strictly for data/provenance sensitivity
work, never fitted by an ordinary log-VAF regression: this avoids artificial
same-floor correlations and computational blowup from 153k selected pairs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from clone_inference import CloneInferenceConfig, run_clone_inference
from progression_features import ProgressionFeatureConfig, patient_progression_features
from model_spec import CORE_DYNAMICS_V23

VERSION = "2.7"
KEY = ["patient_id", "cohort", "variant_id"]
PK = ["patient_id", "cohort"]
NUMERIC_LOD_LOWER = 0.0
# Cohort-level mean ages from publicly reported LBC1921 study waves;
# APPROXIMATE, never patient ages, and not a validated individual time axis.
# Map labels actually present in the source table: 1, 3, 4, 5.
APPROX_LBC1921_AGE = {1: 79.1, 3: 86.6, 4: 90.1, 5: 92.1}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def correct_time_axis(serial: pd.DataFrame, use_approximate: bool) -> pd.DataFrame:
    """Versioned mapping: never overwrite the original time_years."""
    d = serial.copy()
    d['time_years_v26_original'] = d['time_years']
    if not use_approximate:
        return d
    m = d.cohort.eq('GSE288742')
    if not m.any():
        return d
    ages = pd.to_numeric(d.loc[m, 'wave'], errors='coerce').map(APPROX_LBC1921_AGE)
    if ages.isna().any():
        raise ValueError(f"Unmapped GSE288742 wave(s): {sorted(d.loc[m & d.wave.map(APPROX_LBC1921_AGE).isna(), 'wave'].unique())}")
    first_age = ages.groupby(d.loc[m, 'patient_id']).transform('min')
    d.loc[m, 'time_years'] = (ages - first_age).to_numpy(dtype=float)
    return d


def validate_inputs(filt: pd.DataFrame, raw: pd.DataFrame):
    for name, d in [('filtered', filt), ('raw', raw)]:
        req = set(KEY + ['time_years', 'wave', 'vaf', 'gene', 'depth'])
        missing = req - set(d.columns)
        if missing:
            raise ValueError(f"{name} missing columns {sorted(missing)}")
        if d.duplicated(KEY+['time_years']).any():
            raise ValueError(f"{name} has duplicate patient-variant-time rows")
        if not d.vaf.between(0, 1).all():
            raise ValueError(f"{name} has VAF out of range")
    req = ['detection_limit', 'detection_limit_source']
    missing = set(req) - set(raw.columns)
    if missing:
        raise ValueError(f"raw lacks {sorted(missing)}")
    for d in [filt, raw]:
        if d.groupby('patient_id').cohort.nunique().max() > 1:
            raise ValueError('Patient ID collision across cohorts')


def prepare_versions(filtered: pd.DataFrame, raw: pd.DataFrame, *, approx_time: bool = True,
                     min_reliable_tp: int = 3) -> Dict[str, pd.DataFrame]:
    """Distinct raw/floored and inference-ready datasets, never silently impute LoD.

    V0 is the historical 1%-selected dataset; V1 raw is LoD-floored per *known*
    row limit; the clone-inference dataset uses only quantifiable rows. V2 keeps
    variants reaching 1% STRICTLY before the patient index wave (using raw
    observed values), before applying the reliability gate. Version-specific
    QC sizes and attrition must be reported. No future information in V2 key.
    """
    validate_inputs(filtered, raw)
    d0 = correct_time_axis(filtered, approx_time)
    # Truncated inputs can already contain prior stage's QC annotations.
    # Always rebuild the flags from raw VAF/time rather than inheriting a
    # future-index eligibility decision or creating _x/_y merge suffixes.
    stale = ['lod_known','depth_qc','quantifiable','vaf_lod_floored','assay_censored',
             'known_pre_index','time_years_v26_original']
    d1 = correct_time_axis(raw.drop(columns=stale, errors='ignore'), approx_time)
    d1['detection_limit'] = pd.to_numeric(d1.detection_limit, errors='coerce')
    valid_limit = d1.detection_limit.notna() & d1.detection_limit.between(1e-7, 0.5)
    d1['lod_known'] = valid_limit
    depth_qc = (~d1.cohort.isin(['GSE178936','GSE288742'])) | (d1.depth >= 500)
    d1['depth_qc'] = depth_qc
    d1['quantifiable'] = valid_limit & depth_qc & (d1.vaf >= d1.detection_limit)
    d1['vaf_lod_floored'] = np.where(valid_limit,
                                    np.maximum(d1.vaf, d1.detection_limit), np.nan)
    d1['assay_censored'] = valid_limit & (d1.vaf < d1.detection_limit)
    # The patient index is the last sampled wave, not each variant's terminal wave.
    last_t = d1.groupby('patient_id').time_years.transform('max')
    # Prospectively *detected* means satisfying the actual row LoD and depth
    # QC, not just a numeric value >= the source selection 1% threshold.
    prior_variant = d1.loc[(d1.time_years < last_t) & (d1.vaf >= 0.01)
                           & d1.quantifiable, KEY].drop_duplicates()
    prior_variant['known_pre_index'] = True
    d1 = d1.merge(prior_variant, on=KEY, how='left', validate='many_to_one')
    d1['known_pre_index'] = d1['known_pre_index'].eq(True)

    # *Only* quantifiable values may be passed to the unmodified log-VAF clone
    # inference. LoD floors are upper constraints, not quantitative observations.
    high = d1[d1.quantifiable].copy()
    enough = high.groupby(KEY).time_years.transform('nunique') >= min_reliable_tp
    reliable = high.loc[enough].copy()
    prospective = reliable.loc[reliable.known_pre_index].copy()

    # same names/semantic data contract expected by existing v2 algorithms.
    result = {
        'V0': d0,
        'V1_LOD_FLOORED_AUDIT_ONLY': d1,
        'V1_RELIABLE': reliable,
        'V2_PREINDEX_RELIABLE': prospective,
    }
    return result


def _valid_empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=['patient_id', 'cohort', 'variant_a', 'variant_b', 'support', 'candidate_same_clone'])


def compute_features(serial: pd.DataFrame, draws: int = 500, seed: int = 17) -> pd.DataFrame:
    """Call frozen clone/progression algorithms unchanged, preserving nulls."""
    if serial.empty:
        return pd.DataFrame(columns=PK+list(CORE_DYNAMICS_V23))
    cfg = CloneInferenceConfig(n_draws=draws, seed=seed)
    c = run_clone_inference(serial, cfg)
    pairs = c['pair_support']
    if pairs.empty:
        pairs = _valid_empty_pairs()
    f, _, _ = patient_progression_features(
        serial,
        trajectory_posteriors=c['trajectory_posteriors'],
        pair_support=pairs,
        assignments=c['assignments'],
        clone_posteriors=c['clone_posteriors'],
        patient_clone_features=c['patient_features'],
        cfg=ProgressionFeatureConfig(),
    )
    for col in CORE_DYNAMICS_V23:
        if col not in f:
            f[col] = np.nan
    # IMPORTANT: pair ratios with no evaluated pair are undefined, not zero.
    no_pairs = f['n_variant_pairs_evaluated'].fillna(0).eq(0) if 'n_variant_pairs_evaluated' in f else pd.Series(True,index=f.index)
    f.loc[no_pairs,'fraction_evaluated_pairs_supported'] = np.nan
    # Clone count does not mean zero merely because clone posterior is unavailable.
    return f[PK+list(CORE_DYNAMICS_V23)].copy()


def _rho(a: pd.Series, b: pd.Series) -> tuple[float | None,int,str]:
    y = pd.DataFrame({'x':pd.to_numeric(a,errors='coerce'),'y':pd.to_numeric(b,errors='coerce')}).dropna()
    n=len(y)
    if n<3:
        return None,n,'too_few_paired'
    if y.x.nunique()<2 or y.y.nunique()<2:
        return None,n,'constant_or_undefined'
    result = spearmanr(y.x.to_numpy(),y.y.to_numpy())
    rho = float(result.statistic)
    return (rho if np.isfinite(rho) else None),n,'estimated'


def comparisons(features: Dict[str,pd.DataFrame], min_cohort: int = 20) -> tuple[pd.DataFrame,pd.DataFrame]:
    combinations=[('V0','V1_RELIABLE'),('V0','V2_PREINDEX_RELIABLE'),('V1_RELIABLE','V2_PREINDEX_RELIABLE')]
    result=[]; status=[]
    for left,right in combinations:
        merged=features[left].merge(features[right],on=PK,how='inner',suffixes=('_a','_b'),validate='one_to_one')
        for cohort,part in [('OVERALL',merged), *list(merged.groupby('cohort'))]:
            for col in CORE_DYNAMICS_V23:
                x,y=part[col+'_a'],part[col+'_b']
                rho,n,reason=_rho(x,y)
                pair=pd.DataFrame({'a':pd.to_numeric(x,errors='coerce'),'b':pd.to_numeric(y,errors='coerce')}).dropna()
                if len(pair)<5:
                    qagree=None;qden=0;tie_problem=True
                elif pair.a.nunique()<2 or pair.b.nunique()<2:
                    qagree=None;qden=int(len(pair));tie_problem=True
                else:
                    # If the 80th percentile is tied, 'top 20%' is not uniquely
                    # identifiable; never count a sea of zeros as full agreement.
                    qa,qb=pair.a.quantile(.8),pair.b.quantile(.8)
                    na,nb=int((pair.a>=qa).sum()),int((pair.b>=qb).sum())
                    cutoff=int(np.ceil(.20*len(pair)))
                    tie_problem=(na>cutoff or nb>cutoff)
                    if tie_problem:
                        qagree=None
                    else:
                        aa=pair.a>=qa;bb=pair.b>=qb
                        qagree=float((aa==bb).mean())
                    qden=int(len(pair))
                result.append(dict(left=left,right=right,cohort=cohort,feature=col,
                                   n_patient_overlap=len(part),n_paired=n,spearman=rho,
                                   spearman_status=reason,top_quintile_agreement=qagree,
                                   n_top_quintile=qden,top_quintile_tie_ambiguous=bool(tie_problem),
                                   n_unique_left=int(pair.a.nunique()),n_unique_right=int(pair.b.nunique()),
                                   n_nonzero_left=int(pair.a.ne(0).sum()),n_nonzero_right=int(pair.b.ne(0).sum())))
    return pd.DataFrame(result),pd.DataFrame(status)


def within_stability(full:Dict[str,pd.DataFrame], prior:Dict[str,pd.DataFrame]) -> pd.DataFrame:
    out=[]
    for key in ['V0','V1_RELIABLE','V2_PREINDEX_RELIABLE']:
        joined=full[key].merge(prior[key],on=PK,how='inner',suffixes=('_full','_prior'),validate='one_to_one')
        for cohort,part in [('OVERALL',joined),*list(joined.groupby('cohort'))]:
            for col in CORE_DYNAMICS_V23:
                a=part[col+'_full'];b=part[col+'_prior']
                rho,n,reason=_rho(a,b)
                out.append(dict(version=key,cohort=cohort,feature=col,
                                n_patient_overlap=len(part),n_paired=n,spearman=rho,
                                status=reason))
    return pd.DataFrame(out)


def _patient_truncate(serial:pd.DataFrame)->pd.DataFrame:
    max_t=serial.groupby('patient_id').time_years.transform('max')
    return serial.loc[serial.time_years<max_t].copy()


def policy(comparison:pd.DataFrame,within:pd.DataFrame,cohorts:tuple[str,...],
           rho_min:float=.8, n_min:int=20)->pd.DataFrame:
    rows=[]
    for feature in CORE_DYNAMICS_V23:
        tests=[]
        for cohort in cohorts:
            # Eligibility is based on paired n, not the size of either input alone.
            z=comparison[(comparison.left=='V0')&(comparison.right=='V2_PREINDEX_RELIABLE')&
                         (comparison.cohort==cohort)&(comparison.feature==feature)].iloc[0]
            if int(z.n_patient_overlap)>=n_min:
                valid=int(z.n_paired)>=n_min and z.spearman_status=='estimated' and z.spearman>=rho_min
                tests.append((cohort,int(z.n_paired),z.spearman, bool(valid)))
        # Requiring >=2 independently powered cohorts avoids vacuous promotion.
        confirmable=(len(tests)>=2) and all(t[-1] for t in tests)
        rows.append(dict(feature=feature,classification=('confirmatory_v27' if confirmable else 'exploratory_v27'),
                         n_cohorts_eligible=len(tests),eligible_cohort_rhos=json.dumps(tests),
                         reason=('passes_prefrozen_rank_threshold' if confirmable else
                                 'fails_rank_or_insufficient_independent_cohorts'),
                         within_patient_stability_reported=bool((within.feature==feature).any())))
    return pd.DataFrame(rows)


def run(filtered_path:str,raw_path:str,outdir:str,draws:int=500,seed:int=17,include_approx=True):
    dest=Path(outdir);dest.mkdir(parents=True,exist_ok=True)
    p0,p1=Path(filtered_path),Path(raw_path)
    f=pd.read_csv(p0);r=pd.read_csv(p1)
    manifest={'version':VERSION,'research_use_only':True,'point_forecasting_frozen':True,
              'inputs':{str(p0):sha256(p0),str(p1):sha256(p1)},
              'thresholds_fixed':{'min_reliable_timepoints':3,'prior_vaf':0.01,
                                  'rho_min':0.8,'n_cohort_min':20,'pair_support':0.8},
              'draws':draws,'seed':seed,'time_axis_sensitivity':{},'runs':{}}
    for axis in (['original','approx_age'] if include_approx else ['original']):
        v=prepare_versions(f,r,approx_time=(axis=='approx_age'))
        # V1 floor is generated, but never treated as uncensored quantitative input.
        raw_audit=v['V1_LOD_FLOORED_AUDIT_ONLY']
        raw_audit[['patient_id','cohort','variant_id','time_years','wave','vaf',
                   'detection_limit','lod_known','depth_qc','assay_censored','quantifiable',
                   'known_pre_index','vaf_lod_floored']].to_csv(
                   dest/f'{axis}_lod_provenance.csv',index=False)
        mask=raw_audit.lod_known
        true_na=raw_audit.detection_limit.isna()
        info={'n_raw':len(raw_audit), 'n_missing_lod':int(true_na.sum()),
              'n_invalid_or_uninformative_lod':int((~mask & ~true_na).sum()),
              'n_low_depth_lbc':int((~raw_audit.depth_qc).sum()),
              'missing_lod_by_cohort':true_na.groupby(raw_audit.cohort).sum().astype(int).to_dict(),
              'n_sub_lod':int(raw_audit.assay_censored.sum()),
              'n_quantifiable':int(raw_audit.quantifiable.sum())}
        current={};previous={};sizes={}
        for name in ['V0','V1_RELIABLE','V2_PREINDEX_RELIABLE']:
            d=v[name]
            cur=compute_features(d,draws=draws,seed=seed)
            # Recompute ascertainment / reliable-variant gate after dropping LAST
            # **patient** wave: never let the removed future wave select variants.
            if name=='V0':
                historic=_patient_truncate(v['V0'])
            else:
                historic_full=_patient_truncate(raw_audit)
                vt=prepare_versions(_patient_truncate(v['V0']),historic_full,approx_time=False)
                historic=vt[name]
            prev=compute_features(historic,draws=draws,seed=seed)
            cur.to_csv(dest/f'{axis}_{name}_patient_features.csv',index=False)
            prev.to_csv(dest/f'{axis}_{name}_prior_patient_features.csv',index=False)
            current[name]=cur;previous[name]=prev
            sizes[name]={'rows':int(len(d)),'patients':int(d.patient_id.nunique()),
                         'variants':int(d.groupby(KEY).ngroups),'patients_with_feature_output':int(len(cur)),
                         'prior_patients_with_feature_output':int(len(prev))}
        comp,_=comparisons(current)
        stab=within_stability(current,previous)
        pol=policy(comp,stab,tuple(sorted(v['V0'].cohort.unique())))
        for typ,obj in [('spearman_and_quintile',comp),('within_patient',stab),('feature_status',pol)]:
            obj.to_csv(dest/f'{axis}_{typ}.csv',index=False)
        manifest['runs'][axis]={'sizes':sizes,'lod_audit':info,
                               'classification':pol.set_index('feature').classification.to_dict()}
    (dest/'v27_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--filtered',required=True);pa.add_argument('--unfiltered',required=True)
    pa.add_argument('--outdir',required=True);pa.add_argument('--draws',type=int,default=500)
    pa.add_argument('--seed',type=int,default=17)
    pa.add_argument('--original-only',action='store_true')
    a=pa.parse_args()
    print(json.dumps(run(a.filtered,a.unfiltered,a.outdir,draws=a.draws,seed=a.seed,
                         include_approx=not a.original_only),indent=2))
