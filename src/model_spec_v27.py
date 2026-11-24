"""LeukemiPrediag v2.7 PRELIMINARY feature-governance decisions.

Leaves the v2.3 model_spec.py untouched. No prospective clinical confirmation,
no outcome labels, and no change to frozen VAF point forecasting.
The original rank threshold (rho >= .8 in each powered cohort, n >= 20) was
specified before the v2.7 analysis. Two powered cohorts are required to avoid
vacuous acceptance. The VAF >=20% crossing result has sparse positive cases.
"""
from model_spec import CORE_DYNAMICS_V23

V27_MIN_SPEARMAN = 0.80
V27_MIN_PAIRED_PATIENTS = 20
V27_MIN_POWERED_COHORTS = 2
V27_MIN_QUANTIFIABLE_TIMEPOINTS = 3
V27_SUPPORT_THRESHOLD = 0.80

RANK_STABLE_V27 = (
    'max_delta_vaf_per_year',
    'n_variants_crossed_vaf_ge_20pct',
)
EXPLORATORY_V27 = tuple(x for x in CORE_DYNAMICS_V23 if x not in RANK_STABLE_V27)

MODEL_SPEC_V27 = {
    'version': 'LeukemiPrediag-v2.7',
    'source': 'V27_MODEL_AUDIT.md + output v27 feature_status.csv',
    'predecessor_unchanged': 'model_spec.py / CORE_DYNAMICS_V23',
    'confirmatory_rank_stable_internal_only': list(RANK_STABLE_V27),
    'exploratory_or_unstable': list(EXPLORATORY_V27),
    'continuous_candidate_for_independent_validation': ['max_delta_vaf_per_year'],
    'sparse_discrete_candidate_for_independent_validation': ['n_variants_crossed_vaf_ge_20pct'],
    'top_quintile_tie_warning': True,
    'not_clinically_validated': True,
    'no_aml_mds_outcomes': True,
    'do_not_train_point_forecaster': True,
    'criteria': dict(min_spearman=V27_MIN_SPEARMAN,
                     min_paired_patients=V27_MIN_PAIRED_PATIENTS,
                     min_powered_cohorts=V27_MIN_POWERED_COHORTS,
                     min_quantifiable_timepoints=V27_MIN_QUANTIFIABLE_TIMEPOINTS,
                     pair_support_threshold=V27_SUPPORT_THRESHOLD),
}
