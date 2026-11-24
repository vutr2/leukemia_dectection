"""Frozen model specification for LeukemiPrediag v2.3.

The purpose of this module is to separate a compact, pre-specified feature set
from the much larger exploratory feature catalogue.  This reduces redundancy
and overfitting risk when an outcome-linked cohort becomes available.

Research use only. Not a diagnostic device.
"""
from __future__ import annotations

# Published CHRS high-risk mutation genes (Weeks et al., NEJM Evidence 2023).
CHRS_HIGH_RISK_GENES = (
    "SRSF2", "SF3B1", "ZRSR2", "JAK2", "TP53",
    "IDH1", "IDH2", "FLT3", "RUNX1",
)

# Static clinical/genetic component model.  chrs_score is intentionally NOT
# included here because using the score together with all of its components
# double-counts the same information.  A CHRS-score-only reference baseline can
# be evaluated separately when all CHRS inputs are available.
CORE_STATIC_V23 = (
    "n_mutations", "max_vaf", "age",
    "rdw", "mcv", "n_cytopenias",
    "single_dnmt3a", "high_risk_mutation", "ccus",
)

CHRS_REFERENCE_V23 = ("chrs_score",)

# Compact longitudinal feature set.  These were chosen to represent distinct
# biological/measurement concepts rather than every correlated transformation.
CORE_DYNAMICS_V23 = (
    # Growth magnitude and uncertainty.
    "posterior_max_growth",
    "posterior_median_growth_sd",
    # Raw-scale kinetic complement to log growth.
    "max_delta_vaf_per_year",
    # Candidate co-clonal structure.
    "n_supported_variant_pairs",
    "fraction_evaluated_pairs_supported",
    "n_multivariant_clones",
    # Ecological structure / competition proxies.
    "clone_vaf_entropy",
    "clone_growth_dispersion",
    # Observed dynamic crossing, not a diagnostic rule.
    "n_variants_crossed_vaf_ge_20pct",
)

# Exploratory features are useful for discovery and ablation studies but should
# not all be entered into a small clinical cohort by default.
EXPERIMENTAL_DYNAMICS_V23 = (
    "max_log_vaf_acceleration",
    "median_log_vaf_acceleration",
    "max_end_log_vaf_growth",
)

# Full v2.2 catalogue retained for sensitivity analyses.
EXTENDED_DYNAMICS_V23 = (
    "n_tracked_variants", "n_growing_variants", "max_log_vaf_slope",
    "median_log_vaf_slope", "slope_dispersion", "max_observed_vaf",
    "median_trajectory_rmse", "n_candidate_coclone_pairs",
    "mean_pairwise_log_vaf_corr", "structure_identifiable",
    "n_inferred_clones", "n_multivariant_clones", "fraction_variants_grouped",
    "max_clone_growth", "median_clone_growth", "max_clone_p_growing",
    "max_clone_p_fast_growth", "dominant_clone_vaf", "dominant_clone_growth",
    "dominant_clone_p_growing", "mean_clone_coherence", "min_clone_coherence",
    "structure_supported",
    "max_delta_vaf_per_year", "max_observed_log_vaf_growth",
    "median_observed_log_vaf_growth", "fastest_observed_doubling_time_years",
    "median_observed_doubling_time_years", "n_acceleration_estimable",
    "max_log_vaf_acceleration", "median_log_vaf_acceleration",
    "max_end_log_vaf_growth", "n_variants_crossed_vaf_ge_10pct",
    "n_variants_crossed_vaf_ge_20pct",
    "posterior_max_growth", "posterior_median_growth",
    "posterior_median_growth_sd", "posterior_max_p_growing",
    "posterior_max_p_fast_growth", "fastest_posterior_doubling_time_years",
    "median_posterior_doubling_time_years",
    "n_variant_pairs_evaluated", "max_pair_support", "mean_pair_support",
    "n_supported_variant_pairs", "mean_supported_pair_support",
    "n_supported_gene_pairs", "fraction_evaluated_pairs_supported",
    "n_candidate_clones_v22", "max_variants_per_candidate_clone",
    "clone_growth_dispersion", "clone_growth_range",
    "n_fast_growth_clones_mean", "fastest_clone_doubling_time_years",
    "median_clone_doubling_time_years", "dominant_clone_fraction_proxy",
    "clone_vaf_entropy", "effective_clone_number_proxy",
    "any_clone_member_vaf_ge_0p20", "n_clones_member_vaf_ge_0p20",
)

MODEL_SPEC_V23 = {
    "version": "LeukemiPrediag-v2.3",
    "static_default": list(CORE_STATIC_V23),
    "chrs_reference": list(CHRS_REFERENCE_V23),
    "dynamics_default": list(CORE_DYNAMICS_V23),
    "dynamics_experimental": list(EXPERIMENTAL_DYNAMICS_V23),
    "dynamics_extended": list(EXTENDED_DYNAMICS_V23),
    "principle": (
        "Use the compact core set for confirmatory clinical evaluation; use the "
        "extended set only for predeclared exploratory/sensitivity analyses."
    ),
}
