# LeukemiPrediag v2.2 Feature Audit

Research use only. Not a diagnostic device.

## Input

Built from the existing Stage-B assets produced by notebooks 12 and 13:

- `data/processed/v2/real_serial_filtered.csv`
- `data/processed/v2/clone_inference/trajectory_posteriors.csv`
- `data/processed/v2/clone_inference/pair_support.csv`
- `data/processed/v2/clone_inference/clone_assignments.csv`
- `data/processed/v2/clone_inference/clone_posteriors.csv`
- `data/processed/v2/clone_inference/patient_clone_features.csv`

## Real-data audit

- 265 participants
- 423 patient-variant longitudinal trajectories
- 21 variant pairs at longitudinal pair-support >= 0.80
- 18 participants with supported multi-variant candidate-clone structure
- 129 participants with at least one trajectory having >=4 timepoints, allowing an exploratory acceleration estimate
- 44 participants with at least one observed variant reaching VAF >=20% in the currently filtered trajectories
- 76 numeric/categorical research feature columns excluding identifiers and the human-readable `genes` field

## New v2.2 feature families

1. VAF burden and observed threshold indicators.
2. First-order kinetics: raw dVAF/dt and log-VAF growth.
3. Positive-growth doubling time.
4. Exploratory log-VAF acceleration when >=4 unique timepoints are available.
5. v2.1 posterior growth and uncertainty summaries.
6. Candidate-clone growth dispersion, dominance and diversity proxies.
7. Pairwise co-movement / supported co-mutation architecture.
8. Configurable gene-presence indicators.

## Interpretation boundaries

- VAF thresholds are covariates, not standalone diagnostic rules.
- Doubling time is only defined for positive growth estimates.
- Acceleration is experimental and noise-sensitive because the public trajectories are sparse.
- Longitudinal pair support is candidate same-clone evidence, not proof of single-cell co-occurrence or phylogeny.
- Gene-presence indicators are features, not hard-coded risk labels.
- These public cohorts still do not provide a clean future-leukemia endpoint, so v2.2 does not establish progression prediction.
