# LeukemiPrediag — Research-to-Clinical Validation Plan

**Document version:** 2026-10-09 / v2.7 | **Classification:** research protocol draft | **Current use:** non-clinical, retrospective computational analysis only

## 1. Intended use and current boundary

The research engine characterizes serial blood sequencing measurements (VAF changes, candidate same-clone dynamics and uncertainty). It must **not** generate patient-facing AML/MDS probability, recommend treatment, or change clinical management. It is not a validated screening test.

The public data used in v2.7 do not provide a clean prediagnostic future AML/MDS endpoint. Thus current results cannot estimate diagnostic sensitivity, specificity, PPV, NPV, time-to-progression risk, clinical benefit, or utility.

## 2. Paper that is plausible now

**Research question:** Are pre-defined longitudinal clonal hematopoiesis measurements robust to assay detection limits, prospective variant ascertainment and sampling-time uncertainty?

**Suggested title (non-clinical):** *Assay-detection-aware robustness of longitudinal clonal hematopoiesis features across three public cohorts.*

**Primary analysis:** compare the 9 locked v2.3 features between historical V0 (1% filtered) and V2 (pre-index and above-LoD); Spearman by cohort, with the >=0.80 locked criterion, patient-level 95% bootstrap CI, top-quintile agreement only when top 20% is identifiable. Report all nine results, including negative ones.

**Secondary analyses:** V0/V1 and V1/V2 differences, all-vs-prior-patient-wave stability, original versus APPROXIMATE time axis, missing LoD and depth-QC attrition, fraction of patients with identifiably estimated features.

**Reporting:** use STROBE for observational cohort methods. The future progression prediction study will require TRIPOD+AI (not an assertion that this outcome-free paper is a prediction-model validation). Guidelines: STROBE https://www.equator-network.org/reporting-guidelines/strobe/ ; TRIPOD+AI https://www.bmj.com/content/385/bmj-2023-078378 .

**Reproducibility:** release code hashes, exact upstream data accession/file versions, immutable seed, QC thresholds, environment/requirements, cohort exclusions, raw-to-analysis flowchart, and summary-level results. Respect controlled-access licensing and protect patient-level data from public release unless permitted.

**Independent replication required:** at least one genuinely independent assay/cohort, with per-timepoint sample age/date, known target-locus coverage and alternate/read depth even when negative. Do not call V0/V2 internal stability an external validation.

## 3. Measurement-validation gates before first hospital pilot

1. **Traceability:** documented variant reference genome/build, protein annotation/version, sample/patient linkage, collection timestamps, specimen and assay types, batch and center, QC failures and duplicates.
2. **Assay characteristics:** per-locus validated analytical LoD and blank/false-positive behavior, read and UMI depth, replicate precision near LoD, accuracy/reproducibility, no-call versus true-negative flags, artifact suppression. Test limits in a laboratory quality system.
3. **Reference method:** independently characterize a subset with orthogonal deep sequencing, ddPCR and/or single-cell/colony sequencing where clinically appropriate. Co-movement alone never proves same clone.
4. **Longitudinal robustness:** blinded technical replicates at different sequencing runs/batches; measured intra- and inter-lab feature repeatability; per-cohort and per-platform missingness/identifiability; variation due to sampling times.
5. **Software verification:** unit/integration tests on clean environment, immutable reproducible outputs, deterministic random seed, input validation, controlled dependencies and version tags, model-card limitations, access controls and audit logging.
6. **Feature governance:** use versioned feature spec only; document no-estimate/unknown rather than imputed stability; freeze feature definitions before new datasets; evaluate robustness independently before upgrading any designation.

## 4. Earliest ethically defensible prospective study

Start with an **observational, non-interventional shadow-mode feasibility study** under institutional approvals, led by a clinical hematology and molecular pathology team. No algorithm outputs should alter patient care or trigger testing/treatment. Obtain IRB/ethics oversight or formal determination; consent/waiver and HIPAA/GDPR/local privacy review as applicable.

**Protocol pre-specification:** inclusion/exclusion criteria, defined intended population (e.g., CH/CCUS investigated in hematology), sample collection schedules, loci and depth thresholds, pre-index landmark, failure-to-call handling, precise model/spec version, missing-data handling, outcome adjudication (only if prospective outcomes are later used), analytic sample-size reasoning by statistician, blinded external validation plan, subgroup/assay sensitivity analyses and prospective stopping/safety rules.

**First study endpoints (not clinical efficacy):** specimen-to-report success and QC rate, within/between-run feature reproducibility, fraction of patients whose trajectories are analyzable, agreement against reference quantification, and calibration/coverage of reporting uncertainty. Emphasize laboratory reproducibility before clinical risk.

Once genuinely outcome-linked longitudinal data and enough confirmed events are available, a **separate**, newly specified prognosis study can evaluate pre-index AML/MDS time-to-event prediction, censoring, M0 static baseline versus M1 dynamics versus M2 molecular representation, external calibration, decision analysis and harms. No labels may be derived from VAF or growth itself. Report this future prognostic study following TRIPOD+AI, use prediction-model risk-of-bias assessment, and choose sample size around number of outcome events—not total sequencing rows.

A live decision-support feasibility study, if and only if evidence, ethics and relevant regulatory conditions are met, should consider DECIDE-AI: https://www.bmj.com/content/377/bmj-2022-070904 . That guidance describes reporting early clinical AI decision-support evaluations; it is not permission to deploy an unvalidated tool.

## 5. Evidence gaps blocking clinical use

- No validated prediagnostic AML/MDS event labels and censoring for present cohorts.
- No full all-locus negative-call pileup matrix; source selection remains approximately 1%-ascertained.
- Most LBC measurements lie under X95MDAF; a numerical AF under LoD is not interpretable quantitative growth.
- Important missing per-row LoD and some very low-depth samples; do not impute LoD silently.
- GSE288742 has only a cohort-level approximate time mapping, not individual sample dates.
- Co-clonal groups are computational candidates, not biologically confirmed phylogenies.
- Internal feature stability does not prove external analytic or clinical validity.
- Clinical site partnerships, laboratory governance and regulatory pathway have not been established in this file.

## 6. Release gates

**Gate A (research manuscript):** reproducible runs, complete missingness/attrition table, all features and negative results, transparent limitations, independent statistical/code review.

**Gate B (prospective observational pilot):** approved protocol, partner lab validation/QA, consent/privacy/governance, frozen code and feature spec, safety fallback: no output modifies care.

**Gate C (clinical prediction study):** verified outcome-linked data, sufficient events and follow-up, blinded external multi-site evaluation of pre-frozen M0/M1/M2, calibrated patient risk and net benefit, complete bias/harm assessment.

**Gate D (clinical implementation):** only after external validation, clinician-in-the-loop assessment and all applicable clinical/regulatory requirements are met. This roadmap does not itself establish those requirements have been met.
