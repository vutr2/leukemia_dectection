# LeukemiPrediag v2 — Real Data Audit

Generated from the current Drive data using the v2 real-cohort adapter. Research use only.

## Measurement-quality cohort

Filters: VAF >= 1.00%, LBC depth >= 500, at least 3 longitudinal time points. LBC wave spacing is approximated as 3.0 years; SardiNIA uses reported age.

- Patients retained: **265**
- Patient/variant trajectories: **423**
- Serial rows after QC: **1483**
- Missense serial rows: **818**

## Cohorts

| Cohort | Patients | Trajectories | Median timepoints | Median span |
|---|---:|---:|---:|---:|
| GSE178936 | 57 | 136 | 3.0 | 6.00 y |
| GSE288742 | 28 | 38 | 3.0 | 12.00 y |
| SardiNIA | 180 | 249 | 4.0 | 10.31 y |

## BioNeMo/ESM-2 reuse

The adapter does **not** rerun the protein model. It reuses the existing 1280-dimensional residue-delta artifacts.

- Embedding dimension: **1280**
- Patients with >=1 matched BioNeMo mutation: **174/265 (65.7%)**
- Mean mutation-level embedding coverage: **55.8%**
- Median mutation-level embedding coverage: **66.7%**

## Interpretation

This successfully validates the v2 data path:

`raw longitudinal NGS -> standardized serial table -> clone dynamics -> patient features -> existing BioNeMo molecular layer`.

It does **not** validate prediction of future leukemia. The current public cohorts do not contain a clean future-leukemia outcome label, so M0/M1/M2 progression CV must wait for an outcome-linked pre-diagnostic cohort. Creating a pseudo-label from the same mutations/VAF would be circular leakage.
