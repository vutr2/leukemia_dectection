# LeukemiPrediag — Unfiltered Serial-Call Audit (v2.5)

**Date:** 2026-10-07  
**Status:** research use only; not a diagnostic device.

## Why this audit was done

The v2.4 growth audit identified an important measurement problem: `data/processed/v2/real_serial_filtered.csv` applies `VAF >= 1%` before trajectory modeling. This selects observations on detectability and can bias near-floor slopes upward.

The filter is applied in two places:

1. `src/real_data_adapter.py` — `RealDataConfig.min_vaf = 0.01`, and `qc_and_filter_serial()` keeps `vaf >= min_vaf` together with the LBC depth threshold.
2. `notebooks/12_build_real_features.py` — each cohort is loaded and immediately passed through `qc_and_filter_serial()` before pooling and writing `real_serial_filtered.csv`.

No existing filtered asset was overwritten.

## Raw files inspected

- GSE178936: `GSE178936_LBC_ARCHER.1PCT_VAF.Feb22.non-synonymous.tsv`
- GSE288742: `GSE288742_ELC_CHIP.2024_LBC_Cohort.1PCT_VAF_NON-SYNONYMOUS.tsv`
- SardiNIA: `41586_2022_4785_MOESM5_ESM.xlsx` (Supplementary Table 2)

## Main finding

**Step 2 succeeded. All three supplied raw cohort files contain sub-1% measurements, including zero-VAF rows.** Therefore it is possible to recover within-trajectory measurements that the v2 preprocessing removed.

The new asset is:

```text
data/processed/v2/real_serial_unfiltered.csv
```

It contains exactly one row per `patient_id x cohort x variant_id x time_years` for rows actually present in the supplied source files. No absent patient/variant/timepoint row was invented.

## Recovered rows

| Cohort | Unfiltered rows | v2 filtered rows | Rows recovered | Patients | Trajectories | Rows VAF <1% | Zero-VAF rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSE178936 | 10,173 | 434 | 9,739 | 85 | 3,112 | 6,323 | 566 |
| GSE288742 | 3,616 | 114 | 3,502 | 63 | 1,268 | 2,106 | 123 |
| SardiNIA | 4,292 | 935 | 3,357 | 349 | 1,046 | 2,926 | 717 |
| **Total** | **18,081** | **1,483** | **16,598** | — | **5,426** | **11,355** | **1,406** |

Among the 5,426 represented trajectories, 5,042 have at least 3 timepoints and are eligible for leave-last-timepoint forecasting.

## Measurement fields

### GSE178936 / GSE288742

The raw LBC tables contain:

- `AF` -> `vaf`
- `AO` -> `alt_reads`
- `DP` -> `depth`
- `X95MDAF` -> per-row detection-limit field when available

Detection-limit availability:

- GSE178936: 98.86%
- GSE288742: 79.26%

The v2.5 table preserves positive alt-read measurements even when `AF < X95MDAF`.

### SardiNIA

Supplementary Table 2 contains serial VAF values including values below 1% and exact zeros, but it does **not** provide per-timepoint read depth or alternate-read counts. These fields are therefore left missing; no nominal depth is inserted into `real_serial_unfiltered.csv`.

Fabre et al. report that if a variant was identified at any study timepoint, that site was re-queried at the individual's other timepoints. The paper also uses VAF >= 0.2% as an operational peripheral-blood detection definition. v2.5 records `0.002` as the SardiNIA operational detection limit with an explicit provenance string; it is not represented as a per-call assay field from Supplementary Table 2.

Reference: Fabre et al., *Nature* 606, 335–342 (2022), doi:10.1038/s41586-022-04785-z.

## New columns relevant to censoring

The new table includes:

```text
vaf
alt_reads
depth
detected
censored
observed_alt_signal
detection_limit
detection_limit_source
below_detection_limit
below_pipeline_1pct
pipeline_filter_floor
```

For LBC rows with counts, downstream v2.5 modeling uses the exact binomial count likelihood even for values below the formal detection-limit field. For SardiNIA, where counts are unavailable, the censoring-aware branch uses the reported VAF and the documented operational 0.2% threshold.

## Critical remaining ascertainment limitation

The two LBC source filenames are themselves labelled `1PCT_VAF`. Therefore this new file is best described as **trajectory-unfiltered within the supplied 1%-selected variant universe**, not as a fully ascertainment-free sequencing callset.

It recovers sub-floor and zero measurements for represented variants, but variants that never entered the supplied 1% source universe may still be missing completely. A future assay-level pileup/VCF matrix covering all interrogated loci would be needed to remove that source-level ascertainment bias.

## Conclusion

The raw files do contain usable sub-floor information, so a censoring-aware forecasting experiment is scientifically permissible. The experiment must retain the source-selection limitation above and must not claim that the resulting table is a complete set of all sub-1% clones.
