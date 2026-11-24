"""Build LeukemiPrediag v2 real-cohort feature assets.

This is a Stage-B measurement pipeline, not a progression classifier.
It standardises the currently available LBC/GSE288742/SardiNIA serial data,
builds clone-dynamics features, and reuses the already-computed BioNeMo/ESM-2
residue-delta table to create patient-level molecular representations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from real_data_adapter import (
    RealDataConfig,
    audit_serial,
    bionemo_patient_matrix,
    latest_patient_table,
    load_lbc_serial,
    load_sardinia_serial,
    qc_and_filter_serial,
)
from clone_dynamics import patient_dynamics_features


def build(args):
    cfg = RealDataConfig(
        min_vaf=args.min_vaf,
        min_depth_lbc=args.min_depth,
        min_timepoints=args.min_timepoints,
        years_per_lbc_wave=args.years_per_wave,
    )

    serial_parts = []
    for cohort, path in [
        ("GSE178936", args.lbc178936),
        ("GSE288742", args.lbc288742),
    ]:
        x = load_lbc_serial(path, cohort, cfg)
        x = qc_and_filter_serial(x, cfg)
        serial_parts.append(x)

    s = load_sardinia_serial(args.sardinia, cfg)
    s = qc_and_filter_serial(s, cfg)
    serial_parts.append(s)

    serial = pd.concat(serial_parts, ignore_index=True, sort=False)
    if serial.groupby("patient_id")["cohort"].nunique().gt(1).any():
        raise RuntimeError("patient_id collision across cohorts; prefix IDs before pooling")

    patients = latest_patient_table(serial)
    dyn = patient_dynamics_features(
        serial[["patient_id", "variant_id", "time_years", "vaf"]]
    )

    emb = pd.read_pickle(args.embedding_pickle)
    protein, coverage = bionemo_patient_matrix(
        serial, emb, patients["patient_id"].tolist(), index="last"
    )
    patient_features = (patients
                        .merge(dyn, on="patient_id", how="left", validate="one_to_one")
                        .merge(coverage, on="patient_id", how="left", validate="one_to_one"))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    serial.to_csv(out / "real_serial_filtered.csv", index=False)
    patient_features.to_csv(out / "real_patient_features.csv", index=False)
    np.save(out / "real_patient_bionemo.npy", protein.astype(np.float32))
    np.save(out / "real_patient_ids.npy", patients["patient_id"].to_numpy(dtype=str))

    audit = audit_serial(serial)
    audit.update({
        "measurement_filters": {
            "min_vaf": cfg.min_vaf,
            "min_depth_lbc": cfg.min_depth_lbc,
            "min_timepoints": cfg.min_timepoints,
            "years_per_lbc_wave": cfg.years_per_lbc_wave,
        },
        "bionemo": {
            "embedding_dim": int(protein.shape[1]),
            "patients_with_embedding": int(coverage["has_embedding"].sum()),
            "patients_total": int(len(coverage)),
            "patient_fraction_with_embedding": float(coverage["has_embedding"].mean()),
            "mean_mutation_coverage": float(coverage["bionemo_coverage"].mean()),
            "median_mutation_coverage": float(coverage["bionemo_coverage"].median()),
        },
        "important_limitation": (
            "These cohorts do not provide a clean future-leukemia outcome label. "
            "The generated assets validate feature construction and clone dynamics, "
            "not clinical progression prediction."
        ),
    })
    with open(out / "real_data_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps(audit, indent=2))
    print(f"\nWrote real-data assets to {out}")
    return audit


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--lbc178936", required=True)
    p.add_argument("--lbc288742", required=True)
    p.add_argument("--sardinia", required=True)
    p.add_argument("--embedding-pickle", required=True)
    p.add_argument("--outdir", default="data/processed/v2")
    p.add_argument("--min-vaf", type=float, default=0.01)
    p.add_argument("--min-depth", type=int, default=500)
    p.add_argument("--min-timepoints", type=int, default=3)
    p.add_argument("--years-per-wave", type=float, default=3.0)
    return p


if __name__ == "__main__":
    build(parser().parse_args())
