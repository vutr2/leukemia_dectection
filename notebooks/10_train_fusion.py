"""Train/evaluate LeukemiPrediag v2 fusion ladder from prepared research tables.

Example
-------
python 10_train_fusion.py \
  --patients data/processed/patients.csv \
  --serial data/processed/serial_vaf.csv \
  --mutations data/processed/mutations.csv \
  --embeddings models/pooled_missense_with_delta.pkl \
  --outdir results/fusion_v2

The script evaluates:
  M0 = static clinical/genetic baseline
  M1 = M0 + clone dynamics
  M2 = M1 + BioNeMo/ESM-2 residue-delta representation

It never treats diagnosis-vs-CHIP labels as proof of early detection. For the
progression task, provide a pre-diagnostic cohort with outcome-linked follow-up.
Research use only; not a diagnostic device.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.clone_dynamics import patient_dynamics_features
    from src.embedding import patient_embedding_matrix
    from src.fusion_model import (
        build_feature_bundle,
        evaluate_ladder,
        bootstrap_increment,
        random_embedding_null,
        sensitivity_threshold,
        fit_final,
        save_artifact,
    )
    from src.longitudinal import make_landmark_labels
except ImportError:
    from clone_dynamics import patient_dynamics_features
    from embedding import patient_embedding_matrix
    from fusion_model import (
        build_feature_bundle,
        evaluate_ladder,
        bootstrap_increment,
        random_embedding_null,
        sensitivity_threshold,
        fit_final,
        save_artifact,
    )
    from longitudinal import make_landmark_labels


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    s = p.suffix.lower()
    if s in {".csv"}:
        return pd.read_csv(p)
    if s in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    if s in {".pkl", ".pickle"}:
        obj = pd.read_pickle(p)
        if not isinstance(obj, pd.DataFrame):
            raise TypeError(f"{path} is not a pandas DataFrame")
        return obj
    if s == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported table format: {path}")


def load_embedding_table(path: str) -> pd.DataFrame:
    d = read_table(path)
    if "delta" not in d.columns:
        # Accept expanded columns esm_0...esm_N and fold them into delta arrays.
        cols = [c for c in d.columns if c.startswith("esm_") or c.startswith("delta_")]
        if not cols:
            raise KeyError("embedding table needs a `delta` column or esm_*/delta_* columns")
        d = d.copy()
        d["delta"] = list(d[cols].to_numpy(float))
    return d


def aligned_patient_protein_matrix(
    patients: pd.DataFrame,
    mutation_table: pd.DataFrame,
    embedding_table: pd.DataFrame,
    patient_col: str = "patient_id",
) -> np.ndarray:
    """Aggregate mutation-level deltas into a VAF-weighted patient matrix."""
    need = {patient_col, "mut_key", "vaf"}
    miss = need - set(mutation_table.columns)
    if miss:
        raise KeyError(f"mutations missing {sorted(miss)}")
    if "mut_key" not in embedding_table.columns or "delta" not in embedding_table.columns:
        raise KeyError("embeddings need mut_key + delta")
    # Existing helper expects patient_id specifically.
    m = mutation_table.copy()
    if patient_col != "patient_id":
        m = m.rename(columns={patient_col: "patient_id"})
    pids = patients[patient_col].astype(str).tolist()
    m["patient_id"] = m["patient_id"].astype(str)
    emb = patient_embedding_matrix(m, embedding_table, pids, weight_by_vaf=True)
    # The helper appends a has_embedding flag. Keep that flag as static tabular
    # context and return only the molecular vector here.
    return emb[:, :-1], emb[:, -1]


def maybe_make_labels(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.label_col in df.columns:
        out = df.copy()
        out[args.label_col] = pd.to_numeric(out[args.label_col], errors="raise").astype(int)
        return out
    if args.event_col in df.columns and args.time_col in df.columns:
        out = make_landmark_labels(
            df,
            time_col=args.time_col,
            event_col=args.event_col,
            horizon=args.horizon,
        )
        if "label" != args.label_col:
            out = out.rename(columns={"label": args.label_col})
        return out
    raise KeyError(
        f"Need either `{args.label_col}` or both `{args.event_col}` and `{args.time_col}`"
    )


def parse_genes(s: str):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", required=True, help="One row per prediction index/patient")
    ap.add_argument("--serial", help="Serial VAF table for clone-dynamics features")
    ap.add_argument("--mutations", help="Mutation table with patient_id, mut_key, vaf")
    ap.add_argument("--embeddings", help="Mutation-level BioNeMo/ESM2 table with mut_key, delta")
    ap.add_argument("--outdir", default="results/fusion_v2")
    ap.add_argument("--patient-col", default="patient_id")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--event-col", default="event")
    ap.add_argument("--time-col", default="years_to_event_or_censor")
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--group-col", default=None,
                    help="Optional grouping variable for StratifiedGroupKFold")
    ap.add_argument("--genes", default="",
                    help="Comma-separated genes for one-hot static baseline")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-pca", type=int, default=16)
    ap.add_argument("--target-specificity", type=float, default=0.98)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--null-reps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    patients = read_table(args.patients)
    if args.patient_col not in patients.columns:
        raise KeyError(f"patients missing {args.patient_col}")
    patients[args.patient_col] = patients[args.patient_col].astype(str)
    patients = maybe_make_labels(patients, args).reset_index(drop=True)

    if patients[args.patient_col].duplicated().any():
        raise ValueError("patients must contain one row per prediction index after labeling")

    dynamics = None
    if args.serial:
        serial = read_table(args.serial)
        serial[args.patient_col] = serial[args.patient_col].astype(str)
        # clone_dynamics defaults use patient_id; normalize locally.
        if args.patient_col != "patient_id":
            serial = serial.rename(columns={args.patient_col: "patient_id"})
        # Expected serial columns: patient_id, variant_id, time_years, vaf.
        dynamics = patient_dynamics_features(serial)
        if args.patient_col != "patient_id":
            dynamics = dynamics.rename(columns={"patient_id": args.patient_col})
        dynamics.to_csv(outdir / "patient_dynamics_features.csv", index=False)

    protein = None
    if args.embeddings:
        if not args.mutations:
            raise ValueError("--embeddings requires --mutations to aggregate per patient")
        muts = read_table(args.mutations)
        muts[args.patient_col] = muts[args.patient_col].astype(str)
        embs = load_embedding_table(args.embeddings)
        protein, has_emb = aligned_patient_protein_matrix(
            patients, muts, embs, patient_col=args.patient_col
        )
        patients = patients.copy()
        patients["has_embedding"] = has_emb
        np.save(outdir / "patient_bionemo_matrix.npy", protein)

    genes = parse_genes(args.genes)
    bundle = build_feature_bundle(
        patients,
        dynamics=dynamics,
        protein_matrix=protein,
        genes=genes,
        patient_col=args.patient_col,
    )
    y = patients[args.label_col].to_numpy(int)
    groups = patients[args.group_col].to_numpy() if args.group_col else None

    summary, preds, folds = evaluate_ladder(
        bundle, y,
        groups=groups,
        n_splits=args.n_splits,
        seed=args.seed,
        n_pca=args.n_pca,
        target_specificity=args.target_specificity,
    )
    summary.to_csv(outdir / "fusion_ladder_summary.csv")
    for name, ft in folds.items():
        ft.to_csv(outdir / f"fold_metrics_{name}.csv")

    oof = pd.DataFrame({args.patient_col: patients[args.patient_col], args.label_col: y})
    for name, p in preds.items():
        oof[f"p_{name}"] = p
    oof.to_csv(outdir / "oof_predictions.csv", index=False)

    increments = {}
    names = list(preds)
    for a, b in zip(names[:-1], names[1:]):
        increments[f"{b}_vs_{a}"] = bootstrap_increment(
            y, preds[a], preds[b], n_boot=args.bootstrap, seed=args.seed
        )

    if protein is not None and args.null_reps > 0:
        increments["BioNeMo_random_embedding_null"] = random_embedding_null(
            bundle, y, groups=groups, n_reps=args.null_reps,
            n_splits=args.n_splits, n_pca=args.n_pca,
            target_specificity=args.target_specificity,
            seed=100 + args.seed,
        )

    with open(outdir / "increment_tests.json", "w") as f:
        json.dump(increments, f, indent=2)

    # Fit the highest available level after the evaluation design is frozen.
    if protein is not None:
        level = "M2_plus_BioNeMo"
        X = bundle.X_m2
        n_tab = bundle.X_static.shape[1] + bundle.X_dynamics.shape[1]
        n_prot = bundle.X_protein.shape[1]
    elif bundle.X_dynamics.shape[1]:
        level = "M1_plus_dynamics"
        X = bundle.X_m1
        n_tab, n_prot = X.shape[1], 0
    else:
        level = "M0_static"
        X = bundle.X_m0
        n_tab, n_prot = X.shape[1], 0

    threshold = sensitivity_threshold(y, preds[level], args.target_specificity)
    artifact = fit_final(
        X, y,
        n_tabular=n_tab,
        n_protein=n_prot,
        n_pca=args.n_pca,
        threshold=threshold,
        feature_level=level,
        static_names=bundle.static_names,
        dynamics_names=bundle.dynamics_names,
        genes=genes,
        target_specificity=args.target_specificity,
    )
    model_path = str(outdir / "leukemiprediag_fusion_v2.joblib")
    save_artifact(
        artifact,
        model_path,
        extra_meta={
            "n_train": int(len(y)),
            "n_positive": int(y.sum()),
            "label_horizon_years": float(args.horizon),
            "source_patients": str(args.patients),
            "source_serial": str(args.serial) if args.serial else None,
            "source_embeddings": str(args.embeddings) if args.embeddings else None,
        },
    )

    print("\n=== LeukemiPrediag v2 fusion ladder ===")
    print(summary.round(4).to_string())
    print(f"\nSaved research artifact: {model_path}")
    print("Research use only. Do not interpret this artifact as clinical validation.")


if __name__ == "__main__":
    main()
