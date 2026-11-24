"""Real-cohort adapters for LeukemiPrediag v2.

Standardises the currently available longitudinal cohorts (LBC GSE178936,
LBC GSE288742 and SardiNIA) into one conservative schema:

    patient_id, cohort, variant_id, time_years, vaf, depth,
    gene, protein_change, mut_key, kind

The module is intentionally outcome-agnostic. These public cohorts are useful
for measuring clone dynamics, but they do not provide the prospective leukemia
outcome labels required to train the final progression classifier.

Research use only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .mutation_utils import parse_protein_change, mutation_key
except ImportError:  # script/notebook use
    from mutation_utils import parse_protein_change, mutation_key


@dataclass
class RealDataConfig:
    min_vaf: float = 0.01
    min_depth_lbc: int = 500
    min_timepoints: int = 3
    years_per_lbc_wave: float = 3.0
    sardinia_nominal_depth: float = 1065.0


def _add_mutation_fields(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    parsed = out["protein_change"].apply(parse_protein_change)
    out["kind"] = parsed.apply(lambda p: p["kind"] if p else "unparsed")
    out["aa_wt"] = parsed.apply(lambda p: p["wt"] if p else None)
    out["aa_pos"] = parsed.apply(lambda p: p["pos"] if p else np.nan)
    out["aa_mut"] = parsed.apply(lambda p: p["mut"] if p else None)
    out["mut_key"] = [mutation_key(g, c) for g, c in zip(out["gene"], out["protein_change"])]
    return out


def load_lbc_serial(path: str, cohort: str, cfg: RealDataConfig = RealDataConfig()) -> pd.DataFrame:
    """Load one LBC-style non-synonymous TSV into the v2 serial schema."""
    d = pd.read_csv(path, sep="\t", low_memory=False)
    req = ["participant_id", "chromosome", "position", "reference", "mutation",
           "wave", "AF", "DP", "PreferredSymbol", "protein_substitution"]
    missing = [c for c in req if c not in d.columns]
    if missing:
        raise KeyError(f"LBC table missing columns: {missing}")

    x = pd.DataFrame({
        "patient_id": d["participant_id"].astype(str),
        "cohort": cohort,
        "variant_id": (
            d["chromosome"].astype(str) + ":" + d["position"].astype(str) + ":" +
            d["reference"].astype(str) + ">" + d["mutation"].astype(str)
        ),
        "wave": pd.to_numeric(d["wave"], errors="coerce"),
        "vaf": pd.to_numeric(d["AF"], errors="coerce"),
        "depth": pd.to_numeric(d["DP"], errors="coerce"),
        "gene": d["PreferredSymbol"].astype(str).str.upper().str.strip(),
        "protein_change": d["protein_substitution"],
    })
    # Absolute offset is irrelevant to slopes; use a common wave spacing and
    # then make time relative to the first observation for each participant.
    x["time_years"] = x["wave"] * float(cfg.years_per_lbc_wave)
    x["time_years"] -= x.groupby("patient_id")["time_years"].transform("min")
    x = _add_mutation_fields(x)
    return x


def load_sardinia_serial(path: str, cfg: RealDataConfig = RealDataConfig()) -> pd.DataFrame:
    """Load Fabre et al. SardiNIA Supplementary Table 2 into the serial schema."""
    d = pd.read_excel(path, header=3)
    d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
    d = d.dropna(subset=["Sample ID", "VAF", "Age", "Gene", "Chromosome", "Start"])
    x = pd.DataFrame({
        "patient_id": d["Sample ID"].astype(str),
        "cohort": "SardiNIA",
        "variant_id": (
            d["Chromosome"].astype(str) + ":" + d["Start"].astype(str) + ":" +
            d["WT"].astype(str) + ">" + d["MT"].astype(str)
        ),
        "wave": pd.to_numeric(d["Study Phase"], errors="coerce"),
        "age": pd.to_numeric(d["Age"], errors="coerce"),
        "vaf": pd.to_numeric(d["VAF"], errors="coerce"),
        # Supplement reports cohort-level mean coverage, not per-call depth.
        "depth": float(cfg.sardinia_nominal_depth),
        "gene": d["Gene"].astype(str).str.upper().str.strip(),
        "protein_change": d["Protein"],
    })
    x["time_years"] = x["age"] - x.groupby("patient_id")["age"].transform("min")
    x = _add_mutation_fields(x)
    return x


def qc_and_filter_serial(serial: pd.DataFrame, cfg: RealDataConfig = RealDataConfig()) -> pd.DataFrame:
    """Apply the measurement filters already validated by the fitness pipeline.

    Important: the VAF/depth thresholds are measurement-QC rules, not clinical
    risk thresholds. SardiNIA has no per-call depth, so its nominal depth keeps
    the depth filter inert while VAF/timepoint QC still applies.
    """
    d = serial.copy()
    d = d[np.isfinite(d["vaf"]) & np.isfinite(d["time_years"])]
    d = d[(d["vaf"] >= cfg.min_vaf) & (d["depth"] >= cfg.min_depth_lbc)]

    # Collapse technical duplicates at the same patient/variant/time.
    key = ["patient_id", "cohort", "variant_id", "time_years"]
    agg = {
        "vaf": "mean", "depth": "median", "gene": "first", "protein_change": "first",
        "mut_key": "first", "kind": "first", "aa_wt": "first", "aa_pos": "first",
        "aa_mut": "first", "wave": "first",
    }
    if "age" in d.columns:
        agg["age"] = "first"
    d = d.groupby(key, as_index=False).agg(agg)

    n_tp = d.groupby(["patient_id", "cohort", "variant_id"])["time_years"].transform("nunique")
    d = d[n_tp >= cfg.min_timepoints].reset_index(drop=True)
    return d


def latest_patient_table(serial: pd.DataFrame) -> pd.DataFrame:
    """Build one non-outcome patient table at the last observed time point.

    This is useful for feature engineering/auditing only. In a true progression
    study the index time must be frozen before the future outcome window.
    """
    if serial.empty:
        return pd.DataFrame(columns=["patient_id", "cohort"])
    max_t = serial.groupby(["patient_id", "cohort"])["time_years"].transform("max")
    last = serial[np.isclose(serial["time_years"], max_t)].copy()
    g = last.groupby(["patient_id", "cohort"], sort=False)
    out = pd.DataFrame({
        "n_mutations": g["variant_id"].nunique(),
        "max_vaf": g["vaf"].max(),
        "mean_vaf": g["vaf"].mean(),
        "sum_vaf": g["vaf"].sum(),
        "n_genes": g["gene"].nunique(),
        "genes": g["gene"].apply(lambda s: sorted(set(s))),
        "mut_keys": g["mut_key"].apply(lambda s: sorted(set(s))),
    }).reset_index()
    if "age" in last.columns:
        age = g["age"].max().rename("age").reset_index()
        out = out.merge(age, on=["patient_id", "cohort"], how="left")
    return out


def bionemo_patient_matrix(
    serial: pd.DataFrame,
    embedding_table: pd.DataFrame,
    patient_order: Sequence[str],
    *,
    index: str = "last",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Aggregate existing residue-delta vectors to patient level.

    Uses one row per patient/mutation and VAF weights. With index='last', the
    weight is VAF at the last observed patient time. No protein model is rerun.

    Returns
    -------
    matrix : (n_patients, hidden_dim)
    coverage : patient-level counts/fractions of embeddable mutations
    """
    emb = embedding_table.drop_duplicates("mut_key").copy()
    emb = emb[emb["delta"].apply(lambda x: isinstance(x, np.ndarray) and x.ndim == 1)]
    if emb.empty:
        raise ValueError("embedding_table has no usable delta vectors")
    lookup: Dict[str, np.ndarray] = dict(zip(emb["mut_key"], emb["delta"]))
    dim = len(next(iter(lookup.values())))

    d = serial.copy()
    if index == "last":
        last_t = d.groupby("patient_id")["time_years"].transform("max")
        d = d[np.isclose(d["time_years"], last_t)]
    elif index == "max":
        idx = d.groupby(["patient_id", "mut_key"])["vaf"].idxmax()
        d = d.loc[idx]
    else:
        raise ValueError("index must be 'last' or 'max'")

    # One weight per mutation even if multiple genomic calls collapse to same mut_key.
    d = (d.groupby(["patient_id", "mut_key"], as_index=False)
           .agg(vaf=("vaf", "max"), gene=("gene", "first")))

    rows = []
    cov_rows = []
    for pid in patient_order:
        p = d[d["patient_id"] == pid]
        total = int(p["mut_key"].nunique())
        hit = p[p["mut_key"].isin(lookup)]
        n_hit = int(hit["mut_key"].nunique())
        if n_hit:
            vecs = np.vstack([lookup[k] for k in hit["mut_key"]])
            w = hit["vaf"].to_numpy(float)
            w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
            vec = np.average(vecs, axis=0, weights=w).astype(np.float32)
        else:
            vec = np.zeros(dim, dtype=np.float32)
        rows.append(vec)
        cov_rows.append({
            "patient_id": pid,
            "n_mutations_at_index": total,
            "n_bionemo_mutations": n_hit,
            "bionemo_coverage": (n_hit / total) if total else 0.0,
            "has_embedding": int(n_hit > 0),
        })
    return np.vstack(rows), pd.DataFrame(cov_rows)


def audit_serial(serial: pd.DataFrame) -> dict:
    """Compact, JSON-safe audit summary for reproducibility."""
    if serial.empty:
        return {"rows": 0, "patients": 0, "variants": 0}
    by_cohort = {}
    for c, g in serial.groupby("cohort"):
        by_cohort[str(c)] = {
            "rows": int(len(g)),
            "patients": int(g["patient_id"].nunique()),
            "variants": int(g[["patient_id", "variant_id"]].drop_duplicates().shape[0]),
            "median_timepoints_per_variant": float(
                g.groupby(["patient_id", "variant_id"])["time_years"].nunique().median()
            ),
            "median_span_years": float(
                g.groupby(["patient_id", "variant_id"])["time_years"].agg(lambda x: x.max()-x.min()).median()
            ),
        }
    return {
        "rows": int(len(serial)),
        "patients": int(serial["patient_id"].nunique()),
        "variants": int(serial[["patient_id", "variant_id"]].drop_duplicates().shape[0]),
        "missense_rows": int((serial["kind"] == "missense").sum()),
        "cohorts": by_cohort,
    }
