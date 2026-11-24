"""
longitudinal.py
Turning a follow-up cohort into a supervised early-detection problem.

The modelling is the easy part. Almost every error in this kind of study is made
before any model is fitted, when the label is defined. Four traps, each of which
produces a good-looking AUC that means nothing:

1. CENSORING COLLAPSED INTO A BINARY LABEL
   A participant who did not progress in 2 years is not the same as one who did
   not progress in 15. Calling both "0" discards the difference and biases the
   estimate toward whoever happened to be followed longest.

2. IMMORTAL TIME BIAS
   If a case must survive long enough to be sequenced, and controls are not held
   to the same requirement, the model learns survival, not biology. Fix by
   landmarking: pick a time origin, require everyone to be event-free at it, and
   predict forward from there.

3. AGE AS A HIDDEN LABEL
   CHIP prevalence rises steeply with age, and so does myeloid neoplasm
   incidence. A model given mutations but not age will happily rediscover age.
   Include it explicitly, or match on it, and report performance both ways.

4. NO BLOOD COUNTS
   Published work (CHRS; the 2025 multi-biobank analysis) finds that developing
   a cytopenia is close to a required step on the path from CHIP to myeloid
   neoplasm. A pipeline with no slot for complete blood counts is missing the
   strongest available predictor and will understate what is achievable.
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def make_landmark_labels(
    df: pd.DataFrame,
    time_col: str = "years_to_event_or_censor",
    event_col: str = "event",
    horizon: float = 5.0,
    min_followup: float = None,
) -> pd.DataFrame:
    """
    Convert time-to-event follow-up into a clean binary task at a fixed horizon.

    label = 1 : event occurred within `horizon` years
    label = 0 : event-free AND followed for at least `horizon` years
    dropped   : censored before `horizon` - their outcome is unknown, and
                silently calling them 0 is the single most common way this
                analysis goes wrong.

    Returns the frame with `label` added and unusable rows removed.
    """
    min_followup = horizon if min_followup is None else min_followup
    d = df.copy()

    event_within = (d[event_col] == 1) & (d[time_col] <= horizon)
    clean_control = (d[event_col] == 0) & (d[time_col] >= min_followup)
    late_event = (d[event_col] == 1) & (d[time_col] > horizon)

    keep = event_within | clean_control | late_event
    dropped = int((~keep).sum())

    d = d[keep].copy()
    d["label"] = event_within[keep].astype(int)

    print(f"[landmark] horizon={horizon}y  ->  {int(d.label.sum())} cases / "
          f"{int((1 - d.label).sum())} controls  ({d.label.mean():.2%} positive)")
    if dropped:
        print(f"[landmark] dropped {dropped} censored before {min_followup}y - "
              f"outcome unknown, NOT counted as controls")
    if late_event.sum():
        print(f"[landmark] {int(late_event.sum())} events after the horizon are "
              f"treated as controls (correct: they did not progress within {horizon}y)")
    return d


def check_immortal_time(df: pd.DataFrame, time_col: str = "years_to_event_or_censor",
                        event_col: str = "event",
                        entry_col: Optional[str] = None) -> None:
    """
    Warn when cases and controls have systematically different follow-up.
    Equal medians are what you want; a large gap means the label is partly a
    function of how long someone was observed.
    """
    g = df.groupby(event_col)[time_col].agg(["count", "median", "min", "max"])
    print("\n[immortal-time] follow-up by outcome:")
    print(g.round(2).to_string())
    if len(g) > 1:
        ratio = g["median"].max() / max(g["median"].min(), 1e-9)
        if ratio > 1.5:
            print(f"  !! median follow-up differs {ratio:.1f}x between cases and "
                  f"controls.\n     Landmark the analysis, or the model learns "
                  f"observation time.")
        else:
            print(f"  ok: follow-up comparable ({ratio:.2f}x)")
    if entry_col and entry_col in df.columns:
        e = df.groupby(event_col)[entry_col].median()
        print(f"  entry time by outcome: {e.round(2).to_dict()}")


def age_matched_subsample(df: pd.DataFrame, label_col: str = "label",
                          age_col: str = "age", ratio: int = 4,
                          bin_width: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Draw `ratio` age-matched controls per case.

    Use this as a SECOND analysis, not the primary one. If performance collapses
    once age is matched out, then age was carrying the signal - which is worth
    knowing, and worth reporting, before anyone calls the model a biomarker.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["_bin"] = (d[age_col] // bin_width).astype(int)
    cases = d[d[label_col] == 1]
    picked = [cases]
    for b, n in cases.groupby("_bin").size().items():
        pool = d[(d[label_col] == 0) & (d["_bin"] == b)]
        take = min(len(pool), n * ratio)
        if take:
            picked.append(pool.sample(take, random_state=int(rng.integers(1e6))))
    out = pd.concat(picked).drop(columns="_bin").reset_index(drop=True)
    print(f"[age-match] {int(out[label_col].sum())} cases + "
          f"{int((1-out[label_col]).sum())} controls, {bin_width}y bins")
    return out


# Complete blood count fields worth carrying. Cytopenia is close to an obligatory
# step from CHIP to myeloid neoplasm, so these are not optional extras.
CBC_FIELDS = [
    "hemoglobin", "mcv", "platelet_count", "wbc_count",
    "neutrophil_count", "lymphocyte_count", "rdw",
]


def add_cbc_features(patients: pd.DataFrame, cbc: pd.DataFrame,
                     fields: Sequence[str] = CBC_FIELDS,
                     on: str = "patient_id") -> pd.DataFrame:
    """
    Merge blood counts and derive simple cytopenia flags.

    Report the model both with and without these. Mutations alone is the
    interesting scientific question; mutations plus blood counts is the
    clinically useful model. They are different claims.
    """
    have = [f for f in fields if f in cbc.columns]
    missing = set(fields) - set(have)
    if missing:
        print(f"[cbc] not available: {sorted(missing)}")
    out = patients.merge(cbc[[on] + have], on=on, how="left")

    # WHO/ICC cytopenia thresholds
    if "hemoglobin" in have:
        out["anemia"] = (out.hemoglobin < 12).astype("Int64")
    if "platelet_count" in have:
        out["thrombocytopenia"] = (out.platelet_count < 150).astype("Int64")
    if "neutrophil_count" in have:
        out["neutropenia"] = (out.neutrophil_count < 1.8).astype("Int64")

    flags = [c for c in ["anemia", "thrombocytopenia", "neutropenia"] if c in out]
    if flags:
        out["n_cytopenias"] = out[flags].sum(axis=1)
        print(f"[cbc] merged {len(have)} counts; "
              f"{int((out.n_cytopenias > 0).sum())} patients have >=1 cytopenia")
    return out


def screening_ppv(sensitivity: float, specificity: float,
                  prevalence: float) -> dict:
    """
    Translate sens/spec into what a screening programme would actually see.

    At 1% prevalence a test with 98% specificity still produces roughly two
    false alarms for every true one. This is the number to put on the slide,
    not the AUC.
    """
    tp = prevalence * sensitivity
    fp = (1 - prevalence) * (1 - specificity)
    fn = prevalence * (1 - sensitivity)
    tn = (1 - prevalence) * specificity
    return {
        "prevalence": prevalence,
        "PPV": tp / max(tp + fp, 1e-12),
        "NPV": tn / max(tn + fn, 1e-12),
        "false_alarms_per_true_case": fp / max(tp, 1e-12),
        "people_flagged_per_1000": 1000 * (tp + fp),
        "cases_missed_per_1000": 1000 * fn,
    }
