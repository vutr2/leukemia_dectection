"""
fitness.py
Clonal fitness from longitudinal VAF, calibrated against synonymous mutations.

WHY FITNESS AND NOT A BINARY LABEL
Progression from CHIP to myeloid neoplasm is rare (~0.5-1%/year), so a binary
"will progress" label wastes almost every participant. Clonal fitness - the
exponential growth rate of a mutant clone - is a continuous quantity measurable
in EVERY carrier, whether or not they ever fall ill. Same cohort, orders of
magnitude more signal. And it is mechanistic rather than a convenient proxy:
a clone that expands is a clone on its way to becoming a disease.

    s = d(log VAF)/dt        # per year

THE MEASUREMENT PROBLEM, AND THE FIX
VAF is noisy, and the noise is worst exactly where the clones are smallest.
Fitting a slope through low-VAF points mostly fits sequencing error.

Synonymous mutations solve this. They do not change the amino acid, so their
true fitness is zero by construction. Whatever spread you measure in their
slopes IS your noise floor. Every threshold in this module should be chosen by
looking at that floor, not by intuition.

Measured on the Lothian Birth Cohorts (GSE178936):

    filter                          synonymous sd    above 95th pct
    VAF>=0.005, >=2 waves, DP>=200      0.104            7.3%   <- mostly noise
    VAF>=0.01,  >=3 waves, DP>=500      0.034           21.8%   <- real signal

At the tighter filter the known drivers emerge unprompted, with growth rates
matching the published literature:

    TET2    0.053/yr      JAK2  0.040/yr      DNMT3A  0.036/yr

That agreement is the validation. If your top genes are not the canonical CHIP
drivers, your filter is too loose and you are ranking noise.
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# LBC waves are spaced roughly three years apart.
YEARS_PER_WAVE = 3.0

CANONICAL_DRIVERS = ["DNMT3A", "TET2", "ASXL1", "JAK2", "TP53", "PPM1D",
                     "SF3B1", "SRSF2", "U2AF1"]


def load_many(dirs, threshold: str = "1PCT") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pool several GEO releases. GSE178936 (85 participants) and the LBC file in
    GSE288742 (63 participants) share a column layout and have ZERO participant
    overlap, so they simply concatenate — more variants, same measurement
    process, same noise model.

    Always re-run fitness_report() on the pooled set. Pooling cohorts that were
    sequenced differently would break the shared noise floor; these two were not.
    """
    ns_all, sy_all = [], []
    for d in dirs:
        a, b = load_lbc(d, threshold)
        ns_all.append(a); sy_all.append(b)
    ns = pd.concat(ns_all, ignore_index=True)
    sy = pd.concat(sy_all, ignore_index=True)
    # A vid legitimately repeats once per wave, so row duplicates mean nothing.
    # What would be a real problem is the SAME participant appearing in two
    # releases - that would double-count them in every downstream estimate.
    seen, overlap = set(), set()
    for d in ns_all:
        overlap |= seen & set(d.participant_id)
        seen |= set(d.participant_id)
    print(f"[pool] {len(ns):,} non-syn / {len(sy):,} syn rows, "
          f"{ns.participant_id.nunique()} participants"
          + (f"  !! {len(overlap)} participants appear in more than one release"
             if overlap else ""))
    return ns, sy


def load_lbc(data_dir: str = "data/raw/lbc_gse178936",
             threshold: str = "1PCT") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the GEO GSE178936 tables (non-synonymous, synonymous).

    Take the 1PCT files, not 2PCT: more trajectories, and you filter yourself
    afterwards using the synonymous noise floor rather than someone else's
    arbitrary cut.
    """
    def _find(kind):
        # "synonymous" is a substring of "non-synonymous" - match on the exact
        # token or you silently load the same file twice.
        for f in sorted(os.listdir(data_dir)):
            fl = f.lower()          # GSE288742 uses UPPERCASE in filenames
            if threshold.lower() not in fl or not fl.endswith(".tsv"):
                continue
            is_non = "non-synonymous" in fl or "non_synonymous" in fl
            if (kind == "non-synonymous") == is_non:
                return os.path.join(data_dir, f)
        raise FileNotFoundError(
            f"No {threshold} {kind} .tsv in {data_dir}. Expected files like "
            f"GSE178936_LBC_ARCHER_{threshold}_VAF_Feb22_{kind}.tsv")

    out = []
    for kind in ("non-synonymous", "synonymous"):
        d = pd.read_csv(_find(kind), sep="\t", low_memory=False)
        d.columns = [c.strip() for c in d.columns]
        # Identify a variant by GENOMIC coordinates, not by protein notation.
        # ~15% of rows (splice, UTR) have no protein_substitution, and how a
        # missing value behaves in string concatenation is pandas-version
        # dependent: newer pandas keeps NA (groupby then silently DROPS those
        # trajectories), older pandas coerces to the literal "nan" (which MERGES
        # every distinct variant in that gene into one fake trajectory). Both
        # are wrong, and they give different totals on the same file.
        d["vid"] = (d.participant_id.astype(str) + "|"
                    + d.chromosome.astype(str) + ":" + d.position.astype(str)
                    + ":" + d.reference.astype(str) + ">" + d.mutation.astype(str))
        d["cohort"] = d.participant_id.str.extract(r"(LBC\d+)")[0]
        out.append(d)
        print(f"[lbc] {kind}: {len(d):,} rows, {d.participant_id.nunique()} participants")
    return out[0], out[1]


def estimate_fitness(
    df: pd.DataFrame,
    min_af: float = 0.01,
    min_dp: int = 500,
    min_waves: int = 3,
    years_per_wave: float = YEARS_PER_WAVE,
) -> pd.DataFrame:
    """
    One row per (participant, variant): the exponential growth rate `s`.

    Defaults are the filter validated on LBC. Loosening them does not give you
    more data, it gives you more noise — verify with noise_floor() before you
    change them.
    """
    d = df[(df.AF >= min_af) & (df.DP >= min_dp)]
    rows = []
    for vid, g in d.groupby("vid"):
        if g.wave.nunique() < min_waves:
            continue
        w = g.groupby("wave").AF.mean()
        t = (w.index.values - w.index.values[0]) * years_per_wave
        s, intercept = np.polyfit(t, np.log(w.values), 1)
        resid = np.log(w.values) - (s * t + intercept)
        rows.append({
            "vid": vid,
            "participant_id": g.participant_id.iloc[0],
            "cohort": g.cohort.iloc[0],
            "gene": g.PreferredSymbol.iloc[0],
            "protein_substitution": g.protein_substitution.iloc[0],
            "consequence": g.consequence.iloc[0],
            "n_waves": int(g.wave.nunique()),
            "mean_af": float(w.mean()),
            "max_af": float(w.max()),
            "median_dp": float(g.DP.median()),
            "s": float(s),
            "rmse": float(np.sqrt((resid ** 2).mean())),
        })
    out = pd.DataFrame(rows)
    print(f"[fitness] {len(out):,} trajectories (VAF>={min_af}, DP>={min_dp}, "
          f">={min_waves} waves)")
    return out


def noise_floor(syn_fitness: pd.DataFrame, pct: float = 95) -> Dict[str, float]:
    """Synonymous slopes define zero. Everything else is measured against them."""
    s = syn_fitness["s"]
    f = {"n": len(s), "mean": float(s.mean()), "sd": float(s.std()),
         f"p{int(pct)}": float(s.quantile(pct / 100)),
         f"p{int(100-pct)}": float(s.quantile(1 - pct / 100))}
    print(f"[noise] synonymous n={f['n']}, mean={f['mean']:+.4f} "
          f"(should be ~0), sd={f['sd']:.4f}, p{int(pct)}={f[f'p{int(pct)}']:+.4f}")
    if abs(f["mean"]) > 0.5 * f["sd"]:
        print("  !! synonymous slopes are not centred on zero — suspect a "
              "systematic drift in coverage or purity across waves.")
    return f


def fitness_report(ns_fit: pd.DataFrame, syn_fit: pd.DataFrame,
                   min_traj: int = 5) -> Dict:
    """
    The three checks that tell you whether the fitness estimates are usable.
    Run this before feeding `s` into any model.
    """
    print("=" * 66)
    print("FITNESS CALIBRATION")
    print("=" * 66)
    nf = noise_floor(syn_fit)
    thr = nf["p95"]

    frac = float((ns_fit["s"] > thr).mean())
    print(f"\n[1] Non-synonymous above the synonymous 95th percentile: "
          f"{frac:.1%}  (5% = chance)")
    if frac < 0.10:
        print("    !! barely above chance. Tighten min_af / min_waves; you are "
              "fitting slopes through measurement error.")
    else:
        print(f"    ok: {frac/0.05:.1f}x enrichment over chance.")

    print("\n[2] Canonical CHIP drivers — these must surface if the estimates "
          "are real:")
    d = (ns_fit[ns_fit.gene.isin(CANONICAL_DRIVERS)]
         .groupby("gene").s.agg(["count", "mean"])
         .sort_values("mean", ascending=False))
    print(d.round(4).to_string() if len(d) else "    (none present)")
    print("    Published growth rates for common drivers land near "
          "0.03-0.15/yr.")

    print(f"\n[3] Gene ranking (>= {min_traj} trajectories):")
    g = (ns_fit.groupby("gene").s.agg(["count", "mean", "std"])
         .query(f"count >= {min_traj}").sort_values("mean", ascending=False))
    print(g.head(12).round(4).to_string())
    top = set(g.head(5).index)
    if not top & set(CANONICAL_DRIVERS):
        print("\n    !! no canonical driver in the top 5. Almost always means "
              "the VAF floor is too low.")
    print("=" * 66)
    return {"noise": nf, "enrichment": frac / 0.05, "gene_ranking": g}


def missense_targets(ns_fit: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse to unique protein variants — the rows an ESM2 regression can use.

    A variant seen in three people gives three fitness estimates of the same
    underlying quantity; average them and keep the count as a weight. The
    number of ROWS here, not the number of trajectories, is your real n.
    """
    mis = ns_fit[ns_fit.consequence.astype(str)
                 .str.contains("missense", case=False, na=False)]
    agg = {"s": ("s", "mean"), "s_sd": ("s", "std"), "n_obs": ("s", "size"),
           "mean_af": ("mean_af", "mean")}
    # Carry the cross-cohort z-score through if standardise_within_cohort ran.
    if "s_z" in mis.columns:
        agg["s_z"] = ("s_z", "mean")
    if "cohort_src" in mis.columns:
        agg["cohort_src"] = ("cohort_src", "first")
    out = (mis.groupby(["gene", "protein_substitution"]).agg(**agg).reset_index())
    print(f"[targets] {len(mis):,} missense trajectories -> {len(out):,} unique "
          f"protein variants")
    if len(out) >= 3:
        r_det = float(np.tanh((1.959964 + 0.8416212) / np.sqrt(len(out) - 3)))
        print(f"  at n={len(out)}, 80% power to detect r >= {r_det:.3f}")
        if r_det > 0.3:
            print(f"  !! underpowered for a moderate effect. Pool another "
                  f"cohort before treating a null as a finding.")
    return out


def required_n(effect_r: float = 0.3, power: float = 0.8, alpha: float = 0.05) -> int:
    """
    Sample size to detect a correlation of `effect_r`. Run this BEFORE the
    experiment, not after — it is how you tell a real negative result from
    an underpowered one.
    """
    from math import atanh, sqrt
    z_a, z_b = 1.959964, 0.8416212 if power == 0.8 else 1.281552
    n = ((z_a + z_b) / atanh(effect_r)) ** 2 + 3
    print(f"[power] detecting r={effect_r} at {power:.0%} power needs "
          f"n≈{int(np.ceil(n))} unique variants")
    return int(np.ceil(n))


def load_sardinia(path: str = "data/raw/sardinia/41586_2022_4785_MOESM5_ESM.xlsx"
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fabre et al. 2022 (Nature), Supplementary Table 2 — the SardiNIA cohort.
    350 participants, up to 5 samplings over a median 12.9 years, 56 genes at
    ~1065x mean coverage.

    Two things make this better than the LBC release, not merely bigger:

    - It carries AGE at each sampling, so time between points is real rather
      than the "3 years per wave" approximation LBC forces on you.
    - It reaches back to age 54, whereas LBC starts at 70. Growth rates are
      age-dependent (DNMT3A expands early and slows in old age; splicing-factor
      mutations only take off later), so the two cohorts cover different parts
      of the curve.

    Returned in the same shape as load_lbc() so the two can be pooled: the
    header sits on row 4 of the sheet, and `Study Phase` stands in for `wave`.
    """
    d = pd.read_excel(path, header=3)
    d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
    d = d.dropna(subset=["Sample ID", "VAF"])

    out = pd.DataFrame({
        "participant_id": d["Sample ID"].astype(str),
        "wave": d["Study Phase"].astype(int),
        "age": d["Age"].astype(float),
        "PreferredSymbol": d["Gene"].astype(str),
        "protein_substitution": d["Protein"].astype(str),
        "AF": d["VAF"].astype(float),
        "consequence": d["Effect"].astype(str).str.lower().str.replace(" ", "_"),
        # No per-variant depth is published; the paper reports ~1065x mean
        # coverage, so a constant placeholder keeps the depth filter inert
        # rather than silently dropping everything.
        "DP": 1065.0,
        "cohort": "SardiNIA",
    })
    # Same rule as load_lbc: genomic coordinates, never protein notation.
    out["vid"] = (out.participant_id + "|" + d["Chromosome"].astype(str) + ":"
                  + d["Start"].astype(str) + ":" + d["WT"].astype(str) + ">"
                  + d["MT"].astype(str)).values

    syn = out[out.consequence.str.contains("synonymous|silent", na=False)]
    ns = out[~out.index.isin(syn.index)]
    print(f"[sardinia] {len(ns):,} non-syn / {len(syn):,} syn rows, "
          f"{out.participant_id.nunique()} participants, "
          f"age {out.age.min():.0f}-{out.age.max():.0f}")
    return ns.copy(), syn.copy()


def estimate_fitness_by_age(df: pd.DataFrame, min_af: float = 0.01,
                            min_dp: int = 0, min_waves: int = 3) -> pd.DataFrame:
    """
    Same as estimate_fitness, but regresses on actual age rather than assuming
    equal spacing between sampling waves. Use this whenever the table carries
    an age or date column — unequal gaps otherwise bias every slope.
    """
    d = df[(df.AF >= min_af) & (df.DP >= min_dp)]
    rows = []
    for vid, g in d.groupby("vid"):
        if g.wave.nunique() < min_waves:
            continue
        g = g.sort_values("age")
        t = g.age.values - g.age.values[0]
        if t[-1] <= 0:
            continue
        s, b = np.polyfit(t, np.log(g.AF.values), 1)
        resid = np.log(g.AF.values) - (s * t + b)
        rows.append({
            "vid": vid, "participant_id": g.participant_id.iloc[0],
            "cohort": g.cohort.iloc[0], "gene": g.PreferredSymbol.iloc[0],
            "protein_substitution": g.protein_substitution.iloc[0],
            "consequence": g.consequence.iloc[0], "n_waves": int(g.wave.nunique()),
            "mean_af": float(g.AF.mean()), "max_af": float(g.AF.max()),
            "age_start": float(g.age.iloc[0]), "years_spanned": float(t[-1]),
            "median_dp": float(g.DP.median()),
            "s": float(s), "rmse": float(np.sqrt((resid ** 2).mean())),
        })
    out = pd.DataFrame(rows)
    print(f"[fitness/age] {len(out):,} trajectories "
          f"(VAF>={min_af}, >={min_waves} timepoints, median span "
          f"{out.years_spanned.median():.1f}y)" if len(out) else "[fitness/age] none")
    return out


def standardise_within_cohort(fits: Dict[str, pd.DataFrame],
                              syns: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pool cohorts by expressing every fitness estimate in units of its OWN
    cohort's noise floor:

        z = (s - synonymous_mean) / synonymous_sd

    Why not just concatenate: on LBC the synonymous slopes sit at -0.005 with
    sd 0.029; on SardiNIA they sit at +0.012 with sd 0.073. Different centre,
    2.5x different spread. Raw pooling would let a model separate the cohorts
    on that offset alone and call it biology - the same confound `leakage_report`
    catches when you merge two sequencing studies.

    A caveat you must report: SardiNIA's synonymous slopes are not centred on
    zero at any VAF floor. Its supplementary table lists only variants that were
    detected, and a growing clone is likelier to be detected, so the synonymous
    set is itself enriched for growth. The z-scores correct the offset; they do
    not undo the ascertainment.

    A cleaner alternative when you can afford it: fit within each cohort
    separately and check the two answers agree. Agreement across cohorts with
    different assays is stronger evidence than one pooled number.
    """
    out = []
    for name, F in fits.items():
        S = syns[name]
        if len(S) < 10:
            print(f"  !! {name}: only {len(S)} synonymous trajectories - noise "
                  f"floor unreliable, z-scores are approximate")
        mu, sd = S.s.mean(), S.s.std()
        f = F.copy()
        f["cohort_src"] = name
        f["s_raw"] = f["s"]
        f["s_z"] = (f["s"] - mu) / sd
        print(f"  [{name}] n={len(f):4d}  noise mean={mu:+.4f} sd={sd:.4f}")
        out.append(f)
    pooled = pd.concat(out, ignore_index=True)
    print(f"[standardise] pooled {len(pooled):,} trajectories from "
          f"{len(fits)} cohorts; use column `s_z`, not `s`")
    return pooled
