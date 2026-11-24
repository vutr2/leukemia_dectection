"""
diagnostics.py
Study-design checks to run BEFORE training anything.

The failure mode these catch: cases downloaded from an AML study, controls from
a separate CH study. Any systematic difference between the two files — gene
panel, sequencing depth, VAF floor, annotation pipeline — becomes a perfect
label predictor. You get AUC 0.95 and learn nothing about early detection.

Run `leakage_report` on the cleaned mutation table. If it prints a warning,
fix the study design; no amount of model work will rescue it.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


def cohort_label_crosstab(mut: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Patients per (cohort, label). A diagonal table means cohort == label."""
    pc = mut.groupby("patient_id")["cohort"].first().rename("cohort")
    j = labels[["patient_id", "label"]].drop_duplicates("patient_id").merge(
        pc, on="patient_id", how="left")
    return pd.crosstab(j["cohort"], j["label"])


def gene_frequency_by_group(mut: pd.DataFrame, labels: pd.DataFrame,
                            group_col: str = "label") -> pd.DataFrame:
    """
    Per gene: fraction of patients in each group carrying a mutation in it,
    plus how exclusive it is to one group.
    """
    pat = mut[["patient_id", "gene"]].drop_duplicates()
    j = pat.merge(labels[["patient_id", group_col]], on="patient_id", how="inner")
    n = labels.groupby(group_col)["patient_id"].nunique()

    tab = (j.groupby(["gene", group_col])["patient_id"].nunique()
             .unstack(fill_value=0))
    for g in tab.columns:
        tab[g] = tab[g] / n.get(g, 1)
    tab.columns = [f"freq_{c}" for c in tab.columns]

    vals = tab.to_numpy()
    tot = vals.sum(axis=1)
    # 1.0 = gene appears in exactly one group, 0.0 = perfectly balanced
    tab["exclusivity"] = np.where(tot > 0, vals.max(axis=1) / np.maximum(tot, 1e-9), 0)
    tab["exclusivity"] = (tab["exclusivity"] - 1 / vals.shape[1]) / (1 - 1 / vals.shape[1])
    # How many patients carry this gene at all - exclusivity on 2 patients is noise.
    tab["n_patients"] = j.groupby("gene")["patient_id"].nunique()
    return tab.sort_values(["exclusivity", "n_patients"], ascending=[False, False])


def depth_by_group(mut: pd.DataFrame, labels: pd.DataFrame,
                   group_col: str = "label") -> pd.DataFrame:
    """Sequencing depth and VAF floor per group. These should overlap."""
    # mut may already carry the group column (e.g. a per-row label); drop it so
    # the merge below does not create label_x / label_y and lose the name.
    d = mut.drop(columns=[group_col], errors="ignore").copy()
    if {"alt_count", "ref_count"}.issubset(d.columns):
        d["depth"] = d["alt_count"].fillna(0) + d["ref_count"].fillna(0)
    else:
        d["depth"] = np.nan
    j = d.merge(labels[["patient_id", group_col]], on="patient_id", how="inner")
    return j.groupby(group_col).agg(
        n_mutations=("gene", "size"),
        median_depth=("depth", "median"),
        min_vaf=("vaf", "min"),
        median_vaf=("vaf", "median"),
        n_genes=("gene", "nunique"),
    )


def leakage_report(mut: pd.DataFrame, labels: pd.DataFrame,
                   exclusivity_threshold: float = 0.90,
                   min_gene_patients: int = 10,
                   top_n: int = 10) -> dict:
    """
    Three checks, printed plainly. Returns the underlying tables.
    """
    out = {}
    print("=" * 68)
    print("STUDY DESIGN CHECK")
    print("=" * 68)

    # 1. cohort vs label
    if "cohort" in mut.columns:
        ct = cohort_label_crosstab(mut, labels)
        out["cohort_label"] = ct
        print("\n[1] Cohort x label\n")
        print(ct)
        base = ct.sum(axis=0).max() / ct.to_numpy().sum()   # majority class overall
        purity = (ct.max(axis=1) / ct.sum(axis=1)).rename("purity")
        worst = purity.max()

        if ct.shape[0] < 2:
            print(f"\n  single source — nothing to compare. Base rate {base:.0%}.")
            print("  This check only becomes meaningful once you merge two studies.")
            worst = 0.0
        else:
            print("\n  label purity per cohort "
                  f"(base rate if cohort were uninformative: {base:.0%}):")
            print("   " + purity.round(3).to_string().replace("\n", "\n   "))
            # Only excess purity over the base rate indicates confounding.
            worst = 0.0 if worst <= base + 0.02 else worst

        if worst >= 0.90:
            print(f"\n  !! A cohort is {worst:.0%} one label. Cohort is effectively the label.")
            print("     Any file-level difference between the two sources — panel,")
            print("     depth, annotation pipeline — is now a perfect predictor.")
            print("     Fix this in the data. No model change will rescue it.")
        elif worst >= 0.75:
            print(f"\n  ~  Partial confounding ({worst:.0%}). Report leave-one-cohort-out")
            print("     performance alongside pooled CV.")
        else:
            print("\n  ok: labels are well mixed within cohorts.")
    else:
        print("\n[1] No `cohort` column — cannot check. Tag sources in load_maf().")

    # 2. gene exclusivity
    gf = gene_frequency_by_group(mut, labels)
    out["gene_freq"] = gf
    print(f"\n[2] Most label-exclusive genes (1.0 = appears in one class only)\n")
    print(gf.head(top_n).round(3))
    bad = gf[(gf["exclusivity"] >= exclusivity_threshold)
             & (gf["n_patients"] >= min_gene_patients)]
    rare = gf[(gf["exclusivity"] >= exclusivity_threshold)
              & (gf["n_patients"] < min_gene_patients)]
    if len(bad):
        print(f"\n  !! {len(bad)} gene(s) at exclusivity >= {exclusivity_threshold} "
              f"in >= {min_gene_patients} patients:")
        print("     " + ", ".join(bad.index[:15]))
        print("     Likely panel difference, not biology. Restrict to the gene")
        print("     intersection of both sources before training.")
    else:
        print("\n  ok: no COMMON gene is confined to a single class.")
    if len(rare):
        print(f"     ({len(rare)} rare genes are class-exclusive but seen in "
              f"< {min_gene_patients} patients - that is small-N noise, ignore.)")

    # 3. assay depth
    dep = depth_by_group(mut, labels)
    out["depth"] = dep
    print("\n[3] Assay characteristics per class\n")
    print(dep.round(3))
    if dep["median_depth"].notna().all() and len(dep) > 1:
        ratio = dep["median_depth"].max() / max(dep["median_depth"].min(), 1)
        if ratio > 1.5:
            print(f"\n  !! Median depth differs {ratio:.1f}x between classes.")
            print("     Raise min_depth until the distributions overlap.")
        else:
            print(f"\n  ok: depth comparable ({ratio:.2f}x).")

    print("\n" + "=" * 68)
    return out


def shared_gene_panel(mut: pd.DataFrame, labels: pd.DataFrame,
                      min_patients: int = 3) -> list:
    """Genes seen in BOTH classes often enough to be usable. Train on these."""
    pat = mut[["patient_id", "gene"]].drop_duplicates()
    j = pat.merge(labels[["patient_id", "label"]], on="patient_id", how="inner")
    counts = j.groupby(["gene", "label"])["patient_id"].nunique().unstack(fill_value=0)
    keep = counts[(counts >= min_patients).all(axis=1)].index.tolist()
    print(f"[panel] {len(keep)}/{len(counts)} genes present in both classes "
          f"with >= {min_patients} patients each")
    return sorted(keep)
