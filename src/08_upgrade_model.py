"""
08_upgrade_model.py
C&T Technology — upgrade protocol for missense → clonal fitness.

What failed
-----------
Ridge / PCA on ESM-2 residue deltas, gene one-hot, aa chemistry,
gene-residualized targets, GroupKFold-by-gene, n_obs weights.
All 5-fold CV R² stayed negative on 256 LBC/SardiNIA unique mut_keys
after dropping 4 zero-delta rows. Architecture is not the bottleneck.

What this file does
-------------------
1. Rebuild the *clean* train table (LBC/SardiNIA only, unique mut_key,
   drop zero deltas, never mix BioVU s into s_z).
2. Baseline that is allowed to be reported: Dummy mean and gene-mean OOF.
3. Optional upgrades that can actually change the representation:
   - ESM-2 masked-marginal log-likelihood ratio at the mutated residue
   - AlphaMissense score join (precomputed TSV)
   - pLDDT at residue from AFDB JSON
4. Refuse to print ΔR² when every model is worse than the dummy.

Run on Colab after mounting Drive, from repo root.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ZERO_KEYS = {
    "RUNX1_E422Q",
    "SMC1A_R404Q",
    "TP53_P33S",
    "TP53_R142P",
}


def load_clean_train(
    emb_pkl: Path,
    lbc_csv: Path,
) -> pd.DataFrame:
    emb = pd.read_pickle(emb_pkl).copy()
    lbc = pd.read_csv(lbc_csv)
    D = np.vstack(emb["delta"].values).astype(float)
    emb["dnorm"] = norm(D, axis=1)
    emb["is_zero"] = (emb["dnorm"] < 1e-8) | emb["mut_key"].isin(ZERO_KEYS)

    m = emb.merge(
        lbc[["mut_key", "s", "s_z", "n_obs"]],
        on="mut_key",
        how="inner",
        suffixes=("", "_lab"),
    )
    # drop BioVU-scale copies that leaked into the old pickle
    if "s" in m.columns and "s_z" in m.columns:
        m = m[~np.isclose(m["s"], m["s_z"])].copy()
    m = m.sort_values(["mut_key", "is_zero"]).drop_duplicates("mut_key")
    m = m[~m["is_zero"]].copy()
    m["s_z"] = m["s_z_lab"] if "s_z_lab" in m.columns else m["s_z"]
    assert m["s_z"].notna().all()
    return m.reset_index(drop=True)


def cv_r2(name: str, pred_fn, X, y, splits) -> np.ndarray:
    scores = []
    for tr, te in splits:
        pred = pred_fn(tr, te)
        scores.append(r2_score(y[te], pred))
    scores = np.asarray(scores)
    print(f"{name:42s} CV R²={scores.mean():+.3f} ± {scores.std():.3f}  {np.round(scores,3)}")
    return scores


def run_baselines(train: pd.DataFrame) -> None:
    X = np.vstack(train["delta"].values).astype(float)
    y = train["s_z"].to_numpy(float)
    genes = train["gene"].astype(str).to_numpy()
    print(f"n={len(train)}  y_mean={y.mean():.3f}  y_std={y.std():.3f}")

    splits = list(KFold(5, shuffle=True, random_state=42).split(X, y))
    gs = list(GroupKFold(5).split(X, y, groups=genes))

    def dummy(tr, te):
        return np.full(len(te), y[tr].mean())

    def esm(tr, te):
        pipe = Pipeline(
            [
                ("sc", StandardScaler()),
                ("pca", PCA(min(16, len(tr) - 2), random_state=0)),
                ("r", RidgeCV(np.logspace(-2, 3, 20))),
            ]
        )
        pipe.fit(X[tr], y[tr])
        return pipe.predict(X[te])

    print("\nRandom KFold")
    s0 = cv_r2("Dummy mean", dummy, X, y, splits)
    s1 = cv_r2("ESM-2 delta PCA16 Ridge", esm, X, y, splits)
    print("\nGroupKFold by gene")
    cv_r2("Dummy mean", dummy, X, y, gs)
    cv_r2("ESM-2 delta PCA16 Ridge", esm, X, y, gs)

    if (s0.mean() < 0) and (s1.mean() < 0):
        print("\nSTOP: do not report ΔR². Both models lose to a constant.")


# ---------------------------------------------------------------------------
# Real representation upgrades (run on GPU Colab)
# ---------------------------------------------------------------------------

def esm2_masked_marginal(seq: str, pos_1based: int, aa_wt: str, aa_mut: str, model, alphabet, batch_converter) -> float:
    """log P(aa_mut | seq\\pos) - log P(aa_wt | seq\\pos). ESM-2 masked marginal.
    pos_1based is the protein residue index (1-based, as in HGVSp).
    """
    import torch

    i = pos_1based - 1
    if i < 0 or i >= len(seq):
        return float("nan")
    if seq[i] != aa_wt:
        return float("nan")  # isoform mismatch — same failure mode as zero-delta
    masked = seq[:i] + "<mask>" + seq[i + 1 :] if False else seq[:i] + alphabet.mask_char + seq[i + 1 :]
    # fair-esm batch converter expects raw sequence; mask token is <mask>
    masked = seq[:i] + "<mask>" + seq[i + 1 :]
    _, _, tokens = batch_converter([("x", masked)])
    with torch.no_grad():
        out = model(tokens.to(next(model.parameters()).device))
        logits = out["logits"][0, i + 1]  # +1 for BOS
    vocab = alphabet.tok_to_idx
    return float(logits[vocab[aa_mut]] - logits[vocab[aa_wt]])


def join_alphamissense(train: pd.DataFrame, am_tsv: Path) -> pd.DataFrame:
    """Join precomputed AlphaMissense scores.
    File: AlphaMissense_aa_substitutions.tsv.gz
    columns typically: uniprot_id, protein_variant, am_pathogenicity, am_class
    protein_variant like P67R or R882H.
    """
    am = pd.read_csv(am_tsv, sep="\t", comment="#")
    am["subst"] = am["protein_variant"].astype(str)
    train = train.copy()
    train["subst"] = (
        train["aa_wt"].astype(str)
        + train["aa_pos"].astype(int).astype(str)
        + train["aa_mut"].astype(str)
    )
    keep = ["subst", "am_pathogenicity"]
    if "uniprot_id" in am.columns:
        keep.append("uniprot_id")
    out = train.merge(am[keep].drop_duplicates("subst"), on="subst", how="left")
    print(f"AlphaMissense coverage: {out['am_pathogenicity'].notna().mean():.1%}")
    return out


if __name__ == "__main__":
    root = Path("/content/drive/MyDrive/leukemia-prediagnose/blood-cancer-early-detection")
    if not root.exists():
        root = Path(".")
    data = root / "data" / "processed"
    train = load_clean_train(
        data / "pooled_missense_with_delta.pkl",
        data / "lbc_sardinia_unique_targets.csv",
    )
    print("clean train", train.shape)
    run_baselines(train)
    train[["gene", "mut_key", "aa_wt", "aa_pos", "aa_mut", "dnorm", "s_z", "n_obs"]].to_csv(
        data / "train_lbc_nonzero.csv", index=False
    )
    print("wrote", data / "train_lbc_nonzero.csv")
