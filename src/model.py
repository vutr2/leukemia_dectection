"""
model.py
Feature assembly, baselines, and evaluation framed for a screening test.

Evaluation note: raw accuracy is meaningless here. In any realistic screening
population the positive rate is ~1%, so "always predict 0" scores 99%. The
number that matters is sensitivity at a FIXED high specificity (98-99%), i.e.
how many true pre-leukaemic cases you catch at a tolerable false-alarm rate.
That is the metric a reviewer will ask about.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class EmbeddingPCA(BaseEstimator, TransformerMixin):
    """
    Compress the trailing `n_emb` columns (raw ESM2 deltas) with PCA, leaving
    the leading clinical + gene columns untouched.

    This exists so PCA is fitted INSIDE each CV fold. Fitting it on the full
    matrix before splitting lets the test fold influence the components - mild,
    because PCA never sees the labels, but it is still information crossing the
    split, and it is the kind of thing a reviewer will look for.
    """

    def __init__(self, n_emb: int = 0, n_components: int = 32):
        self.n_emb = n_emb
        self.n_components = n_components

    def fit(self, X, y=None):
        if self.n_emb > 0:
            k = min(self.n_components, X.shape[0] - 1, self.n_emb)
            self.pca_ = PCA(n_components=k, random_state=0).fit(X[:, -self.n_emb:])
        return self

    def transform(self, X):
        if self.n_emb <= 0:
            return X
        head, tail = X[:, :-self.n_emb], X[:, -self.n_emb:]
        return np.hstack([head, self.pca_.transform(tail)])


# --- Features ----------------------------------------------------------------

CLINICAL_FEATURES = ["n_mutations", "max_vaf", "mean_vaf", "sum_vaf", "n_genes"]


def gene_onehot(patients: pd.DataFrame, genes: List[str]) -> pd.DataFrame:
    """One column per panel gene: is it mutated in this patient?"""
    data = {f"gene_{g}": patients["genes"].apply(lambda gs, g=g: int(g in gs))
            for g in genes}
    return pd.DataFrame(data, index=patients.index)


def build_features(
    patients: pd.DataFrame,
    genes: List[str],
    embeddings: Optional[np.ndarray] = None,
    n_pca: int = 32,
    extra_features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], int]:
    """
    Returns (X, feature_names, n_emb) where n_emb is the count of trailing raw
    embedding columns. Pass n_emb to make_model/cross_validate so PCA is fitted
    per fold.

    Blocks: clinical burden + optional extras + gene one-hot + (optional) raw
    ESM2 deltas. 1280 raw dimensions against a few hundred patients overfits
    instantly, so the embeddings get compressed - but by the pipeline, inside
    the fold.
    """
    blocks, names = [], []

    clin = patients[CLINICAL_FEATURES].fillna(0).to_numpy(float)
    blocks.append(clin); names += CLINICAL_FEATURES

    # Age, blood counts, cytopenia flags. For the early-detection task these are
    # not optional: age drives both CHIP prevalence and myeloid neoplasm
    # incidence, and cytopenia is close to an obligatory step on the path from
    # CHIP to neoplasm. Report the model with and without them - "mutations
    # alone" and "mutations plus blood counts" are different claims.
    if extra_features:
        have = [f for f in extra_features if f in patients.columns]
        miss = set(extra_features) - set(have)
        if miss:
            print(f"[features] extra_features not in table: {sorted(miss)}")
        if have:
            blocks.append(patients[have].astype(float).fillna(
                patients[have].astype(float).median()).to_numpy())
            names += have
            print(f"[features] + {len(have)} extra: {have}")

    oh = gene_onehot(patients, genes)
    blocks.append(oh.to_numpy(float)); names += list(oh.columns)

    n_emb = 0
    if embeddings is not None:
        n_emb = embeddings.shape[1]
        blocks.append(embeddings)
        names += [f"esm_raw{i}" for i in range(n_emb)]
        print(f"[features] ESM2 {embeddings.shape} appended RAW; PCA to "
              f"{n_pca} components happens inside each CV fold, not here.")

    X = np.hstack(blocks)
    print(f"[features] X = {X.shape} ({len(names) - n_emb} named + {n_emb} embedding)")
    return X, names, n_emb


# --- Models ------------------------------------------------------------------

def make_model(kind: str = "logreg", n_emb: int = 0, n_pca: int = 32, **kw):
    """`n_emb` = number of trailing raw-embedding columns to PCA inside the fold."""
    pre = [("embpca", EmbeddingPCA(n_emb=n_emb, n_components=n_pca))] if n_emb else []

    if kind == "logreg":
        return Pipeline(pre + [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       C=kw.get("C", 1.0))),
        ])
    if kind == "gbm":
        return Pipeline(pre + [("clf", GradientBoostingClassifier(
            n_estimators=kw.get("n_estimators", 300),
            max_depth=kw.get("max_depth", 3),
            learning_rate=kw.get("learning_rate", 0.05),
            random_state=0))])
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return Pipeline(pre + [
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  early_stopping=True, random_state=0)),
        ])
    raise ValueError(f"Unknown model kind: {kind}")


# --- Evaluation --------------------------------------------------------------

def sensitivity_at_specificity(y, p, target_spec: float = 0.98) -> Tuple[float, float]:
    """Sensitivity achievable at >= target specificity, and the threshold."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.where(fpr <= 1 - target_spec)[0]
    if len(ok) == 0:
        return 0.0, float(np.nanmax(p)) + 1e-9
    i = ok[-1]
    t = thr[i]
    # roc_curve prepends an artificial inf threshold; never return it.
    if not np.isfinite(t):
        t = float(np.nanmax(p)) + 1e-9
    return float(tpr[i]), float(t)


def evaluate(y, p, target_spec: float = 0.98,
             threshold: Optional[float] = None) -> Dict[str, float]:
    """
    Pass `threshold` to apply an operating point chosen elsewhere (on training
    data). Leaving it None picks the threshold on THIS data, which makes
    specificity 0.98 true by construction and sensitivity optimistic. That is
    fine for a final fitted model, wrong inside cross-validation.
    """
    if threshold is None:
        sens, thr = sensitivity_at_specificity(y, p, target_spec)
    else:
        thr = threshold
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        f"sens@spec{int(target_spec*100)}": tp / max(tp + fn, 1),
        "threshold": thr,
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "n_pos": int(y.sum()),
        "n": int(len(y)),
    }


def cross_validate(X, y, kind: str = "logreg", n_splits: int = 5,
                   target_spec: float = 0.98, seed: int = 0,
                   n_emb: int = 0, n_pca: int = 32) -> pd.DataFrame:
    """
    Stratified CV. If a patient can appear more than once in your table, switch
    to StratifiedGroupKFold(groups=patient_id) or your AUC will be inflated.
    """
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        m = make_model(kind, n_emb=n_emb, n_pca=n_pca)
        m.fit(X[tr], y[tr])

        # Operating point comes from the training fold only.
        p_tr = m.predict_proba(X[tr])[:, 1]
        _, thr = sensitivity_at_specificity(y[tr], p_tr, target_spec)

        p_te = m.predict_proba(X[te])[:, 1]
        r = evaluate(y[te], p_te, target_spec, threshold=thr)
        r["fold"] = fold
        rows.append(r)
    df = pd.DataFrame(rows).set_index("fold")
    print(f"\n[{kind}] mean AUC = {df['auc'].mean():.3f} +/- {df['auc'].std():.3f}")
    return df


def ablation(X_base, X_full, y, kind: str = "logreg", n_emb: int = 0,
             **kw) -> pd.DataFrame:
    """
    The single most important experiment in this project: does ESM2 beat
    gene identity + VAF alone? If it does not, you do not have an AI story,
    you have a lookup table. Run this before writing any application.
    """
    a = cross_validate(X_base, y, kind, n_emb=0, **kw).mean(numeric_only=True)
    b = cross_validate(X_full, y, kind, n_emb=n_emb, **kw).mean(numeric_only=True)
    out = pd.DataFrame({"gene+VAF only": a, "with ESM2": b})
    out["delta"] = out["with ESM2"] - out["gene+VAF only"]
    return out


def save_model(model, path: str, meta: Optional[dict] = None) -> None:
    import joblib, os, json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    if meta:
        with open(path.replace(".joblib", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    print(f"[save] {path}")


# --- Honest uncertainty --------------------------------------------------------

def nested_cv(X, y, kind="logreg", n_emb=0, pca_grid=(4, 8, 16, 32),
              outer_splits=5, inner_splits=4, seed=0, target_spec=0.90):
    """
    Select n_pca inside the training fold, evaluate on the held-out fold.

    Picking n_pca by looking at cross-validated AUC and then reporting the best
    one is the same mistake as picking a threshold on the test set: the reported
    number is the maximum of several noisy draws. Nested CV pays for the choice.
    """
    y = np.asarray(y)
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    picked = []
    for tr, te in outer.split(X, y):
        best_k, best_auc = None, -np.inf
        if n_emb > 0:
            for k in pca_grid:
                inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed)
                scores = []
                for itr, ite in inner.split(X[tr], y[tr]):
                    m = make_model(kind, n_emb=n_emb, n_pca=k)
                    m.fit(X[tr][itr], y[tr][itr])
                    scores.append(roc_auc_score(y[tr][ite],
                                                m.predict_proba(X[tr][ite])[:, 1]))
                a = float(np.mean(scores))
                if a > best_auc:
                    best_auc, best_k = a, k
        else:
            best_k = 0
        picked.append(best_k)
        m = make_model(kind, n_emb=n_emb, n_pca=best_k or 32)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    if n_emb:
        print(f"[nested] n_pca chosen per outer fold: {picked}")
    return oof


def paired_bootstrap_delta(y, p_base, p_full, n_boot=2000, seed=0):
    """
    Bootstrap PATIENTS (not folds) to get a confidence interval on the AUC gain.

    Resampling folds only tells you how stable the partition is. Resampling
    patients tells you how much of the gain would survive a different sample of
    405 people, which is the question a reviewer is actually asking.

    Caveat: this conditions on ONE particular CV run, so it can report a tight
    interval around a gain that was itself a lucky fit. Always read it next to
    null_delta_distribution below.
    """
    y = np.asarray(y); rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]; idx_neg = np.where(y == 0)[0]
    deltas = []
    for _ in range(n_boot):
        # stratified resample keeps the positive count stable
        b = np.concatenate([rng.choice(idx_pos, len(idx_pos), replace=True),
                            rng.choice(idx_neg, len(idx_neg), replace=True)])
        if len(np.unique(y[b])) < 2:
            continue
        deltas.append(roc_auc_score(y[b], p_full[b]) - roc_auc_score(y[b], p_base[b]))
    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_gain_positive": float((d > 0).mean()), "n_boot": len(d)}


def null_delta_distribution(X_base, y, n_random_dims=1280, n_reps=8,
                            kind="logreg", pca_grid=(4, 8, 16, 32),
                            target_spec=0.90, seed0=100):
    """
    Calibrate the ablation against random embeddings, which carry zero
    information by construction.

    Why this is necessary: appending 1280 noise dimensions and running the whole
    nested-CV + PCA machinery does NOT reliably give a delta of exactly zero.
    On the BeatAML cohort it gives about +0.005, occasionally +0.016. Compare a
    real ESM2 gain against THIS distribution, not against zero.
    """
    y = np.asarray(y)
    deltas = []
    for i in range(n_reps):
        rng = np.random.default_rng(seed0 + i)
        Xe = np.hstack([rng.normal(size=(len(y), n_random_dims)),
                        np.ones((len(y), 1))])
        Xf = np.hstack([X_base, Xe])
        a = roc_auc_score(y, nested_cv(X_base, y, kind, n_emb=0, seed=i,
                                       target_spec=target_spec))
        b = roc_auc_score(y, nested_cv(Xf, y, kind, n_emb=Xe.shape[1],
                                       pca_grid=pca_grid, seed=i,
                                       target_spec=target_spec))
        deltas.append(b - a)
    d = np.array(deltas)
    return {"mean": float(d.mean()), "std": float(d.std()),
            "p95": float(np.percentile(d, 95)), "deltas": d}


# --- External validation -------------------------------------------------------

def external_validation(X, y, groups, test_group, kind="logreg", n_emb=0,
                        n_pca=16, target_spec=0.90):
    """
    Train on everything outside `test_group`, predict the held-out group once.

    No refitting, no threshold tuning on the test group, no peeking. This is the
    line between an internal number and one a reviewer will believe: CV tells
    you how the model does on rearrangements of the same patients, this tells
    you how it does on a group it has never seen.

    Use a split that mirrors deployment. BeatAML gives two natural ones:
      COHORT     - enrolment waves, i.e. a temporal split (train on the past,
                   test on the future). This is the more convincing of the two.
      CENTER_ID  - a site split (train on some hospitals, test on another).
    """
    y = np.asarray(y); groups = np.asarray(groups)
    te = groups == test_group
    tr = ~te
    if te.sum() == 0:
        raise ValueError(f"test_group {test_group!r} not present in groups")
    if len(np.unique(y[te])) < 2:
        raise ValueError(f"test_group {test_group!r} has only one class - "
                         "cannot compute AUC. Pick a different split.")

    m = make_model(kind, n_emb=n_emb, n_pca=n_pca)
    m.fit(X[tr], y[tr])

    # operating point from TRAINING data only
    _, thr = sensitivity_at_specificity(y[tr], m.predict_proba(X[tr])[:, 1],
                                        target_spec)
    p_te = m.predict_proba(X[te])[:, 1]
    r = evaluate(y[te], p_te, target_spec, threshold=thr)
    r.update({"test_group": test_group,
              "n_train": int(tr.sum()), "n_train_pos": int(y[tr].sum())})
    return r, p_te, np.where(te)[0]


def external_ablation(X_base, X_full, y, groups, test_group, kind="logreg",
                      n_emb=0, n_pca=16, target_spec=0.90) -> pd.DataFrame:
    """Run external_validation with and without embeddings, side by side."""
    a, _, _ = external_validation(X_base, y, groups, test_group, kind,
                                  n_emb=0, target_spec=target_spec)
    b, _, _ = external_validation(X_full, y, groups, test_group, kind,
                                  n_emb=n_emb, n_pca=n_pca, target_spec=target_spec)
    out = pd.DataFrame({"gene+VAF only": pd.Series(a), "with ESM2": pd.Series(b)})
    out["delta"] = out["with ESM2"] - out["gene+VAF only"]
    print(f"\n[external] held out {test_group!r}: "
          f"n={a['n']} ({a['n_pos']} positive), trained on {a['n_train']}")
    return out
