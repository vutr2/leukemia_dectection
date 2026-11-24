"""LeukemiPrediag v2.3 fusion model.

Feature ladder
--------------
M0 = static clinical/genetic context
M1 = M0 + compact pre-specified longitudinal clone dynamics
M2 = M1 + NVIDIA BioNeMo / ESM-2 residue-delta patient representation

The model is deliberately organized as an ablation ladder. v2.3 uses a compact
confirmatory dynamics set by default to reduce collinearity/overfitting; the full
v2.2 catalogue remains available for sensitivity analyses. BioNeMo is the
molecular-intelligence layer, not a stand-alone leukemia predictor. All learned
preprocessing (imputation, scaling, PCA) is fitted inside each training fold.

Research use only. Not a diagnostic device.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


try:
    from .model_spec import (
        CORE_STATIC_V23, CORE_DYNAMICS_V23, EXTENDED_DYNAMICS_V23,
        EXPERIMENTAL_DYNAMICS_V23, CHRS_REFERENCE_V23,
    )
except ImportError:  # direct script/notebook use
    from model_spec import (
        CORE_STATIC_V23, CORE_DYNAMICS_V23, EXTENDED_DYNAMICS_V23,
        EXPERIMENTAL_DYNAMICS_V23, CHRS_REFERENCE_V23,
    )

# v2.3 defaults are deliberately compact.  The previous v2.2 feature catalogue
# is retained as EXTENDED_DYNAMICS for predeclared sensitivity analyses.
DEFAULT_STATIC = list(CORE_STATIC_V23)
DEFAULT_DYNAMICS = list(CORE_DYNAMICS_V23)
EXTENDED_DYNAMICS = list(EXTENDED_DYNAMICS_V23)
EXPERIMENTAL_DYNAMICS = list(EXPERIMENTAL_DYNAMICS_V23)
CHRS_REFERENCE = list(CHRS_REFERENCE_V23)



class AdaptivePCA(BaseEstimator, TransformerMixin):
    """PCA that safely adapts n_components to each training fold.

    A fixed PCA(16) fails when a small fold has fewer than 16 usable samples.
    This transformer chooses min(requested, n_features, n_samples-1) *during fit*,
    keeping the entire choice inside the training fold.
    """

    def __init__(self, n_components: int = 16, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError("AdaptivePCA expects a 2D matrix")
        k = min(int(self.n_components), X.shape[1], max(1, X.shape[0] - 1))
        self.n_components_ = int(k)
        self.pca_ = PCA(n_components=self.n_components_, random_state=self.random_state)
        self.pca_.fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(np.asarray(X, float))


@dataclass
class FeatureBundle:
    patient_ids: np.ndarray
    X_static: np.ndarray
    X_dynamics: np.ndarray
    X_protein: Optional[np.ndarray]
    static_names: List[str]
    dynamics_names: List[str]

    @property
    def X_m0(self) -> np.ndarray:
        return self.X_static

    @property
    def X_m1(self) -> np.ndarray:
        return np.hstack([self.X_static, self.X_dynamics])

    @property
    def X_m2(self) -> np.ndarray:
        if self.X_protein is None:
            raise ValueError("M2 requested but no BioNeMo/ESM-2 matrix was supplied")
        return np.hstack([self.X_static, self.X_dynamics, self.X_protein])


@dataclass
class FittedFusion:
    model: Pipeline
    threshold: float
    feature_level: str
    n_tabular: int
    n_protein: int
    n_pca: int
    static_names: List[str]
    dynamics_names: List[str]
    genes: List[str]
    target_specificity: float


def _normalise_gene_iterable(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, set, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # permissive CSV-friendly forms: DNMT3A;TET2 or DNMT3A,TET2
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast
                v = ast.literal_eval(s)
                return list(v) if isinstance(v, (list, tuple, set)) else [str(v)]
            except Exception:
                pass
        sep = ";" if ";" in s else ","
        return [t.strip() for t in s.split(sep) if t.strip()]
    return [str(x)]


def _gene_onehot(patients: pd.DataFrame, genes: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    genes = list(genes)
    if not genes:
        return np.zeros((len(patients), 0), dtype=float), []
    if "genes" not in patients.columns:
        raise KeyError("patients must contain a `genes` column when gene one-hot is requested")
    parsed = patients["genes"].apply(_normalise_gene_iterable)
    block = np.column_stack([
        parsed.apply(lambda gs, g=g: float(g in set(gs))).to_numpy()
        for g in genes
    ])
    return block.astype(float), [f"gene_{g}" for g in genes]


def build_feature_bundle(
    patients: pd.DataFrame,
    *,
    dynamics: Optional[pd.DataFrame] = None,
    protein_matrix: Optional[np.ndarray] = None,
    genes: Sequence[str] = (),
    patient_col: str = "patient_id",
    static_features: Sequence[str] = DEFAULT_STATIC,
    dynamics_features: Sequence[str] = DEFAULT_DYNAMICS,
) -> FeatureBundle:
    """Assemble aligned M0/M1/M2 matrices.

    Every row in ``patients`` is one prediction index. Features must be known at
    or before that index. ``protein_matrix`` is aligned row-for-row to patients.
    """
    if patient_col not in patients.columns:
        raise KeyError(f"patients missing {patient_col!r}")
    if patients[patient_col].duplicated().any():
        raise ValueError("patients must contain one row per prediction index/patient")

    p = patients.copy().reset_index(drop=True)
    have_static = [c for c in static_features if c in p.columns]
    if not have_static and not genes:
        raise ValueError("No static baseline features were found")

    blocks: List[np.ndarray] = []
    static_names: List[str] = []
    if have_static:
        blocks.append(p[have_static].apply(pd.to_numeric, errors="coerce").to_numpy(float))
        static_names += have_static

    gene_block, gene_names = _gene_onehot(p, genes)
    if gene_block.shape[1]:
        blocks.append(gene_block)
        static_names += gene_names
    X_static = np.hstack(blocks) if blocks else np.zeros((len(p), 0), dtype=float)

    dyn_names: List[str] = []
    if dynamics is None:
        X_dyn = np.zeros((len(p), 0), dtype=float)
    else:
        if patient_col not in dynamics.columns:
            raise KeyError(f"dynamics missing {patient_col!r}")
        if dynamics[patient_col].duplicated().any():
            raise ValueError("dynamics must have one row per patient")
        d = p[[patient_col]].merge(dynamics, on=patient_col, how="left", validate="one_to_one")
        dyn_names = [c for c in dynamics_features if c in d.columns]
        X_dyn = (
            d[dyn_names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            if dyn_names else np.zeros((len(p), 0), dtype=float)
        )

    X_protein = None
    if protein_matrix is not None:
        X_protein = np.asarray(protein_matrix, dtype=float)
        if X_protein.ndim != 2 or X_protein.shape[0] != len(p):
            raise ValueError(
                f"protein_matrix must be (n_patients, n_dims); got {X_protein.shape}; "
                f"expected first dimension {len(p)}"
            )

    return FeatureBundle(
        patient_ids=p[patient_col].to_numpy(),
        X_static=X_static,
        X_dynamics=X_dyn,
        X_protein=X_protein,
        static_names=static_names,
        dynamics_names=dyn_names,
    )


def _make_pipeline(n_tabular: int, n_protein: int = 0, n_pca: int = 16,
                   C: float = 1.0) -> Pipeline:
    if n_tabular < 1:
        raise ValueError("At least one tabular feature is required")

    transformers = [
        ("tabular", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), list(range(n_tabular)))
    ]
    if n_protein:
        emb_idx = list(range(n_tabular, n_tabular + n_protein))
        transformers.append((
            "protein",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("pca", AdaptivePCA(n_components=n_pca, random_state=0)),
                ("scale", StandardScaler()),
            ]),
            emb_idx,
        ))

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=float(C),
            solver="liblinear",
            random_state=0,
        )),
    ])


def sensitivity_threshold(y: Sequence[int], p: Sequence[float],
                          target_specificity: float = 0.98) -> float:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    fpr, _, thr = roc_curve(y, p)
    ok = np.where(fpr <= 1.0 - target_specificity)[0]
    if not len(ok):
        return float(np.nanmax(p)) + 1e-9
    t = float(thr[ok[-1]])
    return t if np.isfinite(t) else float(np.nanmax(p)) + 1e-9


def _splitter(y: np.ndarray, n_splits: int, seed: int,
              groups: Optional[np.ndarray] = None):
    if groups is not None:
        if StratifiedGroupKFold is None:
            raise RuntimeError("This sklearn version lacks StratifiedGroupKFold")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return cv.split(np.zeros(len(y)), y, groups)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cv.split(np.zeros(len(y)), y)


def expected_calibration_error(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    """Simple equal-width ECE for research diagnostics."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def cross_validated_predictions(
    X: np.ndarray,
    y: Sequence[int],
    *,
    n_tabular: int,
    n_protein: int = 0,
    groups: Optional[Sequence] = None,
    n_splits: int = 5,
    seed: int = 0,
    n_pca: int = 16,
    C: float = 1.0,
    target_specificity: float = 0.98,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """OOF probabilities plus fold metrics using train-fold thresholds only."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if len(np.unique(y)) != 2:
        raise ValueError("y must contain both 0 and 1")
    if X.shape[0] != len(y):
        raise ValueError("X/y length mismatch")
    g = None if groups is None else np.asarray(groups)

    # Fail early with an interpretable message instead of sklearn's cryptic error.
    minority = int(min((y == 0).sum(), (y == 1).sum()))
    if minority < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds minority-class count={minority}")

    oof = np.full(len(y), np.nan, dtype=float)
    rows: List[Dict[str, float]] = []
    for fold, (tr, te) in enumerate(_splitter(y, n_splits, seed, g), 1):
        model = _make_pipeline(n_tabular, n_protein, n_pca=n_pca, C=C)
        model.fit(X[tr], y[tr])
        p_tr = model.predict_proba(X[tr])[:, 1]
        threshold = sensitivity_threshold(y[tr], p_tr, target_specificity)
        p_te = model.predict_proba(X[te])[:, 1]
        oof[te] = p_te
        yhat = (p_te >= threshold).astype(int)
        tn = int(((y[te] == 0) & (yhat == 0)).sum())
        fp = int(((y[te] == 0) & (yhat == 1)).sum())
        fn = int(((y[te] == 1) & (yhat == 0)).sum())
        tp = int(((y[te] == 1) & (yhat == 1)).sum())
        rows.append({
            "fold": fold,
            "auc": roc_auc_score(y[te], p_te),
            "auprc": average_precision_score(y[te], p_te),
            "brier": brier_score_loss(y[te], p_te),
            "ece": expected_calibration_error(y[te], p_te),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1),
            "threshold_train": threshold,
            "n_test": len(te),
            "n_test_pos": int(y[te].sum()),
        })
    if np.isnan(oof).any():
        raise RuntimeError("Some rows did not receive out-of-fold predictions")
    return oof, pd.DataFrame(rows).set_index("fold")


def evaluate_ladder(
    bundle: FeatureBundle,
    y: Sequence[int],
    *,
    groups: Optional[Sequence] = None,
    n_splits: int = 5,
    seed: int = 0,
    n_pca: int = 16,
    target_specificity: float = 0.98,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """Evaluate M0 -> M1 -> M2 using the same split design."""
    y = np.asarray(y, int)
    configs = {
        "M0_static": (bundle.X_m0, bundle.X_static.shape[1], 0),
        "M1_plus_dynamics": (bundle.X_m1, bundle.X_m1.shape[1], 0),
    }
    if bundle.X_protein is not None:
        configs["M2_plus_BioNeMo"] = (
            bundle.X_m2,
            bundle.X_static.shape[1] + bundle.X_dynamics.shape[1],
            bundle.X_protein.shape[1],
        )

    preds: Dict[str, np.ndarray] = {}
    fold_tables: Dict[str, pd.DataFrame] = {}
    rows = []
    for name, (X, n_tab, n_prot) in configs.items():
        p, ft = cross_validated_predictions(
            X, y, n_tabular=n_tab, n_protein=n_prot, groups=groups,
            n_splits=n_splits, seed=seed, n_pca=n_pca,
            target_specificity=target_specificity,
        )
        preds[name] = p
        fold_tables[name] = ft
        rows.append({
            "model": name,
            "oof_auc": roc_auc_score(y, p),
            "oof_auprc": average_precision_score(y, p),
            "oof_brier": brier_score_loss(y, p),
            "oof_ece": expected_calibration_error(y, p),
            "mean_sensitivity": ft["sensitivity"].mean(),
            "mean_specificity": ft["specificity"].mean(),
            "mean_ppv": ft["ppv"].mean(),
            "n": len(y),
            "n_pos": int(y.sum()),
        })

    summary = pd.DataFrame(rows).set_index("model")
    summary["delta_auc_vs_prev"] = summary["oof_auc"].diff()
    summary["delta_auprc_vs_prev"] = summary["oof_auprc"].diff()
    return summary, preds, fold_tables


def bootstrap_increment(
    y: Sequence[int], p_a: Sequence[float], p_b: Sequence[float],
    *, n_boot: int = 2000, seed: int = 0,
) -> Dict[str, float]:
    """Paired patient bootstrap for B-vs-A OOF AUC and AUPRC increments."""
    y = np.asarray(y, int)
    a = np.asarray(p_a, float)
    b = np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    da, dp = [], []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        da.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
        dp.append(average_precision_score(y[idx], b[idx]) - average_precision_score(y[idx], a[idx]))
    da, dp = np.asarray(da), np.asarray(dp)
    alo, ahi = np.percentile(da, [2.5, 97.5])
    plo, phi = np.percentile(dp, [2.5, 97.5])
    return {
        "delta_auc_mean": float(da.mean()), "delta_auc_ci95_low": float(alo),
        "delta_auc_ci95_high": float(ahi), "p_auc_delta_positive": float((da > 0).mean()),
        "delta_auprc_mean": float(dp.mean()), "delta_auprc_ci95_low": float(plo),
        "delta_auprc_ci95_high": float(phi), "p_auprc_delta_positive": float((dp > 0).mean()),
        "n_boot": int(n_boot),
    }


def random_embedding_null(
    bundle: FeatureBundle,
    y: Sequence[int],
    *,
    groups: Optional[Sequence] = None,
    n_reps: int = 10,
    n_splits: int = 5,
    n_pca: int = 16,
    target_specificity: float = 0.98,
    seed: int = 100,
) -> Dict[str, object]:
    """Null control for the apparent BioNeMo AUC gain.

    Runs the exact M2 pipeline after replacing the protein vectors by random
    Gaussian features with the same dimensionality. A real BioNeMo increment
    should be interpreted relative to this null, not merely relative to zero.
    """
    if bundle.X_protein is None:
        raise ValueError("random_embedding_null requires a protein matrix")
    y = np.asarray(y, int)
    base_p, _ = cross_validated_predictions(
        bundle.X_m1, y, n_tabular=bundle.X_m1.shape[1], n_protein=0,
        groups=groups, n_splits=n_splits, seed=0, n_pca=n_pca,
        target_specificity=target_specificity,
    )
    base_auc = roc_auc_score(y, base_p)
    dims = bundle.X_protein.shape[1]
    deltas = []
    for i in range(n_reps):
        rng = np.random.default_rng(seed + i)
        z = rng.normal(size=(len(y), dims))
        X = np.hstack([bundle.X_m1, z])
        p, _ = cross_validated_predictions(
            X, y, n_tabular=bundle.X_m1.shape[1], n_protein=dims,
            groups=groups, n_splits=n_splits, seed=0, n_pca=n_pca,
            target_specificity=target_specificity,
        )
        deltas.append(float(roc_auc_score(y, p) - base_auc))
    d = np.asarray(deltas)
    return {
        "null_delta_auc_mean": float(d.mean()),
        "null_delta_auc_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "null_delta_auc_p95": float(np.percentile(d, 95)),
        "deltas": deltas,
    }


def fit_final(
    X: np.ndarray, y: Sequence[int], *, n_tabular: int, n_protein: int = 0,
    n_pca: int = 16, C: float = 1.0, threshold: float = 0.5,
    feature_level: str = "M2_plus_BioNeMo", static_names: Sequence[str] = (),
    dynamics_names: Sequence[str] = (), genes: Sequence[str] = (),
    target_specificity: float = 0.98,
) -> FittedFusion:
    model = _make_pipeline(n_tabular, n_protein, n_pca=n_pca, C=C)
    model.fit(np.asarray(X, float), np.asarray(y, int))
    return FittedFusion(
        model=model, threshold=float(threshold), feature_level=feature_level,
        n_tabular=int(n_tabular), n_protein=int(n_protein), n_pca=int(n_pca),
        static_names=list(static_names), dynamics_names=list(dynamics_names),
        genes=list(genes), target_specificity=float(target_specificity),
    )


def save_artifact(artifact: FittedFusion, model_path: str, extra_meta: Optional[dict] = None) -> str:
    """Persist the fitted research model and a human-readable metadata sidecar."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(artifact, model_path)
    meta = {
        "feature_level": artifact.feature_level,
        "threshold": artifact.threshold,
        "target_specificity": artifact.target_specificity,
        "n_tabular": artifact.n_tabular,
        "n_protein": artifact.n_protein,
        "n_pca": artifact.n_pca,
        "static_names": artifact.static_names,
        "dynamics_names": artifact.dynamics_names,
        "genes": artifact.genes,
        "research_use_only": True,
        "not_a_diagnostic_device": True,
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path = model_path.replace(".joblib", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def load_artifact(model_path: str) -> FittedFusion:
    obj = joblib.load(model_path)
    if not isinstance(obj, FittedFusion):
        raise TypeError("Expected a FittedFusion artifact")
    return obj
