"""Research inference for a fitted LeukemiPrediag v2 fusion artifact.

This module consumes *prepared* index-time features. It does not fabricate
missing longitudinal history and it does not turn a research score into a
clinical diagnosis.

Typical inputs:
- one patient-level static row,
- optional serial VAF history -> clone_dynamics features,
- optional patient BioNeMo/ESM-2 vector.

Research use only; not for patient care.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from src.clone_dynamics import patient_dynamics_features
    from src.fusion_model import build_feature_bundle, load_artifact
except ImportError:
    from clone_dynamics import patient_dynamics_features
    from fusion_model import build_feature_bundle, load_artifact


class FusionPredictor:
    def __init__(self, model_path: str):
        self.artifact = load_artifact(model_path)

    def _level_matrix(self, bundle):
        level = self.artifact.feature_level
        if level == "M0_static":
            return bundle.X_m0
        if level == "M1_plus_dynamics":
            return bundle.X_m1
        if level == "M2_plus_BioNeMo":
            return bundle.X_m2
        raise ValueError(f"Unknown feature level: {level}")

    def predict_prepared(
        self,
        patient_row: Dict,
        *,
        serial_vaf: Optional[pd.DataFrame] = None,
        protein_vector: Optional[Sequence[float]] = None,
        patient_col: str = "patient_id",
    ) -> Dict:
        """Return a research probability from index-time information only.

        ``protein_vector`` must be the same patient-level BioNeMo representation
        used during training (for example VAF-weighted residue-delta aggregation).
        """
        p = pd.DataFrame([patient_row]).copy()
        if patient_col not in p.columns:
            p[patient_col] = "QUERY"
        p[patient_col] = p[patient_col].astype(str)

        dynamics = None
        if serial_vaf is not None:
            s = serial_vaf.copy()
            if patient_col not in s.columns:
                s[patient_col] = p[patient_col].iloc[0]
            s[patient_col] = s[patient_col].astype(str)
            if patient_col != "patient_id":
                s = s.rename(columns={patient_col: "patient_id"})
            dynamics = patient_dynamics_features(s)
            if patient_col != "patient_id":
                dynamics = dynamics.rename(columns={"patient_id": patient_col})

        protein = None
        if protein_vector is not None:
            protein = np.asarray(protein_vector, float).reshape(1, -1)
            p["has_embedding"] = 1.0
        elif self.artifact.n_protein:
            raise ValueError(
                "This fitted M2 model requires a patient BioNeMo/ESM-2 vector. "
                "Do not substitute zeros silently at inference time."
            )

        # Reuse exactly the feature names frozen in the artifact. Gene names are
        # also frozen so inference cannot accidentally change the design matrix.
        bundle = build_feature_bundle(
            p,
            dynamics=dynamics,
            protein_matrix=protein,
            genes=self.artifact.genes,
            patient_col=patient_col,
            static_features=self.artifact.static_names,
            dynamics_features=self.artifact.dynamics_names,
        )
        X = self._level_matrix(bundle)
        prob = float(self.artifact.model.predict_proba(X)[0, 1])
        flag = bool(prob >= self.artifact.threshold)
        return {
            "research_probability": prob,
            "research_flag": flag,
            "threshold": float(self.artifact.threshold),
            "feature_level": self.artifact.feature_level,
            "target_specificity_internal": float(self.artifact.target_specificity),
            "has_longitudinal_dynamics": bool(dynamics is not None),
            "has_bionemo": bool(protein is not None),
            "research_use_only": True,
            "not_a_diagnosis": True,
        }
