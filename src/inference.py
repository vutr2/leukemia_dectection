"""
inference.py
Single-patient prediction: list of mutations + VAF -> probability + class.
"""

from __future__ import annotations
import json
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .data_processing import clean_mutations, aggregate_by_patient
from .model import build_features


class Predictor:
    def __init__(self, model_path: str, meta_path: Optional[str] = None):
        self.model = joblib.load(model_path)
        meta_path = meta_path or model_path.replace(".joblib", "_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.genes = self.meta["genes"]
        self.threshold = self.meta.get("threshold", 0.5)
        self.uses_embeddings = self.meta.get("uses_embeddings", False)

    def predict(
        self,
        mutations: List[Dict],
        embedder=None,
        sequences: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        mutations: [{"gene": "DNMT3A", "protein_change": "p.R882H", "vaf": 0.34}, ...]
        """
        df = pd.DataFrame(mutations)
        df["Tumor_Sample_Barcode"] = "QUERY"
        df = df.rename(columns={"gene": "Hugo_Symbol",
                                "protein_change": "HGVSp_Short",
                                "vaf": "VAF"})
        clean = clean_mutations(df, min_vaf=0.0)
        if clean.empty:
            return {"probability": None, "prediction": None,
                    "reason": "no mutation survived filtering"}

        patients = aggregate_by_patient(clean)

        emb = None
        if self.uses_embeddings:
            if embedder is None or sequences is None:
                raise ValueError("This model needs `embedder` and `sequences`.")
            from .embedding import embed_mutations, patient_embedding_matrix
            et = embed_mutations(clean, sequences, embedder)
            # Raw deltas: the saved pipeline carries its own fitted PCA.
            emb = patient_embedding_matrix(clean, et, list(patients["patient_id"]))

        X, _, _ = build_features(patients, self.genes, embeddings=emb)

        p = float(self.model.predict_proba(X)[:, 1][0])
        return {
            "probability": round(p, 4),
            "prediction": int(p >= self.threshold),
            "threshold": self.threshold,
            "n_mutations_used": int(len(clean)),
        }
