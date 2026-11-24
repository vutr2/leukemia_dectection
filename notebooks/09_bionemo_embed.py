"""
09_bionemo_embed.py
Re-embed LBC/SardiNIA missense targets with NVIDIA BioNeMo ESM-2.

Does NOT retrain. It only rebuilds the delta column so you can drop it into
the existing 08_upgrade_model.py / fitness regressor without changing labels.

Colab
-----
1. Runtime → GPU (T4 is enough for 650M at batch-1).
2. Mount Drive.
3. pip install transformers torch accelerate
4. For the NVIDIA TE weights (Inception path):
       pip install transformer-engine
   If that fails on Colab, the script falls back to facebook/esm2_t33_650M_UR50D
   — same architecture, same residue-delta contract.

Expected output
---------------
data/processed/pooled_missense_with_delta_bionemo.pkl
data/processed/bionemo_embed_meta.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from embedding_bionemo import BioNeMoESM2, embed_table


def find_root() -> Path:
    candidates = [
        Path("/content/drive/MyDrive/leukemia-prediagnose/blood-cancer-early-detection"),
        Path("."),
    ]
    for p in candidates:
        if (p / "data" / "processed").exists() or p.exists():
            return p
    return Path(".")


def load_targets(processed: Path) -> pd.DataFrame:
    for name in (
        "lbc_sardinia_unique_targets.csv",
        "train_lbc_nonzero.csv",
        "pooled_missense_targets.csv",
    ):
        f = processed / name
        if f.exists():
            df = pd.read_csv(f)
            print(f"[load] {f.name}  n={len(df)}")
            return df
    raise FileNotFoundError(f"no target csv in {processed}")


def attach_sequences(df: pd.DataFrame, processed: Path, raw: Path) -> pd.DataFrame:
    """Need a wild-type protein sequence per row.

    Looks for a cache written by sequences.py (uniprot_cache.json) or a
    column already named `seq` / `sequence`.
    """
    if "seq" in df.columns:
        return df
    if "sequence" in df.columns:
        df = df.rename(columns={"sequence": "seq"})
        return df

    cache_paths = [
        processed / "uniprot_cache.json",
        raw.parent / "external" / "uniprot_cache.json" if raw else None,
        Path("uniprot_cache.json"),
    ]
    cache = {}
    for p in cache_paths:
        if p is not None and p.exists():
            cache = json.loads(p.read_text())
            print(f"[seq] loaded cache {p}  keys={len(cache)}")
            break
    if not cache:
        raise FileNotFoundError(
            "Need wild-type sequences. Put uniprot_cache.json next to the "
            "targets or add a `seq` column. Do not embed without sequences."
        )

    def pick(row):
        gene = str(row.get("gene", ""))
        # cache layouts seen in this repo: {gene: seq} or {uniprot: {seq, gene}}
        if gene in cache and isinstance(cache[gene], str):
            return cache[gene]
        if gene in cache and isinstance(cache[gene], dict):
            return cache[gene].get("seq") or cache[gene].get("sequence")
        for v in cache.values():
            if isinstance(v, dict) and v.get("gene") == gene:
                return v.get("seq") or v.get("sequence")
        return None

    df = df.copy()
    df["seq"] = df.apply(pick, axis=1)
    missing = df["seq"].isna().sum()
    print(f"[seq] missing sequences: {missing}/{len(df)}")
    df = df[df["seq"].notna()].copy()
    return df


def main(size: str = "650m", backend: str = "auto") -> None:
    root = find_root()
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    raw = root / "data" / "raw"

    df = load_targets(processed)
    df = attach_sequences(df, processed, raw)
    need = {"mut_key", "aa_wt", "aa_pos", "aa_mut", "seq"}
    missing_cols = need - set(df.columns)
    if missing_cols:
        raise ValueError(f"target table missing columns: {missing_cols}")

    enc = BioNeMoESM2(size=size, backend=backend)
    rows = df.to_dict("records")
    embedded = embed_table(rows, enc)
    out = pd.DataFrame(embedded)
    merged = df.merge(out, on="mut_key", how="inner")

    pkl_path = processed / "pooled_missense_with_delta_bionemo.pkl"
    merged.to_pickle(pkl_path)
    meta = {
        "backend": enc.backend,
        "model_id": enc.model_id,
        "size": size,
        "n_in": int(len(df)),
        "n_out": int(len(merged)),
        "hidden_size": enc.hidden_size,
        "pkl": str(pkl_path),
    }
    (processed / "bionemo_embed_meta.json").write_text(json.dumps(meta, indent=2))
    print("[wrote]", pkl_path)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
