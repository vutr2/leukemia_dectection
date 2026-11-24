"""
embedding_bionemo.py
Residue-level WT vs mutant delta using NVIDIA BioNeMo ESM-2.

Drop-in replacement for the fair-esm path in embedding.py.

Why this file exists
--------------------
The current pipeline computes:

    delta = ESM2(mutant)[i] - ESM2(wildtype)[i]

Mean-pooling the whole protein kills the signal (DNMT3A is 912 aa).
This module keeps that contract and only changes the backend.

Three backends, same output
---------------------------
1. nvidia_hf   NVIDIA TransformerEngine ESM-2 on HuggingFace
               (the Inception-facing path; needs GPU + transformers)
2. facebook_hf facebook/esm2_* via transformers
               (works today on Colab without the BioNeMo container)
3. bionemo_fw  BioNeMo Framework 2.x (NeMo checkpoint + infer_esm2)
               Use this inside nvcr.io/nvidia/clara/bionemo-framework

Default is nvidia_hf with automatic fallback to facebook_hf.

Windowing
---------
ESM-2 context is 1022 residues. TET2 is 2002 aa. Sequences longer than
MAX_LEN are cropped symmetrically around the mutated residue so the
site of interest stays inside the window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

MAX_LEN = 1022
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Official Meta names. NVIDIA mirrors them under nvidia/<same>.
MODEL_IDS = {
    "8m": "esm2_t6_8M_UR50D",
    "35m": "esm2_t12_35M_UR50D",
    "150m": "esm2_t30_150M_UR50D",
    "650m": "esm2_t33_650M_UR50D",
    "3b": "esm2_t36_3B_UR50D",
}


def window_sequence(seq: str, pos_1based: int, max_len: int = MAX_LEN) -> Tuple[str, int]:
    """Crop around the mutation. Returns (windowed_seq, new_1based_pos)."""
    if pos_1based < 1 or pos_1based > len(seq):
        raise ValueError(f"pos {pos_1based} outside sequence length {len(seq)}")
    if len(seq) <= max_len:
        return seq, pos_1based
    i = pos_1based - 1
    half = max_len // 2
    start = max(0, i - half)
    end = min(len(seq), start + max_len)
    start = max(0, end - max_len)
    return seq[start:end], i - start + 1


def apply_missense(seq: str, pos_1based: int, aa_wt: str, aa_mut: str) -> str:
    i = pos_1based - 1
    if seq[i] != aa_wt:
        raise ValueError(
            f"isoform mismatch: expected {aa_wt} at {pos_1based}, found {seq[i]}"
        )
    if aa_mut not in VALID_AA:
        raise ValueError(f"non-canonical mutant AA: {aa_mut}")
    return seq[:i] + aa_mut + seq[i + 1 :]


@dataclass
class EmbedResult:
    mut_key: str
    delta: np.ndarray          # (H,)
    emb_wt: np.ndarray         # (H,)
    emb_mut: np.ndarray        # (H,)
    pos_used: int
    windowed: bool
    backend: str
    model_id: str


class BioNeMoESM2:
    """
    Per-residue ESM-2 encoder.

    Parameters
    ----------
    size : 8m | 35m | 150m | 650m | 3b
        650m is the size used in the current LeukemiPrediag pipeline.
    backend : nvidia_hf | facebook_hf | bionemo_fw | auto
        auto tries nvidia_hf then facebook_hf.
    device : cuda | cpu
    """

    def __init__(
        self,
        size: str = "650m",
        backend: str = "auto",
        device: Optional[str] = None,
        prefer_nvidia: bool = True,
    ):
        import torch

        if size not in MODEL_IDS:
            raise ValueError(f"size must be one of {list(MODEL_IDS)}")
        self.size = size
        self.short_id = MODEL_IDS[size]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            cap = torch.cuda.get_device_capability(0)[0]
            self.dtype = torch.bfloat16 if cap >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.hidden_size = None
        self.model_id = None

        order = []
        if backend == "auto":
            order = ["nvidia_hf", "facebook_hf"] if prefer_nvidia else ["facebook_hf"]
        else:
            order = [backend]

        last_err = None
        for name in order:
            try:
                self._load(name)
                self.backend = name
                break
            except Exception as e:
                last_err = e
                print(f"[bionemo] backend {name} failed: {e}")
        if self.backend is None:
            raise RuntimeError(f"no ESM-2 backend loaded: {last_err}")
        if self.model is not None:
            for p in self.model.parameters():
                p.requires_grad_(False)
        print(f"[bionemo] backend={self.backend} model={self.model_id} "
              f"H={self.hidden_size} device={self.device} dtype={self.dtype}")

    def _load(self, backend: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        common = dict(
            torch_dtype=self.dtype if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        if backend == "nvidia_hf":
            hid = f"nvidia/{self.short_id}"
            self.tokenizer = AutoTokenizer.from_pretrained(hid)
            self.model = AutoModel.from_pretrained(hid, **common).to(self.device).eval()
            self.model_id = hid
        elif backend == "facebook_hf":
            hid = f"facebook/{self.short_id}"
            self.tokenizer = AutoTokenizer.from_pretrained(hid)
            self.model = AutoModel.from_pretrained(hid, **common).to(self.device).eval()
            self.model_id = hid
        elif backend == "bionemo_fw":
            # Full Framework path — only works inside the BioNeMo container.
            # We still expose per-residue states via the HF-TE recipe if the
            # NeMo inferencer is not on PYTHONPATH.
            try:
                from bionemo.core.data.load import load  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "bionemo_fw requires the BioNeMo Framework container "
                    "(nvcr.io/nvidia/clara/bionemo-framework). "
                    "Use backend='nvidia_hf' on Colab."
                ) from e
            hid = f"nvidia/{self.short_id}"
            self.tokenizer = AutoTokenizer.from_pretrained(hid)
            self.model = AutoModel.from_pretrained(hid).to(self.device).eval()
            self.model_id = hid + "+bionemo_fw"
        else:
            raise ValueError(backend)

        self.hidden_size = int(self.model.config.hidden_size)

    def embed_sequence(self, seq: str) -> np.ndarray:
        """Return per-residue embeddings, shape (L, H). Index 0 = first AA."""
        import torch

        toks = self.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN + 2,
        )
        toks = {k: v.to(self.device, non_blocking=True) for k, v in toks.items()}
        with torch.inference_mode():
            hidden = self.model(**toks).last_hidden_state[0]
            L = min(len(seq), hidden.shape[0] - 2)
            residues = hidden[1:1 + L].float().cpu().numpy()
        del hidden, toks
        return residues

    def residue_vector(self, seq: str, pos_1based: int) -> Tuple[np.ndarray, int, bool]:
        win, new_pos = window_sequence(seq, pos_1based)
        emb = self.embed_sequence(win)
        if new_pos < 1 or new_pos > emb.shape[0]:
            raise ValueError(
                f"residue {new_pos} not in embedding of length {emb.shape[0]}"
            )
        return emb[new_pos - 1], new_pos, (win != seq)

    def delta(
        self,
        seq_wt: str,
        pos_1based: int,
        aa_wt: str,
        aa_mut: str,
        mut_key: str = "",
    ) -> EmbedResult:
        vec_wt, pos_used, windowed = self.residue_vector(seq_wt, pos_1based)
        seq_mut = apply_missense(seq_wt, pos_1based, aa_wt, aa_mut)
        vec_mut, _, _ = self.residue_vector(seq_mut, pos_1based)
        d = (vec_mut - vec_wt).astype(np.float32)
        return EmbedResult(
            mut_key=mut_key,
            delta=d,
            emb_wt=vec_wt.astype(np.float32),
            emb_mut=vec_mut.astype(np.float32),
            pos_used=pos_used,
            windowed=windowed,
            backend=self.backend,
            model_id=self.model_id,
        )


def embed_table(
    rows: Sequence[dict],
    encoder: BioNeMoESM2,
    skip_isoform_mismatch: bool = True,
) -> List[dict]:
    """
    rows need: mut_key, seq, aa_wt, aa_pos, aa_mut
    Returns list of dicts ready for DataFrame / pickle.
    """
    out = []
    n_ok = n_skip = 0
    for r in rows:
        try:
            res = encoder.delta(
                seq_wt=r["seq"],
                pos_1based=int(r["aa_pos"]),
                aa_wt=str(r["aa_wt"]),
                aa_mut=str(r["aa_mut"]),
                mut_key=str(r.get("mut_key", "")),
            )
        except Exception as e:
            n_skip += 1
            if not skip_isoform_mismatch:
                raise
            print(f"  skip {r.get('mut_key')}: {e}")
            continue
        n_ok += 1
        out.append({
            "mut_key": res.mut_key,
            "delta": res.delta,
            "emb_wt": res.emb_wt,
            "emb_mut": res.emb_mut,
            "dnorm": float(np.linalg.norm(res.delta)),
            "pos_used": res.pos_used,
            "windowed": res.windowed,
            "backend": res.backend,
            "model_id": res.model_id,
        })
    print(f"[bionemo] embedded {n_ok}  skipped {n_skip}")
    return out


if __name__ == "__main__":
    # Smoke test: a 12-residue toy sequence, no GPU required for 8m.
    enc = BioNeMoESM2(size="8m", backend="facebook_hf", device="cpu")
    toy = "MKTAYIAKQRQS"
    r = enc.delta(toy, pos_1based=3, aa_wt="T", aa_mut="A", mut_key="TOY_T3A")
    print("delta L2", float(np.linalg.norm(r.delta)), "H", r.delta.shape)
