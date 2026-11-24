"""
embedding.py
Residue-level mutation deltas via NVIDIA BioNeMo ESM-2
(HuggingFace TransformerEngine weights, with Facebook ESM-2 fallback).

Design decision worth understanding before you run this:

  We do NOT embed the whole mutant protein and use that vector. Two proteins
  that differ by one residue out of 900 have nearly identical mean-pooled
  embeddings, so the mutation signal drowns in noise.

  Instead, for each mutation we compute the DELTA at the mutated residue:

      delta = ESM2(mutant_seq)[pos] - ESM2(wildtype_seq)[pos]        (1280-d)

  That is the model's local representation of "what changed here".

Long proteins (TET2 is 2002 aa, ASXL1 is 1541 aa) exceed ESM2's context, so we
take a symmetric window around the mutation before encoding.

Backends
--------
1. nvidia/<esm2_*>   BioNeMo / TransformerEngine weights on HuggingFace
2. facebook/<esm2_*> original Meta weights if NVIDIA weights are unavailable

Same architecture, same residue-delta contract. Switching backend is an
inference-stack change, not a new biological signal.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_MODEL = "nvidia/esm2_t33_650M_UR50D"
FACEBOOK_FALLBACK = "facebook/esm2_t33_650M_UR50D"
DEBUG_MODEL_NVIDIA = "nvidia/esm2_t12_35M_UR50D"
DEBUG_MODEL_FACEBOOK = "facebook/esm2_t12_35M_UR50D"
MAX_TOKENS = 1022


def window_sequence(seq: str, pos: int, max_len: int = MAX_TOKENS) -> Tuple[str, int]:
    """
    Symmetric window around a 1-based position.
    Returns (subsequence, new 1-based position inside it).
    """
    if len(seq) <= max_len:
        return seq, pos
    half = max_len // 2
    start = max(0, min(pos - 1 - half, len(seq) - max_len))
    return seq[start:start + max_len], pos - start


def apply_missense(seq: str, wt: str, pos: int, mut: str, strict: bool = False
                   ) -> Optional[str]:
    """
    Substitute residue at 1-based `pos`. Returns None if the position is out of
    range, or if the reference residue disagrees and strict=True.

    Reference mismatches are common and usually mean the MAF was annotated
    against a different transcript than UniProt's canonical one. Track the rate:
    if more than ~15% mismatch, your transcript mapping is the problem, not ESM2.
    """
    if pos < 1 or pos > len(seq):
        return None
    if seq[pos - 1] != wt:
        if strict:
            return None
    return seq[:pos - 1] + mut + seq[pos:]


def resolve_model_name(model_name: Optional[str]) -> str:
    """Map short aliases and legacy facebook ids onto a loadable checkpoint."""
    aliases = {
        None: DEFAULT_MODEL,
        "": DEFAULT_MODEL,
        "650m": DEFAULT_MODEL,
        "35m": DEBUG_MODEL_NVIDIA,
        "bionemo": DEFAULT_MODEL,
        "bionemo-650m": DEFAULT_MODEL,
        "esm2": DEFAULT_MODEL,
        FACEBOOK_FALLBACK: DEFAULT_MODEL,
        "facebook/esm2_t12_35M_UR50D": DEBUG_MODEL_NVIDIA,
    }
    if model_name in aliases:
        return aliases[model_name]
    if model_name and model_name.startswith("facebook/esm2_"):
        return "nvidia/" + model_name.split("/", 1)[1]
    return model_name or DEFAULT_MODEL


def _pick_dtype(torch, device: str) -> "object":
    if device != "cuda":
        return torch.float32
    # Ampere+ (sm80) prefers bf16; T4/P100 use fp16.
    major = torch.cuda.get_device_capability(0)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def gpu_mem_mb() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        alloc = torch.cuda.memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        return f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB"
    except Exception:
        return "n/a"


class ESM2Embedder:
    """
    Residue-level ESM-2 encoder.

    Tries NVIDIA BioNeMo HF weights first, then the original Facebook checkpoint.
    Public methods (encode / dim) stay compatible with the previous fair-esm path.

    GPU memory
    ----------
    Weights in fp16/bf16, inference_mode, batch=1 by default, only the mutated
    residue is copied off GPU. Hidden states are deleted each step.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        dtype: Optional[str] = "auto",
        low_mem: bool = True,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.low_mem = low_mem
        if dtype == "auto" or dtype is None:
            self.dtype = _pick_dtype(torch, self.device)
        else:
            self.dtype = getattr(torch, str(dtype))

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        requested = resolve_model_name(model_name)
        candidates = [requested]
        if requested.startswith("nvidia/"):
            candidates.append("facebook/" + requested.split("/", 1)[1])
        elif requested.startswith("facebook/"):
            candidates.append("nvidia/" + requested.split("/", 1)[1])

        last_err = None
        self.model_name = None
        for hid in candidates:
            try:
                print(f"[bionemo-esm2] loading {hid} on {self.device} {self.dtype} ...")
                self.tokenizer = AutoTokenizer.from_pretrained(hid)
                self.model = AutoModel.from_pretrained(
                    hid,
                    torch_dtype=self.dtype if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.model.to(self.device)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad_(False)
                self.model_name = hid
                break
            except Exception as e:
                last_err = e
                print(f"[bionemo-esm2] {hid} failed: {e}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        if self.model_name is None:
            raise RuntimeError(f"Could not load ESM-2 from {candidates}: {last_err}")

        self.dim = self.model.config.hidden_size
        self.backend = "nvidia_hf" if self.model_name.startswith("nvidia/") else "facebook_hf"
        print(f"[bionemo-esm2] ready  backend={self.backend}  "
              f"model={self.model_name}  hidden dim={self.dim}  {gpu_mem_mb()}")

    def encode(self, seqs: List[str], positions: Optional[List[int]] = None,
               batch_size: Optional[int] = None) -> np.ndarray:
        """
        Encode sequences.
          positions=None -> mean-pooled embedding over real residues  (N, D)
          positions given -> embedding of that 1-based residue        (N, D)
        """
        torch = self.torch
        if batch_size is None:
            batch_size = 1 if self.low_mem else 4
        if self.device == "cuda" and self.low_mem:
            batch_size = 1

        out = []
        ctx = torch.inference_mode()
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            # Do not pad to MAX_TOKENS — pad only to the longest sequence in the chunk.
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS + 2,
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            with ctx:
                hidden = self.model(**enc).last_hidden_state   # (B, L, D)

            if positions is None:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mask[:, 0] = 0
                idx = enc["attention_mask"].sum(1) - 1
                for b, j in enumerate(idx):
                    mask[b, int(j)] = 0
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pos_chunk = positions[i:i + batch_size]
                rows = torch.arange(hidden.size(0), device=self.device)
                cols = torch.tensor(pos_chunk, device=self.device).clamp(
                    max=hidden.size(1) - 1)
                vec = hidden[rows, cols]

            out.append(vec.detach().float().cpu().numpy())
            del hidden, vec, enc
            if self.device == "cuda" and (i // batch_size) % 32 == 31:
                torch.cuda.empty_cache()
            if (i // batch_size) % 20 == 0:
                print(f"  [bionemo-esm2] {min(i + batch_size, len(seqs))}/{len(seqs)}  {gpu_mem_mb()}")
        return np.vstack(out)

    def unload(self) -> None:
        """Free GPU weights between notebooks."""
        del self.model
        self.model = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        print(f"[bionemo-esm2] unloaded  {gpu_mem_mb()}")


# Alias used by newer notebooks / Inception write-up
BioNeMoESM2Embedder = ESM2Embedder


def embed_mutations(
    mut_table: pd.DataFrame,
    sequences: Dict[str, str],
    embedder: ESM2Embedder,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    mut_table needs: mut_key, gene, aa_wt, aa_pos, aa_mut (missense rows only).
    Returns a table of unique mutations with a `delta` column of shape (D,).
    """
    uniq = (mut_table[mut_table["kind"] == "missense"]
            .drop_duplicates("mut_key")
            .loc[:, ["mut_key", "gene", "aa_wt", "aa_pos", "aa_mut"]]
            .reset_index(drop=True))
    print(f"[embed] {len(mut_table):,} rows -> {len(uniq):,} unique missense mutations")

    wt_seqs, mt_seqs, keep_idx, wpos, mismatches = [], [], [], [], 0
    for i, r in uniq.iterrows():
        seq = sequences.get(r["gene"])
        if not seq:
            continue
        pos = int(r["aa_pos"])
        if pos > len(seq):
            continue
        if seq[pos - 1] != r["aa_wt"]:
            mismatches += 1
        mut_seq = apply_missense(seq, r["aa_wt"], pos, r["aa_mut"])
        if mut_seq is None:
            continue
        w, p = window_sequence(seq, pos)
        m, _ = window_sequence(mut_seq, pos)
        wt_seqs.append(w); mt_seqs.append(m); wpos.append(p); keep_idx.append(i)

    if not keep_idx:
        raise RuntimeError("No mutation could be mapped to a sequence. "
                           "Check that get_sequences() actually returned data.")
    print(f"[embed] mapped {len(keep_idx)}/{len(uniq)} "
          f"(reference mismatches: {mismatches}, {mismatches/max(len(keep_idx),1):.1%})")

    emb_wt = embedder.encode(wt_seqs, positions=wpos, batch_size=batch_size)
    emb_mt = embedder.encode(mt_seqs, positions=wpos, batch_size=batch_size)
    delta = emb_mt - emb_wt

    out = uniq.loc[keep_idx].copy().reset_index(drop=True)
    out["delta"] = list(delta)
    out["backend"] = getattr(embedder, "backend", "unknown")
    out["model_name"] = getattr(embedder, "model_name", "")
    return out


def patient_embedding_matrix(
    mut_table: pd.DataFrame,
    emb_table: pd.DataFrame,
    patient_ids: List[str],
    weight_by_vaf: bool = True,
) -> np.ndarray:
    """
    Collapse per-mutation deltas into one vector per patient.
    VAF weighting is the point: a mutation at VAF 0.40 represents a much larger
    clone than one at VAF 0.02 and should dominate the patient's representation.
    """
    lookup = dict(zip(emb_table["mut_key"], emb_table["delta"]))
    dim = len(next(iter(lookup.values())))
    rows, n_empty = [], 0
    for pid, sub in mut_table.groupby("patient_id"):
        vecs, ws = [], []
        for k, v in zip(sub["mut_key"], sub["vaf"]):
            if k in lookup:
                vecs.append(lookup[k])
                ws.append(float(v) if weight_by_vaf else 1.0)
        if vecs:
            w = np.asarray(ws, dtype=float)
            w = w / w.sum() if w.sum() else np.ones_like(w) / len(w)
            rows.append((pid, np.average(np.vstack(vecs), axis=0, weights=w)))
        else:
            rows.append((pid, np.zeros(dim)))
            n_empty += 1
    by_pid = dict(rows)
    X = np.vstack([by_pid.get(p, np.zeros(dim)) for p in patient_ids])
    has = np.array([[float(np.any(by_pid.get(p, np.zeros(dim)) != 0))]
                    for p in patient_ids])
    if n_empty:
        print(f"[embed] {n_empty}/{len(patient_ids)} patients have no embeddable "
              f"missense mutation; flagged via the trailing has_embedding column.")
    return np.hstack([X, has])
