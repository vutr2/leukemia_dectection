"""
sequences.py
Fetch canonical human protein sequences from UniProt, with a JSON cache on disk.

Needs internet the first time (works fine on Colab). After that everything is
served from data/external/uniprot_cache.json, so commit that file and the
pipeline becomes reproducible offline.
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, Iterable, Optional

import requests

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
DEFAULT_CACHE = "data/external/uniprot_cache.json"


def _load_cache(path: str) -> Dict[str, str]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: Dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


def fetch_sequence(gene: str, timeout: int = 20) -> Optional[str]:
    """Canonical reviewed human sequence for a HGNC gene symbol."""
    params = {
        "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
        "fields": "sequence",
        "format": "json",
        "size": 1,
    }
    try:
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        return results[0]["sequence"]["value"]
    except Exception as e:  # network off, rate limit, schema change
        print(f"[uniprot] {gene}: {type(e).__name__}: {e}")
        return None


def get_sequences(
    genes: Iterable[str],
    cache_path: str = DEFAULT_CACHE,
    sleep: float = 0.2,
) -> Dict[str, str]:
    """Return {gene: sequence} for every gene we can resolve. Caches to disk."""
    cache = _load_cache(cache_path)
    missing = [g for g in sorted(set(genes)) if g not in cache]
    if missing:
        print(f"[uniprot] fetching {len(missing)} sequences...")
    for i, g in enumerate(missing, 1):
        seq = fetch_sequence(g)
        if seq:
            cache[g] = seq
        print(f"  [{i}/{len(missing)}] {g}: {'ok, %d aa' % len(seq) if seq else 'FAILED'}")
        time.sleep(sleep)
    if missing:
        _save_cache(cache, cache_path)
    return {g: cache[g] for g in set(genes) if g in cache}
