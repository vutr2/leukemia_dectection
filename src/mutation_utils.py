"""
mutation_utils.py
Parse and normalise protein-level mutation notation (HGVSp).

Handles the formats you actually see in cBioPortal / MAF files:
    p.R882H      p.Arg882His      R882H
    p.Q1523*     p.S1471fs        p.X123_splice
"""

from __future__ import annotations
import re
from typing import Optional, Dict

AA3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U",
}
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Ordered patterns. Real MAFs carry far more than simple substitutions:
#   p.R882H        missense
#   p.Q1523*       nonsense
#   p.I679Dfs*21   frameshift  <- the old regex missed these entirely
#   p.X216_splice  splice site
#   p.P606_R607insSSSD...  in-frame insertion
_RE_SPLICE = re.compile(r"^p?\.?[A-Z]?\d+_splice", re.I)
_RE_FS_1   = re.compile(r"^p?\.?([A-Z])(\d+)([A-Z])?fs")
_RE_FS_3   = re.compile(r"^p?\.?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})?fs", re.I)
_RE_INDEL  = re.compile(r"^p?\.?([A-Z][a-z]{0,2})(\d+).*(ins|del|dup)", re.I)
_RE_NONS_1 = re.compile(r"^p?\.?([A-Z])(\d+)\*$")
_RE_NONS_3 = re.compile(r"^p?\.?([A-Z][a-z]{2})(\d+)(Ter|\*)$")
_RE_MIS_1  = re.compile(r"^p?\.?([A-Z])(\d+)([A-Z])$")
_RE_MIS_3  = re.compile(r"^p?\.?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
_RE_SIL    = re.compile(r"^p?\.?([A-Z][a-z]{0,2})(\d+)=$")


def parse_protein_change(change: str) -> Optional[Dict]:
    """
    Return {'wt', 'pos', 'mut', 'kind'} or None if unparseable.

    kind: missense | nonsense | frameshift | splice | indel | silent

    Only 'missense' can be turned into an ESM2 delta (it is a clean single-residue
    substitution). The rest still matter: they carry gene identity and clonal
    burden, and ASXL1 frameshifts in particular are a hallmark of secondary AML.
    Keep them in the table and let the embedding step filter to missense.
    """
    if change is None:
        return None
    s = str(change).strip()
    if not s or s.lower() in {"nan", "none", ".", "-"}:
        return None

    def _1(aa):
        return AA3_TO_1.get(aa.capitalize(), aa if len(aa) == 1 else None)

    if _RE_SPLICE.match(s):
        m = re.search(r"(\d+)", s)
        return {"wt": None, "pos": int(m.group(1)) if m else None,
                "mut": None, "kind": "splice"}

    for rx, three in ((_RE_FS_3, True), (_RE_FS_1, False)):
        m = rx.match(s)
        if m:
            return {"wt": _1(m.group(1)), "pos": int(m.group(2)),
                    "mut": "fs", "kind": "frameshift"}

    for rx, three in ((_RE_NONS_3, True), (_RE_NONS_1, False)):
        m = rx.match(s)
        if m:
            return {"wt": _1(m.group(1)), "pos": int(m.group(2)),
                    "mut": "*", "kind": "nonsense"}

    m = _RE_SIL.match(s)
    if m:
        return {"wt": _1(m.group(1)), "pos": int(m.group(2)),
                "mut": None, "kind": "silent"}

    for rx, three in ((_RE_MIS_3, True), (_RE_MIS_1, False)):
        m = rx.match(s)
        if m:
            wt, mut = _1(m.group(1)), _1(m.group(3))
            if wt and mut and wt in VALID_AA and mut in VALID_AA:
                kind = "silent" if wt == mut else "missense"
                return {"wt": wt, "pos": int(m.group(2)), "mut": mut, "kind": kind}

    m = _RE_INDEL.match(s)
    if m:
        return {"wt": _1(m.group(1)), "pos": int(m.group(2)),
                "mut": m.group(3).lower(), "kind": "indel"}

    return None


def is_missense(change: str) -> bool:
    p = parse_protein_change(change)
    return bool(p and p["kind"] == "missense")


def mutation_key(gene: str, change: str) -> str:
    """Stable cache key, e.g. 'DNMT3A:p.R882H' -> 'DNMT3A_R882H'."""
    p = parse_protein_change(change)
    if p is None:
        return f"{gene}_{str(change).replace('p.', '').replace('*', 'X')}"
    return f"{gene}_{p['wt']}{p['pos']}{p['mut']}"


# --- Gene panels -------------------------------------------------------------

# Genes recurrently mutated in clonal haematopoiesis / myeloid neoplasms.
CH_GENES = [
    "DNMT3A", "TET2", "ASXL1", "PPM1D", "TP53", "JAK2", "SF3B1", "SRSF2",
    "U2AF1", "ZRSR2", "IDH1", "IDH2", "RUNX1", "CBL", "GNB1", "GNAS",
    "KRAS", "NRAS", "FLT3", "NPM1", "CEBPA", "KIT", "WT1", "EZH2",
    "STAG2", "BCOR", "BCORL1", "PHF6", "ETV6", "SETBP1", "CSF3R",
    "CALR", "MPL", "RAD21", "SMC3", "NF1", "PTPN11", "BRAF", "CUX1",
    "SETD2", "KDM6A", "CREBBP", "EP300", "PRPF8", "LUC7L2", "DDX41",
]

# Genotypes associated in the literature (Abelson 2018, Desai 2018, Bolton 2020)
# with elevated risk of progression to AML/MDS.
HIGH_RISK_GENES = [
    "TP53", "SRSF2", "SF3B1", "U2AF1", "ZRSR2", "IDH1", "IDH2",
    "RUNX1", "JAK2", "FLT3", "NPM1", "SETBP1",
]
