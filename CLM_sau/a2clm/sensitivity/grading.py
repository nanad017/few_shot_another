"""Sensitivity grading (paper Sec. III-B1).

Behavior-event space is divided into K=3 sensitivity levels:
1 = benign, 2 = sensitive, 3 = malicious.

Statistics-based rule (Eq. 1): for every extracted string parameter compute
TF and DF separately on malicious / benign corpora; TD_m = TF_m * DF_m and
TD_b = TF_b * DF_b. High TD_m with low TD_b -> level 3, the reverse -> 1,
in-between -> 2. Parameters the statistics cannot decide are clustered with
GSDMM (K=3) and each cluster is mapped to a level by its members' mean
TD_m - TD_b.

Events without decisive parameters fall back to a keyword prior distilled
from the paper (Table I) and the relation-type prior.
"""

import json
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .gsdmm import GSDMM

# Keyword prior distilled from Table I and Sec. III-C4/C5 of the paper.
API_PRIOR: List[Tuple[str, int]] = [
    ("writepefile", 3), ("writevirtualmemory", 3), ("createremotethread", 3),
    ("adjustprivilegestoken", 3), ("deletevaluekey", 3), ("querydns", 3),
    ("setwindowshook", 3), ("terminateprocess", 3), ("queueapcthread", 3),
    ("unmapviewofsection", 3), ("cryptencrypt", 3), ("urldownload", 3),
    ("readfile", 2), ("writefile", 2), ("createmutex", 2), ("openprocess", 2),
    ("setvaluekey", 2), ("createfile", 2), ("copyfile", 2), ("deletefile", 2),
    ("connect", 2), ("send", 2), ("recv", 2), ("virtualalloc", 2),
    ("becreated", 1), ("loadlibrary", 1), ("closehandle", 1),
    ("getsystemtime", 1), ("exitprocess", 1), ("openfile", 1),
    ("queryinformation", 1),
]

RELATION_PRIOR: Dict[str, int] = {
    "fork": 1, "call": 1, "read": 1,
    "access": 2, "open": 2, "set": 2,
    "connect": 3, "download": 3,
}

_PARAM_KIND = [
    (re.compile(r"^[a-z]:\\", re.I), "path"),
    (re.compile(r"\.dll$", re.I), "dll"),
    (re.compile(r"^hkey_", re.I), "regkey"),
    (re.compile(r"^(https?|ftp)://", re.I), "url"),
    (re.compile(r"^\d{1,3}(\.\d{1,3}){3}"), "ip"),
]


def normalize_param(p: str) -> str:
    """Normalize a raw string parameter into a corpus token."""
    p = str(p).strip().lower()
    p = re.sub(r"\d+", "0", p)          # collapse volatile numbers/ids
    p = re.sub(r"\s+", " ", p)
    return p[:200]


def param_kind(p: str) -> str:
    for rx, kind in _PARAM_KIND:
        if rx.search(str(p).strip()):
            return kind
    return "str"


def _tokens_for_clustering(param: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0]+", param) if t]


class SensitivityGrader:
    def __init__(self, ratio_hi: float = 2.0, td_min: float = 1e-6,
                 min_count: int = 3, seed: int = 42):
        self.ratio_hi = ratio_hi
        self.td_min = td_min
        self.min_count = min_count
        self.seed = seed
        self.param_grade: Dict[str, int] = {}
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, samples: Iterable[Tuple[List[dict], int]]) -> "SensitivityGrader":
        """samples: iterable of (events, is_malware). Each event is a dict with
        keys src_type/src/relation/dst_type/dst/parameters."""
        tf = {0: defaultdict(int), 1: defaultdict(int)}   # term counts
        df = {0: defaultdict(int), 1: defaultdict(int)}   # doc counts
        n_docs = {0: 0, 1: 0}
        n_terms = {0: 0, 1: 0}

        for events, label in samples:
            label = int(bool(label))
            n_docs[label] += 1
            seen = set()
            for ev in events:
                for p in ev.get("parameters") or []:
                    tok = normalize_param(p)
                    if not tok:
                        continue
                    tf[label][tok] += 1
                    n_terms[label] += 1
                    if tok not in seen:
                        df[label][tok] += 1
                        seen.add(tok)

        vocab = set(tf[0]) | set(tf[1])
        undecided: List[str] = []
        stats: Dict[str, Tuple[float, float]] = {}
        for tok in vocab:
            tf_m = tf[1][tok] / max(n_terms[1], 1)        # Eq. 1
            tf_b = tf[0][tok] / max(n_terms[0], 1)
            df_m = df[1][tok] / max(n_docs[1], 1)
            df_b = df[0][tok] / max(n_docs[0], 1)
            td_m, td_b = tf_m * df_m, tf_b * df_b
            stats[tok] = (td_m, td_b)
            count = tf[0][tok] + tf[1][tok]
            if count < self.min_count or (td_m < self.td_min and td_b < self.td_min):
                undecided.append(tok)
            elif td_m > self.ratio_hi * td_b:
                self.param_grade[tok] = 3
            elif td_b > self.ratio_hi * td_m:
                self.param_grade[tok] = 1
            else:
                self.param_grade[tok] = 2

        # Clustering-based fallback for undecided parameters (GSDMM, K=3).
        if undecided:
            docs = [_tokens_for_clustering(t) or [t] for t in undecided]
            labels = GSDMM(k=3, seed=self.seed).fit(docs)
            cluster_score = defaultdict(list)
            for tok, c in zip(undecided, labels):
                td_m, td_b = stats[tok]
                cluster_score[c].append(td_m - td_b)
            order = sorted(cluster_score,
                           key=lambda c: sum(cluster_score[c]) / len(cluster_score[c]))
            grade_of_cluster = {c: g for c, g in zip(order, (1, 2, 3))}
            for tok, c in zip(undecided, labels):
                self.param_grade[tok] = grade_of_cluster.get(c, 2)

        self.fitted = True
        return self

    # ---------------------------------------------------------------- grade
    def grade_param(self, p: str) -> int:
        tok = normalize_param(p)
        if tok in self.param_grade:
            return self.param_grade[tok]
        return 0  # unknown

    def grade_event(self, ev: dict) -> int:
        grades = [g for g in (self.grade_param(p) for p in ev.get("parameters") or [])
                  if g > 0]
        if grades:
            return max(grades)
        name = str(ev.get("dst", "")).lower()
        for key, g in API_PRIOR:
            if key in name:
                return g
        return RELATION_PRIOR.get(str(ev.get("relation", "")).lower(), 2)

    # ------------------------------------------------------------------- io
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"param_grade": self.param_grade,
                       "ratio_hi": self.ratio_hi, "min_count": self.min_count},
                      f)

    @classmethod
    def load(cls, path: str) -> "SensitivityGrader":
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        g = cls(ratio_hi=blob.get("ratio_hi", 2.0),
                min_count=blob.get("min_count", 3))
        g.param_grade = {k: int(v) for k, v in blob["param_grade"].items()}
        g.fitted = True
        return g
