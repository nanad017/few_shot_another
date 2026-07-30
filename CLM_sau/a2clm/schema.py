"""Network schema of A2-CLM (paper Sec. III-B, Fig. 3(b) and Fig. 4).

7 entity types, 8 canonical relations and 8 meta-graphs. The meta-graphs in
Fig. 4 are drawn as small patterns anchored at process nodes (fork chains,
shared API / registry / memory / file / system / network objects, and the
network-download-file chain); we encode each one as the set of relations it
uses plus its diameter, which drives both the meta-graph-guided neighborhood
(Eq. 6) and the per-meta-graph encoder subgraphs (Eq. 8-10).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

ENTITY_TYPES: List[str] = [
    "process", "api", "file", "system", "registry", "memory", "network",
]
ENTITY_INDEX: Dict[str, int] = {t: i for i, t in enumerate(ENTITY_TYPES)}

# (src_type, relation, dst_type) — Sec. III-B2
RELATIONS: List[Tuple[str, str, str]] = [
    ("process", "fork", "process"),
    ("process", "call", "api"),
    ("process", "access", "file"),
    ("process", "open", "system"),
    ("process", "connect", "network"),
    ("process", "read", "memory"),
    ("process", "set", "registry"),
    ("network", "download", "file"),
]
RELATION_INDEX: Dict[str, int] = {r: i for i, (_, r, _) in enumerate(RELATIONS)}
NUM_RELATIONS = len(RELATIONS)

# The relation is uniquely determined by the (src_type, dst_type) pair,
# which keeps adapters for heterogeneous report formats simple.
TYPE_PAIR_TO_RELATION: Dict[Tuple[str, str], str] = {
    (s, d): r for (s, r, d) in RELATIONS
}


@dataclass(frozen=True)
class MetaGraph:
    name: str
    relations: Tuple[str, ...]  # base relations the pattern uses
    diameter: int               # walk depth from the target process node


# Fig. 4: eight meta-graphs M1..M8.
META_GRAPHS: List[MetaGraph] = [
    MetaGraph("M1_fork_chain", ("fork",), 2),                    # P -f-> P -f-> P
    MetaGraph("M2_shared_registry", ("fork", "set"), 2),          # P -s-> R <-s- P
    MetaGraph("M3_shared_api", ("fork", "call"), 2),              # P -c-> A <-c- P
    MetaGraph("M4_shared_memory", ("fork", "read"), 2),           # P -r-> M <-r- P
    MetaGraph("M5_fork_shared_file", ("fork", "access"), 2),      # P -f-> P, both -a-> F
    MetaGraph("M6_shared_network", ("fork", "connect"), 2),       # P -cn-> N <-cn- P
    MetaGraph("M7_shared_system", ("fork", "open"), 2),           # P -o-> S <-o- P
    MetaGraph("M8_download_chain", ("connect", "download", "access"), 3),
    #                                                             P -cn-> N -d-> F <-a- P
]
NUM_META_GRAPHS = len(META_GRAPHS)

# Node input features: type one-hot (7) + sensitivity one-hot (3)
# + sensitivity score (1) + log-degree (1).
NUM_SENSITIVITY_LEVELS = 3
F_IN = len(ENTITY_TYPES) + NUM_SENSITIVITY_LEVELS + 2
