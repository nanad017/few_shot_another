"""Network schema of A2-CLM (paper Sec. III-B, Fig. 3(b) and Fig. 4).

7 entity types, 8 canonical relations and 8 meta-graphs. The meta-graphs in
Fig. 4 are drawn as small patterns anchored at process nodes (fork chains,
shared API / registry / memory / file / system / network objects, and the
network-download-file chain). Each pattern is encoded as one or more ordered,
directed walks so Eq. 6 cannot admit unrelated paths that merely reuse the
same relation names.
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
    # Each walk is a sequence of (relation, direction) steps. Direction +1
    # follows the canonical relation; -1 follows its mirrored edge.
    walks: Tuple[Tuple[Tuple[str, int], ...], ...]
    # M5/M6 are fork-plus-shared-resource patterns. The resource must be
    # reachable from both processes, not merely from either process.
    shared_relation: str | None = None

    @property
    def relations(self) -> Tuple[str, ...]:
        """Base relations used by the pattern, kept for encoder filtering."""
        return tuple(dict.fromkeys(
            relation for walk in self.walks for relation, _ in walk
        ))

    @property
    def diameter(self) -> int:
        return max((len(walk) for walk in self.walks), default=0)


# Fig. 4: eight meta-graphs M1..M8.
META_GRAPHS: List[MetaGraph] = [
    MetaGraph("M1_fork", ((("fork", +1),),)),
    MetaGraph("M2_shared_registry", ((("set", +1), ("set", -1)),)),
    MetaGraph("M3_shared_api", ((("call", +1), ("call", -1)),)),
    MetaGraph("M4_shared_memory", ((("read", +1), ("read", -1)),)),
    MetaGraph("M5_fork_shared_file", (
        (("fork", +1),),
        (("access", +1),),
        (("fork", +1), ("access", +1)),
    ), shared_relation="access"),
    MetaGraph("M6_shared_network", (
        (("fork", +1),),
        (("connect", +1),),
        (("fork", +1), ("connect", +1)),
    ), shared_relation="connect"),
    MetaGraph("M7_shared_system", ((("open", +1), ("open", -1)),)),
    MetaGraph("M8_download_chain", (
        (("connect", +1), ("download", +1), ("access", -1)),
    )),
]
NUM_META_GRAPHS = len(META_GRAPHS)

# Node input features: type one-hot (7) + sensitivity one-hot (3)
# + sensitivity score (1) + log-degree (1).
NUM_SENSITIVITY_LEVELS = 3
F_IN = len(ENTITY_TYPES) + NUM_SENSITIVITY_LEVELS + 2
