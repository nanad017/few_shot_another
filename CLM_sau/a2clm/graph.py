"""Sensitivity Heterogeneous Graph of Few-shot Malware (SHGFM, Def. 2 and
Sec. III-B2). A graph G_o = (A_o, S_o): typed nodes, typed edges and a
sensitivity attribute matrix. Edges are stored canonically and mirrored
(reverse relation ids offset by NUM_RELATIONS) so message passing is
bidirectional."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch

from .schema import (ENTITY_INDEX, ENTITY_TYPES, F_IN, META_GRAPHS,
                     NUM_RELATIONS, NUM_SENSITIVITY_LEVELS, RELATION_INDEX,
                     TYPE_PAIR_TO_RELATION, MetaGraph)


@dataclass
class SHGFM:
    node_types: torch.Tensor          # long [N]
    node_names: List[str]             # entity identifier per node
    node_sens: torch.Tensor           # long [N], values in {1,2,3}
    x: torch.Tensor                   # float [N, F_IN]  (= S_o plus type/degree)
    edge_index: torch.Tensor          # long [2, E], includes mirrored edges
    edge_types: torch.Tensor          # long [E]; reversed edges offset by NUM_RELATIONS
    target: int                       # index of the target process node Tar
    label: int = -1                   # 1 malware / 0 benign, -1 unknown
    family: str = ""
    sample_id: str = ""
    _mg_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, repr=False)

    @property
    def num_nodes(self) -> int:
        return int(self.node_types.shape[0])

    def clone(self) -> "SHGFM":
        return SHGFM(self.node_types.clone(), list(self.node_names),
                     self.node_sens.clone(), self.x.clone(),
                     self.edge_index.clone(), self.edge_types.clone(),
                     self.target, self.label, self.family, self.sample_id)

    def to(self, device) -> "SHGFM":
        self.node_types = self.node_types.to(device)
        self.node_sens = self.node_sens.to(device)
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_types = self.edge_types.to(device)
        self._mg_cache = {k: (n.to(device), e.to(device))
                          for k, (n, e) in self._mg_cache.items()}
        return self

    # ---------------------------------------------------- meta-graph views
    def metagraph_neighborhood(self, mg: MetaGraph) -> Set[int]:
        """Eq. 6: nodes visited when the target process walks along M_i
        (BFS through edges whose base relation belongs to M_i, bounded by
        the pattern diameter)."""
        allowed = {RELATION_INDEX[r] for r in mg.relations}
        base = self.edge_types % NUM_RELATIONS
        keep = torch.isin(base, torch.tensor(sorted(allowed),
                                             device=self.edge_types.device))
        ei = self.edge_index[:, keep]
        adj: Dict[int, List[int]] = {}
        for s, d in ei.t().tolist():
            adj.setdefault(s, []).append(d)
        frontier, visited = {self.target}, {self.target}
        for _ in range(mg.diameter):
            nxt = set()
            for u in frontier:
                for v in adj.get(u, ()):  # mirrored edges make this symmetric
                    if v not in visited:
                        visited.add(v)
                        nxt.add(v)
            frontier = nxt
            if not frontier:
                break
        return visited

    def metagraph_edges(self, mg_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Edge subset used by the encoder for meta-graph M_i: edges whose
        base relation occurs in M_i and whose endpoints lie inside the
        meta-graph-guided neighborhood. Cached per graph (results are
        deterministic)."""
        if mg_idx in self._mg_cache:
            return self._mg_cache[mg_idx]
        mg = META_GRAPHS[mg_idx]
        nodes = self.metagraph_neighborhood(mg)
        allowed = {RELATION_INDEX[r] for r in mg.relations}
        base = self.edge_types % NUM_RELATIONS
        keep_rel = torch.isin(base, torch.tensor(sorted(allowed),
                                                 device=self.edge_types.device))
        node_mask = torch.zeros(self.num_nodes, dtype=torch.bool,
                                device=self.edge_index.device)
        node_mask[list(nodes)] = True
        keep = keep_rel & node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
        out = (self.edge_index[:, keep], self.edge_types[keep])
        self._mg_cache[mg_idx] = out
        return out


def _canonical_relation(src_type: str, dst_type: str, relation: str) -> Optional[str]:
    rel = str(relation).lower()
    if rel in RELATION_INDEX:
        return rel
    return TYPE_PAIR_TO_RELATION.get((src_type, dst_type))


def node_features(node_types: torch.Tensor, node_sens: torch.Tensor,
                  edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """S_o rows: type one-hot || sensitivity one-hot || score || log-degree."""
    x = torch.zeros(num_nodes, F_IN)
    x[torch.arange(num_nodes), node_types] = 1.0
    sens_idx = (node_sens.clamp(1, NUM_SENSITIVITY_LEVELS) - 1) + len(ENTITY_TYPES)
    x[torch.arange(num_nodes), sens_idx] = 1.0
    x[:, -2] = node_sens.float() / NUM_SENSITIVITY_LEVELS
    deg = torch.zeros(num_nodes)
    if edge_index.numel():
        deg.index_add_(0, edge_index[0],
                       torch.ones(edge_index.shape[1]))
    x[:, -1] = torch.log1p(deg) / math.log(50.0)
    return x


def build_shgfm(events: Sequence[dict], grader, label: int = -1,
                family: str = "", sample_id: str = "") -> Optional[SHGFM]:
    """Build the SHGFM of one sample from its sensitivity-graded behavior
    events (Sec. III-B2). Events look like:
    {"src_type": "process", "src": "2036", "relation": "call",
     "dst_type": "api", "dst": "NtWriteFile", "parameters": [...]}"""
    node_of: Dict[Tuple[str, str], int] = {}
    node_types: List[int] = []
    node_names: List[str] = []
    node_sens: List[int] = []
    edges: List[Tuple[int, int]] = []
    etypes: List[int] = []
    seen_edges: Set[Tuple[int, int, int]] = set()
    first_process: Optional[int] = None
    child_processes: Set[int] = set()

    def get_node(ntype: str, name: str) -> int:
        key = (ntype, name)
        if key not in node_of:
            node_of[key] = len(node_types)
            node_types.append(ENTITY_INDEX[ntype])
            node_names.append(name)
            node_sens.append(1)
        return node_of[key]

    for ev in events:
        st = str(ev.get("src_type", "")).lower()
        dt = str(ev.get("dst_type", "")).lower()
        if st not in ENTITY_INDEX or dt not in ENTITY_INDEX:
            continue
        rel = _canonical_relation(st, dt, ev.get("relation", ""))
        if rel is None:
            continue
        s = get_node(st, str(ev.get("src")))
        d = get_node(dt, str(ev.get("dst")))
        if st == "process" and first_process is None:
            first_process = s
        if rel == "fork":
            child_processes.add(d)
        sens = grader.grade_event(ev)
        node_sens[d] = max(node_sens[d], sens)
        node_sens[s] = max(node_sens[s], sens)
        rid = RELATION_INDEX[rel]
        if (s, d, rid) not in seen_edges:
            seen_edges.add((s, d, rid))
            edges.append((s, d))
            etypes.append(rid)

    if not node_types:
        return None

    # Target process node Tar: the root of the fork tree.
    target = first_process if first_process is not None else 0
    for i, t in enumerate(node_types):
        if t == ENTITY_INDEX["process"] and i not in child_processes:
            target = i
            break

    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t()
        et = torch.tensor(etypes, dtype=torch.long)
        ei = torch.cat([ei, ei.flip(0)], dim=1)          # mirror edges
        et = torch.cat([et, et + NUM_RELATIONS])
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
        et = torch.zeros(0, dtype=torch.long)

    nt = torch.tensor(node_types, dtype=torch.long)
    ns = torch.tensor(node_sens, dtype=torch.long)
    x = node_features(nt, ns, ei, len(node_types))
    return SHGFM(nt, node_names, ns, x, ei, et, target,
                 int(label), family, sample_id)


def rebuild_features(g: SHGFM) -> None:
    """Recompute S_o after a structural augmentation edited nodes/edges."""
    g.x = node_features(g.node_types, g.node_sens, g.edge_index, g.num_nodes).to(
        g.x.device if g.x.numel() else "cpu")
    g._mg_cache.clear()


def induced_subgraph(g: SHGFM, keep_nodes: Sequence[int]) -> SHGFM:
    """Induced subgraph with node relabeling; used by sampling/DSC attacks."""
    keep = sorted(set(int(i) for i in keep_nodes))
    if g.target not in keep:
        keep.append(g.target)
        keep.sort()
    remap = {old: new for new, old in enumerate(keep)}
    mask = torch.zeros(g.num_nodes, dtype=torch.bool)
    mask[keep] = True
    emask = mask[g.edge_index[0]] & mask[g.edge_index[1]]
    ei = g.edge_index[:, emask]
    ei = torch.tensor([[remap[int(s)] for s in ei[0]],
                       [remap[int(d)] for d in ei[1]]], dtype=torch.long) \
        if ei.numel() else torch.zeros(2, 0, dtype=torch.long)
    sub = SHGFM(
        node_types=g.node_types[keep].clone(),
        node_names=[g.node_names[i] for i in keep],
        node_sens=g.node_sens[keep].clone(),
        x=g.x[keep].clone(),
        edge_index=ei,
        edge_types=g.edge_types[emask].clone(),
        target=remap[g.target],
        label=g.label, family=g.family, sample_id=g.sample_id,
    )
    rebuild_features(sub)
    return sub
