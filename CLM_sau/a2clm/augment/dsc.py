"""Direct system calls attack (Sec. III-C4): malware evades API hooks by
issuing syscalls directly, so the corresponding ntdll.dll API nodes vanish
from the behavior graph. We remove key API nodes (V_api) and their incident
edges: G_p.4 = (V - V_api, E restricted, S_{V - V_api})."""

import random

from ..graph import induced_subgraph
from ..schema import ENTITY_INDEX

# APIs commonly replaced by direct syscalls (paper Sec. III-C4 + [39]).
DSC_APIS = [
    "ntadjustprivilegestoken", "ntwritevirtualmemory", "ntdeletevaluekey",
    "ntcreatethreadex", "ntqueueapcthread", "ntopenprocess",
    "ntprotectvirtualmemory", "ntallocatevirtualmemory", "ntreadvirtualmemory",
    "ntcreatesection", "ntmapviewofsection", "ntresumethread",
    "writevirtualmemory", "adjustprivilegestoken", "deletevaluekey",
    "createremotethread", "queueapcthread",
]


class DirectSystemCallsAttack:
    name = "dsc"

    def __init__(self, drop_prob: float = 1.0, seed: int | None = None):
        self.drop_prob = drop_prob
        self.rng = random.Random(seed)

    def __call__(self, g):
        api_t = ENTITY_INDEX["api"]
        drop = set()
        for i, (t, name) in enumerate(zip(g.node_types.tolist(), g.node_names)):
            if t != api_t:
                continue
            low = name.lower()
            if any(k in low for k in DSC_APIS) and self.rng.random() < self.drop_prob:
                drop.add(i)
        if not drop:
            return g.clone()
        keep = [i for i in range(g.num_nodes) if i not in drop]
        return induced_subgraph(g, keep)
