"""Obfuscation attack (Sec. III-C5): instruction substitution replaces a
malicious API operation with semantically equivalent benign/sensitive ones
(e.g. CopyFileEx -> NtOpenFile + NtReadFile + NtWriteFile); garbage code
insertion adds useless benign operations. Both enlarge the graph edit
distance (Eq. 7) while retaining semantics."""

import random
import torch

from ..graph import rebuild_features
from ..schema import ENTITY_INDEX, NUM_RELATIONS, RELATION_INDEX

SUBSTITUTIONS = {
    "copyfileex": ["NtOpenFile", "NtReadFile", "NtWriteFile"],
    "copyfile": ["NtOpenFile", "NtReadFile", "NtWriteFile"],
    "movefile": ["NtOpenFile", "NtReadFile", "NtWriteFile", "NtDeleteFile"],
    "writeprocessmemory": ["NtOpenProcess", "NtAllocateVirtualMemory",
                           "NtWriteVirtualMemory"],
    "createremotethread": ["NtOpenProcess", "NtCreateThreadEx"],
    "shellexecute": ["NtCreateProcess", "NtResumeThread"],
    "urldownloadtofile": ["InternetOpen", "InternetReadFile", "NtWriteFile"],
    "regdeletekey": ["RegOpenKeyEx", "NtDeleteValueKey"],
    "winexec": ["NtCreateProcess"],
}

GARBAGE_APIS = ["GetSystemTime", "GetTickCount", "CloseHandle",
                "GetCurrentProcessId", "QueryPerformanceCounter", "Sleep",
                "GetVersionEx", "GetModuleHandle"]


class ObfuscationAttack:
    name = "obfuscation"

    def __init__(self, garbage_nodes: int = 3, seed: int | None = None):
        self.garbage_nodes = garbage_nodes
        self.rng = random.Random(seed)

    def __call__(self, g):
        out = g.clone()
        api_t = ENTITY_INDEX["api"]
        call_r = RELATION_INDEX["call"]

        node_types = out.node_types.tolist()
        node_names = list(out.node_names)
        node_sens = out.node_sens.tolist()
        ei = out.edge_index.tolist()
        et = out.edge_types.tolist()
        edges = list(zip(ei[0], ei[1], et)) if out.edge_index.numel() else []

        def add_node(name: str, sens: int) -> int:
            node_types.append(api_t)
            node_names.append(name)
            node_sens.append(sens)
            return len(node_types) - 1

        def add_call(proc: int, api: int) -> None:
            edges.append((proc, api, call_r))
            edges.append((api, proc, call_r + NUM_RELATIONS))

        # --- instruction substitution -----------------------------------
        removed = set()
        for idx, (t, name) in enumerate(zip(node_types[:], node_names[:])):
            if t != api_t:
                continue
            subs = None
            low = name.lower()
            for key, repl in SUBSTITUTIONS.items():
                if key in low:
                    subs = repl
                    break
            if subs is None:
                continue
            callers = {s for (s, d, r) in edges
                       if d == idx and r % NUM_RELATIONS == call_r and s != idx}
            if not callers:
                continue
            removed.add(idx)
            new_ids = [add_node(n, sens=2) for n in subs]
            for proc in callers:
                for nid in new_ids:
                    add_call(proc, nid)

        edges = [(s, d, r) for (s, d, r) in edges
                 if s not in removed and d not in removed]

        # --- garbage code insertion -------------------------------------
        procs = [i for i, t in enumerate(node_types)
                 if t == ENTITY_INDEX["process"] and i not in removed]
        for _ in range(self.garbage_nodes):
            if not procs:
                break
            api = add_node(self.rng.choice(GARBAGE_APIS), sens=1)
            add_call(self.rng.choice(procs), api)

        # --- compact removed nodes --------------------------------------
        keep = [i for i in range(len(node_types)) if i not in removed]
        remap = {old: new for new, old in enumerate(keep)}
        out.node_types = torch.tensor([node_types[i] for i in keep])
        out.node_names = [node_names[i] for i in keep]
        out.node_sens = torch.tensor([node_sens[i] for i in keep])
        edges = [(remap[s], remap[d], r) for (s, d, r) in edges]
        if edges:
            out.edge_index = torch.tensor([[e[0] for e in edges],
                                           [e[1] for e in edges]])
            out.edge_types = torch.tensor([e[2] for e in edges])
        else:
            out.edge_index = torch.zeros(2, 0, dtype=torch.long)
            out.edge_types = torch.zeros(0, dtype=torch.long)
        out.target = remap.get(out.target, 0)
        rebuild_features(out)
        return out
