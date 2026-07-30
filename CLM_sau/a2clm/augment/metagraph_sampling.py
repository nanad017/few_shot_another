"""Meta-graph-guide sampling attack (Sec. III-C3, Eq. 6): keep the
neighborhood N^(i) visited when the target process walks along a randomly
chosen meta-graph M_i."""

import random

from ..graph import induced_subgraph
from ..schema import META_GRAPHS


class MetaGraphSamplingAttack:
    name = "sample"

    def __init__(self, num_metagraphs: int = 2, seed: int | None = None):
        self.num_metagraphs = num_metagraphs
        self.rng = random.Random(seed)

    def __call__(self, g):
        mgs = self.rng.sample(META_GRAPHS, k=min(self.num_metagraphs,
                                                 len(META_GRAPHS)))
        nodes = set()
        for mg in mgs:
            nodes |= g.metagraph_neighborhood(mg)
        if len(nodes) < 2:  # degenerate sample: fall back to the full graph
            return g.clone()
        return induced_subgraph(g, nodes)
