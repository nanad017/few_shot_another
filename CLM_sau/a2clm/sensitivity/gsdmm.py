"""GSDMM short-text clustering (Yin & Wang, KDD 2014), used by the
sensitivity grading method to cluster parameters that the statistics-based
TF-DF rule cannot decide (paper Sec. III-B1)."""

import random
from collections import defaultdict
from typing import List


class GSDMM:
    def __init__(self, k: int = 3, alpha: float = 0.1, beta: float = 0.1,
                 iters: int = 15, seed: int = 42):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.iters = iters
        self.rng = random.Random(seed)

    def fit(self, docs: List[List[str]]) -> List[int]:
        """Cluster token-lists; returns a cluster id per document."""
        if not docs:
            return []
        vocab = {w for d in docs for w in d}
        v_size = max(len(vocab), 1)
        d_count = len(docs)

        m_z = [0] * self.k                     # docs per cluster
        n_z = [0] * self.k                     # words per cluster
        n_zw = [defaultdict(int) for _ in range(self.k)]  # word counts per cluster
        labels = [0] * d_count

        for i, doc in enumerate(docs):
            z = self.rng.randrange(self.k)
            labels[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)
            for w in doc:
                n_zw[z][w] += 1

        for _ in range(self.iters):
            moved = 0
            for i, doc in enumerate(docs):
                z_old = labels[i]
                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)
                for w in doc:
                    n_zw[z_old][w] -= 1

                probs = []
                for z in range(self.k):
                    p = (m_z[z] + self.alpha) / (d_count - 1 + self.k * self.alpha)
                    num, den = 1.0, 1.0
                    j = 0
                    word_seen = defaultdict(int)
                    for w in doc:
                        num *= n_zw[z][w] + self.beta + word_seen[w]
                        word_seen[w] += 1
                        den *= n_z[z] + v_size * self.beta + j
                        j += 1
                        if den > 1e250:  # rescale to avoid overflow
                            num /= 1e200
                            den /= 1e200
                    probs.append(p * num / max(den, 1e-300))

                total = sum(probs)
                if total <= 0:
                    z_new = self.rng.randrange(self.k)
                else:
                    r = self.rng.random() * total
                    acc, z_new = 0.0, self.k - 1
                    for z, p in enumerate(probs):
                        acc += p
                        if r <= acc:
                            z_new = z
                            break

                labels[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)
                for w in doc:
                    n_zw[z_new][w] += 1
                if z_new != z_old:
                    moved += 1
            if moved == 0:
                break
        return labels
