"""Re-implementation of:

Tawfik et al., "Few-shot android malware classification with quantum-enhanced
prototypical learning and drift detection", Scientific Reports 16:10744 (2026).
https://doi.org/10.1038/s41598-026-45738-0
"""

from .config import Config
from .preprocessing import Preprocessor
from .protonet import EmbeddingNet, prototypes, proto_logits, prototypical_loss
from .quantum import HybridQuantumClassifier
