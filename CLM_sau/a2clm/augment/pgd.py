"""PGD attack (Sec. III-C1, Eqs. 2-4). Node attributes are first projected
into a common space by the per-type matrices W_T (Eq. 2, implemented as the
encoder's typed input projection); the perturbation delta is then optimized
by projected gradient ascent inside an l_inf ball of radius epsilon to
maximize the contrastive disagreement (Eq. 4), yielding the augmented
instance G_p.1 = (V, E, S + delta)."""

import torch
import torch.nn.functional as F


class PGDAttack:
    name = "pgd"

    def __init__(self, epsilon: float = 0.05, steps: int = 2,
                 step_size: float | None = None):
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size if step_size is not None else 1.5 * epsilon / max(steps, 1)

    def __call__(self, model, g) -> torch.Tensor:
        """Returns delta [N, d_hidden] to be added to the projected node
        features when encoding g. Gradients flow only into delta (via
        autograd.grad), never into the encoder parameters."""
        enc, head = model.enc_o, model.head_o
        with torch.no_grad():
            h0 = enc.input_proj(g)
            z_ref = F.normalize(head(enc.forward_projected(g, h0)), dim=-1)

        delta = torch.zeros_like(h0, requires_grad=True)
        for _ in range(self.steps):
            z_adv = F.normalize(head(enc.forward_projected(g, h0 + delta)), dim=-1)
            loss = 1.0 - (z_adv * z_ref).sum()          # push apart (Eq. 4)
            (grad,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta += self.step_size * grad.sign()
                delta.clamp_(-self.epsilon, self.epsilon)
        return delta.detach()
