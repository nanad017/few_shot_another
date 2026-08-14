"""Training loop for the hybrid quantum-classical classifier (Fig. 1D pathway).

The hybrid model is trained with cross-entropy on the CatBoost-selected
features, as described in the paper's quantum-enhanced hybrid layer section.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .quantum import HybridQuantumClassifier


def train_quantum_classifier(X_train, y_train, X_val, y_val, n_classes,
                             epochs=4000, batch_size=256, lr=1e-3, seed=42,
                             lr_step=1000, lr_gamma=0.5, n_qubits=4,
                             alpha=0.5, post_hidden=16, device="cpu",
                             verbose=True):
    """Train the paper's alternative quantum classification pathway.

    The paper lists the same Adam + StepLR schedule as the main framework.
    ``epochs`` is configurable so smoke tests and CPU experiments can use a
    short run without changing the default paper configuration.
    """
    torch.manual_seed(seed)
    model = HybridQuantumClassifier(
        X_train.shape[1], n_classes, n_qubits=n_qubits, alpha=alpha,
        post_hidden=post_hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=lr_step, gamma=lr_gamma
    )

    Xt = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
        sched.step()
        if verbose and (epoch % 5 == 0 or epoch == epochs):
            model.eval()
            with torch.no_grad():
                pred = model(Xv).argmax(dim=1).cpu().numpy()
            acc = float((pred == y_val).mean())
            print(f"epoch {epoch:3d}  loss {total / len(Xt):.4f}  "
                  f"val acc {acc*100:.2f}%")
    return model


@torch.no_grad()
def predict(model, X, device="cpu", batch_size=4096):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32,
                             device=device)
        out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)
