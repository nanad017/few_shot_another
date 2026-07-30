"""Episodic training of the prototypical network (Algorithm 2) and evaluation."""

import numpy as np
import torch

from .config import Config
from .episodes import EpisodeSampler
from .protonet import EmbeddingNet, prototypes, proto_logits, prototypical_loss


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_protonet(X_train, y_train, cfg: Config, X_val=None, y_val=None,
                   episodes=None, log_every=200, device="cpu"):
    """Algorithm 2: episodic training with Adam and StepLR(step=1000, gamma=0.5)."""
    episodes = episodes or cfg.episodes
    set_seed(cfg.seed)
    model = EmbeddingNet(X_train.shape[1], cfg.hidden1, cfg.hidden2,
                         cfg.embedding_dim, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step,
                                            gamma=cfg.lr_gamma)
    sampler = EpisodeSampler(X_train, y_train, cfg.n_way, cfg.k_shot,
                             cfg.q_query, cfg.seed)
    history = []
    model.train()
    for ep in range(1, episodes + 1):
        sx, sy, qx, qy = sampler.sample()
        sx = torch.as_tensor(sx, dtype=torch.float32, device=device)
        qx = torch.as_tensor(qx, dtype=torch.float32, device=device)
        sy = torch.as_tensor(sy, dtype=torch.long, device=device)
        qy = torch.as_tensor(qy, dtype=torch.long, device=device)

        emb = model(torch.cat([sx, qx]))
        s_emb, q_emb = emb[: len(sx)], emb[len(sx):]
        protos = prototypes(s_emb, sy, cfg.n_way)
        loss = prototypical_loss(q_emb, qy, protos)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if ep % log_every == 0 or ep == episodes:
            entry = {"episode": ep, "loss": float(loss.item())}
            if X_val is not None:
                entry["val_acc"] = episodic_accuracy(model, X_val, y_val, cfg,
                                                     episodes=50, device=device)
            history.append(entry)
            msg = f"episode {ep:5d}  loss {entry['loss']:.4f}"
            if "val_acc" in entry:
                msg += f"  val 5-shot acc {entry['val_acc']*100:.2f}%"
            print(msg)
    return model, history


@torch.no_grad()
def episodic_accuracy(model, X, y, cfg: Config, episodes=500, device="cpu",
                      seed=None):
    """Mean N-way K-shot accuracy over evaluation episodes."""
    model.eval()
    sampler = EpisodeSampler(X, y, cfg.n_way, cfg.k_shot, cfg.q_query,
                             seed if seed is not None else cfg.seed + 1)
    correct = total = 0
    for _ in range(episodes):
        sx, sy, qx, qy = sampler.sample()
        s_emb = model(torch.as_tensor(sx, dtype=torch.float32, device=device))
        q_emb = model(torch.as_tensor(qx, dtype=torch.float32, device=device))
        protos = prototypes(s_emb, torch.as_tensor(sy, dtype=torch.long,
                                                   device=device), cfg.n_way)
        pred = proto_logits(q_emb, protos).argmax(dim=1).cpu().numpy()
        correct += int((pred == qy).sum())
        total += len(qy)
    model.train()
    return correct / total


@torch.no_grad()
def full_test_evaluation(model, X_support, y_support, X_test, y_test,
                         cfg: Config, device="cpu"):
    """Deployment-phase evaluation: build one prototype per class from K support
    samples drawn from X_support, then classify the entire test set."""
    model.eval()
    rng = np.random.default_rng(cfg.seed)
    classes = np.unique(y_support)
    protos = []
    for c in classes:
        idx = rng.choice(np.where(y_support == c)[0], size=cfg.k_shot,
                         replace=False)
        emb = model(torch.as_tensor(X_support[idx], dtype=torch.float32,
                                    device=device))
        protos.append(emb.mean(dim=0))
    protos = torch.stack(protos)

    preds = []
    for i in range(0, len(X_test), 4096):
        emb = model(torch.as_tensor(X_test[i:i + 4096], dtype=torch.float32,
                                    device=device))
        preds.append(proto_logits(emb, protos).argmax(dim=1).cpu().numpy())
    y_pred = classes[np.concatenate(preds)]
    return y_pred
