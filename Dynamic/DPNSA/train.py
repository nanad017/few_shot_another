"""Episodic training of DPNSA (Section 4.1 of the paper).

Defaults follow the paper: Adam, lr 1e-3, 1000 epochs x 100 episodes,
step LR decay every 20 epochs, early-stopping patience 200 epochs,
horizontal-flip augmentation, best epoch selected on the validation set.

Example (on the training machine):
    python train.py --data-root data --n-way 5 --k-shot 5 --n-query 15
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dpnsa import DPNSA
from dpnsa.data import MalwareImageFolder


def get_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, dataset, episodes, n_way, k_shot, n_query, device):
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(episodes):
            s, q, y = dataset.sample_episode(n_way, k_shot, n_query)
            _, acc, _ = model.episode_loss(
                s.to(device), q.to(device), y.to(device), n_way, k_shot
            )
            accs.append(acc.item())
    accs = np.array(accs)
    ci95 = 1.96 * accs.std() / np.sqrt(len(accs))
    return accs.mean(), ci95


def main():
    p = argparse.ArgumentParser(description="Train DPNSA")
    p.add_argument("--data-root", default="data", help="folder with train/ val/ test/")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=15,
                   help="queries per class (paper: 15 for 1/5-shot, 10 for 10-shot)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--episodes-per-epoch", type=int, default=100)
    p.add_argument("--val-episodes", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-step", type=int, default=20)
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--variant", choices=["ds1", "ds2"], default="ds1")
    p.add_argument("--num-kernels", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--temp-anneal-epochs", type=int, default=10,
                   help="epochs to anneal dynamic-conv softmax temperature 30 -> 1")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/dpnsa")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true",
                   help="tiny run (2 epochs x 5 episodes) to verify the setup")
    args = p.parse_args()

    if args.smoke_test:
        args.epochs, args.episodes_per_epoch, args.val_episodes = 2, 5, 5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    train_set = MalwareImageFolder(
        Path(args.data_root) / "train", args.image_size, augment=True
    )
    val_set = MalwareImageFolder(Path(args.data_root) / "val", args.image_size)
    print(f"train classes: {len(train_set.classes)}, val classes: {len(val_set.classes)}")

    model = DPNSA(num_kernels=args.num_kernels, variant=args.variant).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    best_acc, best_epoch = 0.0, -1
    history = []
    for epoch in range(args.epochs):
        # Anneal the dynamic-convolution attention temperature 30 -> 1.
        if args.temp_anneal_epochs > 0:
            t = max(30.0 - epoch * 29.0 / args.temp_anneal_epochs, 1.0)
            model.set_temperature(t)

        model.train()
        t0 = time.time()
        losses, accs = [], []
        for _ in range(args.episodes_per_epoch):
            s, q, y = train_set.sample_episode(args.n_way, args.k_shot, args.n_query)
            loss, acc, _ = model.episode_loss(
                s.to(device), q.to(device), y.to(device), args.n_way, args.k_shot
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            accs.append(acc.item())
        sched.step()

        val_acc, val_ci = evaluate(
            model, val_set, args.val_episodes, args.n_way, args.k_shot, args.n_query, device
        )
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)),
             "train_acc": float(np.mean(accs)), "val_acc": val_acc, "val_ci95": val_ci}
        )
        print(
            f"epoch {epoch:4d} | loss {np.mean(losses):.4f} | "
            f"train acc {np.mean(accs):.4f} | val acc {val_acc:.4f} ± {val_ci:.4f} | "
            f"{time.time() - t0:.1f}s"
        )

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                 "val_acc": val_acc},
                out_dir / "best.pt",
            )
        torch.save(
            {"model": model.state_dict(), "args": vars(args), "epoch": epoch},
            out_dir / "last.pt",
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if epoch - best_epoch >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    print(f"best val acc {best_acc:.4f} at epoch {best_epoch} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
