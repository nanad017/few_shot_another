"""Self-supervised contrastive training of A2-CLM (Algorithm 1).

Usage:
  python -m a2clm.train --metadata data/processed/metadata.csv \
      --config configs/default.yaml --out runs/exp1 [--device cuda]
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from .augment import (AttributeMaskingAttack, DirectSystemCallsAttack,
                      MetaGraphSamplingAttack, ObfuscationAttack, PGDAttack)
from .data import (GraphDataset, fit_grader, read_metadata, split_families)
from .loss import info_nce
from .model import A2CLM
from .utils import load_config, pick_device, set_seed


def build_augmentations(cfg: dict, seed: int):
    aug_cfg = cfg["augment"]
    static = []
    if aug_cfg["mask"].get("ratio", 0) > 0:
        static.append(AttributeMaskingAttack(
            ratio=aug_cfg["mask"]["ratio"], mu=aug_cfg["mask"]["mu"],
            sigma=aug_cfg["mask"]["sigma"]))
    if aug_cfg["sampling"]["enabled"]:
        static.append(MetaGraphSamplingAttack(
            num_metagraphs=aug_cfg["sampling"].get("num_metagraphs", 2),
            seed=seed))
    if aug_cfg["dsc"]["enabled"]:
        static.append(DirectSystemCallsAttack(seed=seed))
    if aug_cfg["obfuscation"]["enabled"]:
        static.append(ObfuscationAttack(
            garbage_nodes=aug_cfg["obfuscation"].get("garbage_nodes", 3),
            seed=seed))
    pgd = None
    if aug_cfg["pgd"]["instances"] > 0:
        pgd = PGDAttack(epsilon=aug_cfg["pgd"]["epsilon"],
                        steps=aug_cfg["pgd"]["steps"])
    return static, pgd, aug_cfg["pgd"]["instances"]


def train(cfg: dict, metadata: str, out_dir: str, device_arg: str = "auto",
          fmt: str = "auto", epochs_override: int | None = None,
          limit: int | None = None) -> str:
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    device = pick_device(device_arg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metas = read_metadata(metadata)
    if limit:
        metas = metas[:limit]
    train_m, val_m, test_m = split_families(metas, seed=tcfg["seed"])
    print(f"families: train={len({m.family for m in train_m})} "
          f"val={len({m.family for m in val_m})} "
          f"test={len({m.family for m in test_m})} | "
          f"samples: {len(train_m)}/{len(val_m)}/{len(test_m)}")

    # Sensitivity grading fitted on training data only (Sec. III-B1).
    grader = fit_grader(train_m, fmt=fmt, seed=tcfg["seed"])
    grader.save(str(out / "grader.json"))

    train_ds = GraphDataset(train_m, grader, fmt=fmt)
    graphs = [g.to(device) for g in train_ds.graphs]

    model = A2CLM(hidden_dim=cfg["model"]["hidden_dim"],
                  num_layers=cfg["model"]["num_layers"],
                  proj_dim=cfg["model"]["proj_dim"],
                  dropout=cfg["model"]["dropout"]).to(device)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    static_augs, pgd, pgd_instances = build_augmentations(cfg, tcfg["seed"])
    rng = random.Random(tcfg["seed"])
    epochs = epochs_override or tcfg["epochs"]
    bs = tcfg["batch_size"]
    tau = tcfg["temperature"]
    n_neg = tcfg["negatives"]

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(graphs)))
        rng.shuffle(order)
        epoch_loss, n_batches = 0.0, 0
        pbar = tqdm(range(0, len(order), bs), desc=f"epoch {epoch}/{epochs}")
        for start in pbar:
            batch = [graphs[i] for i in order[start:start + bs]]
            if len(batch) < 2:
                continue
            # Negative pool: the batch encoded by the momentum branch q.
            with torch.no_grad():
                zq = torch.stack([model.embed(g, "q") for g in batch])

            opt.zero_grad()
            batch_loss = 0.0
            for i, g in enumerate(batch):
                # negatives: other samples, preferring different families
                cand = [j for j in range(len(batch))
                        if j != i and batch[j].family != g.family]
                if not cand:
                    cand = [j for j in range(len(batch)) if j != i]
                neg_idx = rng.sample(cand, min(n_neg, len(cand)))
                z_negs = zq[neg_idx]

                z_o = model.embed(g, "o")

                # m positive instances: k PGD + static attacks (line 1, Alg. 1)
                positives = []
                for _ in range(pgd_instances):
                    positives.append((g, pgd(model, g)))
                for aug in static_augs:
                    positives.append((aug(g).to(device), None))

                l_anchor = 0.0
                for gp, delta in positives:
                    with torch.no_grad():
                        z_p = model.embed(gp, "p", delta=delta)
                    l_anchor = l_anchor + info_nce(z_o, z_p, z_negs, tau)
                batch_loss = batch_loss + l_anchor / max(len(positives), 1)

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            opt.step()
            model.momentum_update(tcfg["momentum1"], tcfg["momentum2"])

            epoch_loss += float(batch_loss.detach())
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss / max(n_batches, 1):.4f}")

        avg = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg})
        torch.save({"model": model.state_dict(), "config": cfg,
                    "epoch": epoch}, out / "checkpoint.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "splits.json").write_text(json.dumps({
        "train": [m.sample_id for m in train_m],
        "val": [m.sample_id for m in val_m],
        "test": [m.sample_id for m in test_m],
    }, indent=2))
    print(f"saved checkpoint + grader + splits to {out}")
    return str(out / "checkpoint.pt")


def main():
    ap = argparse.ArgumentParser(description="Train A2-CLM (Algorithm 1)")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--format", default="auto", choices=["auto", "json", "xml"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="override config epochs")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: cap number of samples")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, args.metadata, args.out, args.device, args.format,
          args.epochs, args.limit)


if __name__ == "__main__":
    main()
