"""Evaluate a trained DPNSA checkpoint: mean accuracy ± 95% CI over
randomly sampled test episodes (paper: 1000 episodes).

    python evaluate.py --checkpoint runs/dpnsa/best.pt --data-root data \
        --split test --n-way 5 --k-shot 5 --episodes 1000
"""

import argparse
from pathlib import Path

import torch

from dpnsa import DPNSA
from dpnsa.data import MalwareImageFolder
from train import evaluate, get_device


def main():
    p = argparse.ArgumentParser(description="Evaluate DPNSA")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=15)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    model = DPNSA(
        num_kernels=train_args.get("num_kernels", 4),
        variant=train_args.get("variant", "ds1"),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    dataset = MalwareImageFolder(Path(args.data_root) / args.split, args.image_size)
    acc, ci = evaluate(
        model, dataset, args.episodes, args.n_way, args.k_shot, args.n_query, device
    )
    print(
        f"{args.split}: {args.n_way}-way {args.k_shot}-shot over {args.episodes} episodes"
        f" -> {acc * 100:.2f}% ± {ci * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
