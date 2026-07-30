"""Few-shot evaluation (paper Sec. IV-B/C): c-way n-shot episodes on the
held-out TEST families. Queries are classified by cosine similarity to the
support-set prototypes in the learned embedding space (the instance
discriminator of Sec. III-D3). Reports ACC / precision / recall / F1
(macro) and, when a benign family is present, binary detection AUC.

Usage:
  python -m a2clm.evaluate --run runs/exp1 --metadata data/processed/metadata.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .data import GraphDataset, episodes, read_metadata, split_families
from .model import A2CLM
from .sensitivity import SensitivityGrader
from .utils import pick_device, set_seed


@torch.no_grad()
def embed_all(model: A2CLM, ds: GraphDataset, device) -> dict:
    model.eval()
    out = {}
    for g in ds.graphs:
        out[g.sample_id] = model.embed(g.to(device), "o").cpu().numpy()
    return out


def run_episode(emb, support, query):
    protos, fams = [], []
    by_fam = defaultdict(list)
    for m in support:
        if m.sample_id in emb:
            by_fam[m.family].append(emb[m.sample_id])
    for fam, vecs in by_fam.items():
        v = np.mean(vecs, axis=0)
        protos.append(v / (np.linalg.norm(v) + 1e-12))
        fams.append(fam)
    P = np.stack(protos)                      # [C, D]

    y_true, y_pred, mal_scores, bin_true = [], [], [], []
    fam_is_mal = {fam: fam.lower() != "benign" for fam in fams}
    for m in query:
        z = emb.get(m.sample_id)
        if z is None:
            continue
        sims = P @ z
        pred = fams[int(np.argmax(sims))]
        y_true.append(m.family)
        y_pred.append(pred)
        if "benign" in (f.lower() for f in fams):
            mal_sim = max(s for s, f in zip(sims, fams) if fam_is_mal[f])
            ben_sim = max(s for s, f in zip(sims, fams) if not fam_is_mal[f])
            mal_scores.append(float(mal_sim - ben_sim))
            bin_true.append(int(m.family.lower() != "benign"))

    res = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro",
                               zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if bin_true and len(set(bin_true)) == 2:
        res["auc"] = roc_auc_score(bin_true, mal_scores)
    return res


def evaluate(run_dir: str, metadata: str, device_arg: str = "auto",
             fmt: str = "auto", split: str = "test"):
    run = Path(run_dir)
    ckpt = torch.load(run / "checkpoint.pt", map_location="cpu",
                      weights_only=False)
    cfg = ckpt["config"]
    device = pick_device(device_arg)
    set_seed(cfg["train"]["seed"])

    model = A2CLM(hidden_dim=cfg["model"]["hidden_dim"],
                  num_layers=cfg["model"]["num_layers"],
                  proj_dim=cfg["model"]["proj_dim"],
                  dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    grader = SensitivityGrader.load(str(run / "grader.json"))

    metas = read_metadata(metadata)
    train_m, val_m, test_m = split_families(metas, seed=cfg["train"]["seed"])
    eval_m = {"train": train_m, "val": val_m, "test": test_m}[split]
    ds = GraphDataset(eval_m, grader, fmt=fmt)
    emb = embed_all(model, ds, device)

    results = {}
    for shot in cfg["eval"]["shots"]:
        runs = []
        for ep_i, (support, query) in enumerate(episodes(
                eval_m, shot, cfg["eval"]["episodes"],
                cfg["eval"]["query_per_family"],
                seed=cfg["train"]["seed"])):
            if not support or not query:
                continue
            runs.append(run_episode(emb, support, query))
        if not runs:
            continue
        agg = {k: (float(np.mean([r[k] for r in runs if k in r])),
                   float(np.std([r[k] for r in runs if k in r])))
               for k in runs[0]}
        results[f"{shot}-shot"] = agg
        line = " ".join(f"{k}={m*100:.2f}±{s*100:.2f}"
                        for k, (m, s) in agg.items())
        print(f"[{split}] {shot}-shot ({len(runs)} episodes): {line}")

    (run / f"eval_{split}.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    ap = argparse.ArgumentParser(description="Few-shot evaluation of A2-CLM")
    ap.add_argument("--run", required=True, help="training output dir")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--format", default="auto", choices=["auto", "json", "xml"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()
    evaluate(args.run, args.metadata, args.device, args.format, args.split)


if __name__ == "__main__":
    main()
