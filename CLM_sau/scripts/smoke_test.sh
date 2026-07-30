#!/usr/bin/env bash
# End-to-end smoke test on synthetic data (small settings, CPU-friendly).
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/generate_synthetic.py --out data_synth --per-family 12 --benign 24
python -m a2clm.train \
    --metadata data_synth/processed/metadata.csv \
    --config configs/smoke.yaml \
    --out runs/smoke --epochs 3
python -m a2clm.evaluate --run runs/smoke \
    --metadata data_synth/processed/metadata.csv
echo "SMOKE TEST OK"
