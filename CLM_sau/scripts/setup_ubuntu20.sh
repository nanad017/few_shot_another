#!/usr/bin/env bash
# ============================================================================
# Tu dong build A2-CLM tren Ubuntu 20.04 (chay duoc ca Ubuntu 22/24).
#
# Cach dung (tu thu muc goc cua project, sau khi giai nen a2clm.tar.gz):
#   bash scripts/setup_ubuntu20.sh            # tu phat hien GPU
#   bash scripts/setup_ubuntu20.sh --cpu      # ep dung ban torch CPU
#   bash scripts/setup_ubuntu20.sh --no-test  # bo qua smoke test cuoi cung
#
# Script se:
#   1. Tim Python >= 3.10 (Ubuntu 20 mac dinh chi co 3.8):
#        - co san python3.10/3.11/3.12  -> dung luon
#        - co sudo                      -> cai qua PPA deadsnakes
#        - khong co sudo                -> tai uv, uv tu tai Python 3.12
#   2. Tao virtualenv .venv
#   3. Cai PyTorch (CUDA neu thay nvidia-smi, nguoc lai ban CPU)
#   4. Cai cac dependency con lai + package a2clm
#   5. Chay smoke test end-to-end de xac nhan moi truong OK
# ============================================================================
set -euo pipefail

FORCE_CPU=0
RUN_TEST=1
for arg in "$@"; do
    case "$arg" in
        --cpu)     FORCE_CPU=1 ;;
        --no-test) RUN_TEST=0 ;;
        *) echo "Tham so khong ho tro: $arg"; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[x] %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------- 1. Python
find_python() {
    for p in python3.12 python3.11 python3.10; do
        if command -v "$p" >/dev/null 2>&1; then
            echo "$p"; return 0
        fi
    done
    # python3 co the da du moi (Ubuntu 22/24)
    if command -v python3 >/dev/null 2>&1; then
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
            echo "python3"; return 0
        fi
    fi
    return 1
}

PY="$(find_python || true)"

if [ -z "$PY" ]; then
    warn "Khong tim thay Python >= 3.10 (Ubuntu 20 mac dinh chi co 3.8)."
    if sudo -n true 2>/dev/null || { [ -t 0 ] && sudo -v; }; then
        log "Cai Python 3.10 qua PPA deadsnakes (can sudo)..."
        sudo apt-get update -y
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update -y
        sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils
        PY=python3.10
    else
        log "Khong co sudo -> dung uv de tai Python 3.12 standalone..."
        if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        export PATH="$HOME/.local/bin:$PATH"
        command -v uv >/dev/null 2>&1 || die "Cai uv that bai."
        uv python install 3.12
        log "Tao virtualenv .venv (uv)..."
        uv venv -p 3.12 .venv
        PY=""   # venv da duoc tao, khong can PY nua
    fi
fi

# ------------------------------------------------------------ 2. virtualenv
if [ ! -x .venv/bin/python ]; then
    log "Tao virtualenv .venv voi $PY..."
    "$PY" -m venv .venv 2>/dev/null || {
        warn "Thieu module venv, thu cai python3-venv (can sudo)..."
        sudo apt-get install -y "${PY}-venv" || sudo apt-get install -y python3-venv
        "$PY" -m venv .venv
    }
fi
VPY="$ROOT/.venv/bin/python"
"$VPY" -c 'import sys; assert sys.version_info >= (3,10), sys.version' \
    || die "Python trong .venv < 3.10. Xoa .venv roi chay lai script."
log "Dung Python: $("$VPY" --version) tai $VPY"

# venv tao boi uv khong kem pip -> tu bootstrap
if ! "$VPY" -m pip --version >/dev/null 2>&1; then
    log "Bootstrap pip vao .venv..."
    "$VPY" -m ensurepip --upgrade >/dev/null 2>&1 \
        || { curl -LsSf https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
             && "$VPY" /tmp/get-pip.py; }
    "$VPY" -m pip --version >/dev/null 2>&1 || die "Khong cai duoc pip vao .venv."
fi
"$VPY" -m pip install --upgrade pip >/dev/null 2>&1 || true

# --------------------------------------------------------------- 3. PyTorch
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if [ "$FORCE_CPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi -L >/dev/null 2>&1; then
    log "Phat hien GPU: $(nvidia-smi -L | head -1)"
    # cu121 chay duoc voi moi driver >= 525; doi sang cu124/cu126 neu muon
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
else
    log "Khong dung GPU -> cai torch ban CPU."
fi

if "$VPY" -c 'import torch' >/dev/null 2>&1; then
    log "torch da co san: $("$VPY" -c 'import torch; print(torch.__version__)')"
else
    log "Cai PyTorch tu $TORCH_INDEX ..."
    "$VPY" -m pip install torch --index-url "$TORCH_INDEX" \
        || { warn "Cai ban GPU that bai, roi ve ban CPU..."; \
             "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu; }
fi

# --------------------------------------------- 4. dependency + package a2clm
log "Cai dependency + package a2clm..."
"$VPY" -m pip install -r requirements.txt
"$VPY" -m pip install -e . --no-deps

# ------------------------------------------------------------- 5. smoke test
if [ "$RUN_TEST" -eq 1 ]; then
    log "Chay smoke test end-to-end (du lieu tong hop, ~1-2 phut)..."
    "$VPY" scripts/generate_synthetic.py --out data_synth --per-family 12 --benign 24
    "$VPY" -m a2clm.train \
        --metadata data_synth/processed/metadata.csv \
        --config configs/smoke.yaml --out runs/smoke --epochs 3
    "$VPY" -m a2clm.evaluate --run runs/smoke \
        --metadata data_synth/processed/metadata.csv
    log "SMOKE TEST OK - moi truong san sang."
fi

cat <<EOF

============================================================================
Build xong. Cach chay voi du lieu that:

  source .venv/bin/activate
  python -m a2clm.train \\
      --metadata data/processed/metadata.csv \\
      --config configs/default.yaml \\
      --out runs/exp1 --device auto
  python -m a2clm.evaluate --run runs/exp1 \\
      --metadata data/processed/metadata.csv

GPU: $( [ "$FORCE_CPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1 \
        && echo "co, torch se tu dung CUDA" || echo "khong dung / khong co" )
============================================================================
EOF
