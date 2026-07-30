#!/usr/bin/env bash
# Tự động cài đặt DPNSA trên máy mới (Ubuntu 20.04 trở lên, cả CPU lẫn GPU).
#
#   ./setup.sh              # tự phát hiện GPU, cài torch phù hợp, chạy smoke test
#   ./setup.sh --cpu        # ép cài bản CPU
#   ./setup.sh --no-smoke   # bỏ qua bước smoke test
#
# Không cần sudo: dùng uv để tải Python 3.12 standalone (Ubuntu 20.04 chỉ có
# sẵn Python 3.8, quá cũ cho torch mới). Chạy lại script bất cứ lúc nào cũng
# an toàn (idempotent).
set -euo pipefail
cd "$(dirname "$0")"

FORCE_CPU=0
RUN_SMOKE=1
for arg in "$@"; do
    case "$arg" in
        --cpu) FORCE_CPU=1 ;;
        --no-smoke) RUN_SMOKE=0 ;;
        *) echo "tham số lạ: $arg (hỗ trợ: --cpu, --no-smoke)"; exit 1 ;;
    esac
done

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- uv + python
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    say "Cài uv (trình quản lý Python/venv, không cần sudo)"
    command -v curl >/dev/null 2>&1 || {
        echo "Thiếu curl. Cài bằng: sudo apt-get update && sudo apt-get install -y curl"
        exit 1
    }
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
say "uv: $(uv --version)"

if [ ! -x .venv/bin/python ]; then
    say "Tạo venv với Python 3.12 (uv tự tải, không đụng Python hệ thống)"
    uv venv --python 3.12 .venv
fi
PY=.venv/bin/python
say "Python: $($PY --version)"

# ------------------------------------------------------------- chọn bản torch
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
MODE="CPU"
if [ "$FORCE_CPU" -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
        | head -1 | cut -d. -f1 || true)"
    if [ -n "${DRIVER:-}" ]; then
        if [ "$DRIVER" -ge 525 ]; then
            # Driver đủ mới cho CUDA 12 — bản torch mặc định trên PyPI đã kèm CUDA 12.
            TORCH_INDEX=""
            MODE="GPU (CUDA 12, driver $DRIVER)"
        else
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            MODE="GPU (CUDA 11.8, driver cũ $DRIVER)"
        fi
    fi
fi
say "Cài PyTorch bản: $MODE"
if [ -n "$TORCH_INDEX" ]; then
    uv pip install --python "$PY" --index-url "$TORCH_INDEX" torch
else
    uv pip install --python "$PY" torch
fi
uv pip install --python "$PY" numpy pillow

# ------------------------------------------------------------------ kiểm tra
say "Kiểm tra import"
"$PY" - <<'EOF'
import torch, numpy, PIL
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

if [ "$RUN_SMOKE" -eq 1 ]; then
    say "Smoke test: dataset giả + train 2 epoch + evaluate"
    SMOKE_DIR=".smoke-test"
    rm -rf "$SMOKE_DIR"
    "$PY" scripts/make_fake_dataset.py --root "$SMOKE_DIR/data" >/dev/null
    "$PY" train.py --data-root "$SMOKE_DIR/data" --smoke-test --image-size 64 \
        --n-way 3 --k-shot 2 --n-query 3 --out "$SMOKE_DIR/run"
    "$PY" evaluate.py --checkpoint "$SMOKE_DIR/run/best.pt" \
        --data-root "$SMOKE_DIR/data" --split test \
        --n-way 3 --k-shot 2 --n-query 3 --episodes 10
    rm -rf "$SMOKE_DIR"
    say "Smoke test OK — môi trường sẵn sàng."
fi

cat <<'EOF'

Cài đặt xong. Bước tiếp theo:
  1. Chuẩn bị dữ liệu: data/{train,val,test}/<tên_lớp>/*.png
     (binary -> ảnh: .venv/bin/python scripts/binaries_to_images.py \
          --in-root raw/train --out-root data/train)
  2. Train:    .venv/bin/python train.py --data-root data --out runs/5w5s
  3. Đánh giá: .venv/bin/python evaluate.py --checkpoint runs/5w5s/best.pt \
                   --data-root data --split test --episodes 1000
EOF
