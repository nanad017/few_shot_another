#!/usr/bin/env bash
# Tự động build + kiểm chứng trên máy chạy thí nghiệm (Ubuntu 20.04+).
#
# Ubuntu 20.04 chỉ có Python 3.8, quá cũ so với yêu cầu (Python 3.12), nên
# script dùng uv để tự tải Python 3.12 standalone — không cần sudo, không
# ảnh hưởng Python hệ thống.
#
# Cách dùng (từ thư mục gốc của project):
#   bash setup_ubuntu.sh            # cài đặt + chạy test + demo nhanh
#   bash setup_ubuntu.sh --gpu     # cài torch bản CUDA thay vì CPU
#   bash setup_ubuntu.sh --no-demo # chỉ cài đặt + test, bỏ qua demo

set -euo pipefail
cd "$(dirname "$0")"

GPU=0
DEMO=1
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU=1 ;;
        --no-demo) DEMO=0 ;;
        *) echo "Tham số không hợp lệ: $arg"; exit 1 ;;
    esac
done

echo "=== [1/5] Kiểm tra uv ==="
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Chưa có uv, đang cài (không cần sudo)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi
uv --version

echo "=== [2/5] Tạo virtualenv Python 3.12 ==="
uv venv -p 3.12 .venv   # uv tự tải Python 3.12 nếu máy chưa có

echo "=== [3/5] Cài dependencies ==="
if [ "$GPU" -eq 1 ]; then
    uv pip install -p .venv torch
else
    # bản CPU nhẹ hơn nhiều (~200MB thay vì ~2.5GB); bài báo cũng chạy CPU
    uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install -p .venv -r requirements.txt

echo "=== [4/5] Kiểm chứng mạch lượng tử ==="
.venv/bin/python tests/test_quantum_circuit.py

if [ "$DEMO" -eq 1 ]; then
    echo "=== [5/5] Smoke-test toàn pipeline trên dữ liệu tổng hợp ==="
    .venv/bin/python scripts/run_demo.py --episodes 300 --quantum-epochs 5
else
    echo "=== [5/5] Bỏ qua demo (--no-demo) ==="
fi

cat <<'EOF'

============================================================
Build hoàn tất. Môi trường nằm trong .venv/

Chạy thí nghiệm đầy đủ với dataset thật:

  # CCCS-CIC-AndMal-2020 — 15 họ malware, 51 features
  .venv/bin/python scripts/run_experiment.py --csv andmal2020.csv \
      --label-col Class --n-features 51 --episodes 4000

  # KronoDroid real-device — nhị phân, 29 features, kèm drift evaluation
  .venv/bin/python scripts/run_experiment.py --csv kronodroid_real.csv \
      --label-col Malware --timestamp-col FirstSeen \
      --n-features 29 --episodes 4000

Dataset tải riêng (link trong README.md).
============================================================
EOF
