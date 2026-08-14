# Few-shot Android Malware Classification with Quantum-Enhanced Prototypical Learning and Drift Detection

Re-implementation của bài báo:

> Tawfik et al., *"Few-shot android malware classification with quantum-enhanced
> prototypical learning and drift detection"*, Scientific Reports 16:10744 (2026).
> https://doi.org/10.1038/s41598-026-45738-0

## Kiến trúc (5 module, theo Fig. 1)

| Module | File | Mô tả |
|---|---|---|
| A. Tiền xử lý | `src/qproto/preprocessing.py` | Zero-imputation + z-score (Eq. 1), thống kê fit **chỉ trên train** |
| B. Feature selection | `src/qproto/feature_selection.py` | CatBoost + permutation importance (Eq. 2, Algorithm 1); top-51 (CIC-AndMal-2020) / top-29 (KronoDroid) |
| C. Prototypical network | `src/qproto/protonet.py`, `episodes.py`, `train.py` | MLP `d→512→256→128` (BN, ReLU, Dropout 0.3; layer cuối tuyến tính). Episodic 5-way 5-shot, Q=10, 4000 episodes, Adam lr=1e-3, StepLR(1000, γ=0.5) (Algorithm 2, Eq. 3–5) |
| D. Quantum hybrid layer | `src/qproto/quantum.py`, `quantum_train.py` | Pre-quantum MLP `d→64→4` (tanh) → PQC 4 qubit: R_Y encoding → ring CNOT → R_Z(αz), α=0.5 → đo Pauli-Z → head phân loại, huấn luyện cross-entropy |
| E. Drift detection | `src/qproto/drift.py` | T=6 splits thời gian, train tích lũy 100 episodes, đánh giá 50 episodes, Δ(t)=(A(1)−A(t))/A(1)·100, cảnh báo khi Δ > 15% |

Toàn bộ hyperparameters (Table 3) nằm trong `src/qproto/config.py`, seed=42.
Nhánh quantum dùng Adam + StepLR (1000, γ=0.5) và nhận `n_qubits`, `alpha`
từ hàm train thay vì dùng giá trị ẩn trong model.

## Ghi chú so với bài báo

- **Mô phỏng lượng tử**: bài báo dùng Qiskit; ở đây mạch 4 qubit được mô phỏng
  statevector chính xác bằng PyTorch thuần (`quantum.py`) — tương đương toán học,
  gradient qua autograd trùng với parameter-shift, nhanh hơn và không cần thêm
  dependency. File test `tests/test_quantum_circuit.py` đối chiếu với tham chiếu
  ma trận dày.
- **Tham số mạch**: theo đúng mô tả trong bài báo, các góc quay R_Y(z_i) và
  R_Z(αz_i) đều là hàm của z — tham số học được nằm ở mạng classical trước/sau
  mạch lượng tử.
- **Fusion**: bài báo mô tả quantum layer là "alternative classification
  pathway" nhưng không nêu rõ cách hợp nhất với protonet; ở đây hai nhánh được
  huấn luyện/đánh giá độc lập.
- Feature selection mặc định dùng `LossFunctionChange` của CatBoost (nhanh cho
  d=9,503); chế độ `method="permutation"` cài đúng Eq. 2 từng chữ.

## Cài đặt & chạy (trên máy chạy thí nghiệm)

**Cách tự động (Ubuntu 20.04+):** copy toàn bộ thư mục project sang máy chạy rồi:

```bash
bash setup_ubuntu.sh            # cài Python 3.12 + deps, chạy test + demo nhanh
bash setup_ubuntu.sh --gpu      # dùng torch bản CUDA
bash setup_ubuntu.sh --no-demo  # chỉ cài + test, bỏ qua demo
```

Script dùng `uv` tự tải Python 3.12 standalone (Ubuntu 20.04 chỉ có 3.8),
không cần sudo, không đụng Python hệ thống.

**Cách thủ công** (nếu máy đã có Python 3.12):

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.12
pip install -r requirements.txt
```

Kiểm tra mạch lượng tử:

```bash
python tests/test_quantum_circuit.py
```

Demo end-to-end trên dữ liệu tổng hợp (không cần dataset):

```bash
python scripts/run_demo.py --episodes 1000
```

Tái lập thí nghiệm trên dữ liệu thật (tải riêng, không kèm repo):

- CCCS-CIC-AndMal-2020: https://www.unb.ca/cic/datasets/andmal2020.html
- KronoDroid (real-device subset): https://github.com/aleguma/kronodroid

```bash
# CCCS-CIC-AndMal-2020 — phân loại 15 họ malware, 51 features
python scripts/run_experiment.py --csv andmal2020.csv --label-col Class \
    --n-features 51 --episodes 4000

# KronoDroid — phân loại nhị phân, 29 features, kèm drift evaluation
python scripts/run_experiment.py --csv kronodroid_real.csv --label-col Malware \
    --timestamp-col FirstSeen --n-features 29 --episodes 4000
```

Kết quả bài báo để đối chiếu: 99.70% accuracy trên CCCS-CIC-AndMal-2020
(15 họ, 51 features), 99.33% trên KronoDroid (nhị phân, 29 features),
suy giảm tối đa 0.24% qua các temporal splits.
