# A2-CLM — Reimplementation

Cài đặt lại mô hình trong bài báo:

> Liu et al., **"A2-CLM: Few-Shot Malware Detection Based on Adversarial
> Heterogeneous Graph Augmentation"**, IEEE TIFS, vol. 19, 2024.
> DOI: 10.1109/TIFS.2023.3345640

Chỉ phụ thuộc **PyTorch thuần** (không cần PyG/DGL) nên dễ mang sang máy khác.

## Ánh xạ code ↔ bài báo

| Thành phần trong bài báo | Code |
|---|---|
| Schema 7 entity / 8 relation, 8 meta-graph (Fig. 3b, Fig. 4) | `a2clm/schema.py` |
| Sensitivity grading TF-DF (Eq. 1) + GSDMM (Sec. III-B1) | `a2clm/sensitivity/` |
| Xây SHGFM G_o = (A_o, S_o) (Sec. III-B2, Def. 2) | `a2clm/graph.py` |
| PGD attack (Eq. 2–4) | `a2clm/augment/pgd.py` |
| Attribute masking S'=S∘(1−L)+V∘L (Eq. 5) | `a2clm/augment/masking.py` |
| Meta-graph-guide sampling (Eq. 6) | `a2clm/augment/metagraph_sampling.py` |
| Direct system calls attack (Sec. III-C4) | `a2clm/augment/dsc.py` |
| Obfuscation attack (Sec. III-C5, Eq. 7) | `a2clm/augment/obfuscation.py` |
| GAT attention (Eq. 8) + GIN aggregate (Eq. 9) + concat K lớp (Eq. 10) | `a2clm/model/encoder.py` |
| Attention theo meta-graph θ_i, h_G (Eq. 11–12), MLP head (Eq. 13) | `a2clm/model/encoder.py` |
| InfoNCE (Eq. 14–15) | `a2clm/loss.py` |
| 3 encoder GAT_o/GAT_p/GAT_q + momentum MoCo (Alg. 1, dòng 19–20) | `a2clm/model/a2clm.py` |
| Vòng huấn luyện (Algorithm 1) | `a2clm/train.py` |
| Đánh giá few-shot c-way n-shot, split 6:2:2 theo family (Sec. IV-B) | `a2clm/evaluate.py`, `a2clm/data/fewshot.py` |

Hyperparameters mặc định lấy đúng theo Sec. IV-B: Adam lr=0.005, weight decay
1e-5, batch 64, τ=0.07, embedding 128, K=4 lớp GAT, momentum λ₁=λ₂=0.99,
masking ratio 30%, 2 PGD attack.

## Cài đặt

Máy chạy (Linux, có GPU càng tốt):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # hoặc: pip install -e .
```

Nếu cần CUDA, cài torch theo hướng dẫn https://pytorch.org trước
(ví dụ `pip install torch --index-url https://download.pytorch.org/whl/cu124`),
rồi mới `pip install -r requirements.txt`.

Code tự chọn GPU nếu có (`--device auto`), không có thì chạy CPU.

## Smoke test (không cần dữ liệu thật)

```bash
bash scripts/smoke_test.sh
```

Sinh dữ liệu tổng hợp → fit sensitivity grading → build graph → train
contrastive 3 epoch → đánh giá 1-shot/3-shot. Chạy CPU khoảng 1–2 phút.

## Dữ liệu thật

Giữ nguyên data contract của bản build trước:

```
data/processed/metadata.csv   # sample_id,file_path,report_path,label,family,timestamp,source_dataset
data/reports/{sample_id}.json # {"sample_id":..., "events":[{src_type,src,relation,dst_type,dst,parameters}]}
```

- `label`: `1`/`malware` hoặc `0`/`benign` (bỏ trống thì suy từ `family`).
- Loại entity: `process, api, file, system, registry, memory, network`;
  quan hệ suy được từ cặp (src_type, dst_type) nên `relation` ghi tự do.
- Report XML kiểu KingKong (Fig. 3a): thêm `--format xml`.

## Train + đánh giá

```bash
python -m a2clm.train \
    --metadata data/processed/metadata.csv \
    --config configs/default.yaml \
    --out runs/exp1 --device auto

python -m a2clm.evaluate --run runs/exp1 \
    --metadata data/processed/metadata.csv
```

Kết quả lưu ở `runs/exp1/`: `checkpoint.pt`, `grader.json` (bảng sensitivity
đã fit), `splits.json`, `history.json`, `eval_test.json`.

Đánh giá: chia family 6:2:2 (train/val/test **không trùng family**), mỗi shot
chạy nhiều episode, phân lớp query theo cosine với prototype của support set;
báo cáo ACC/Precision/Recall/F1 (macro) và AUC nhị phân nếu có lớp benign —
đúng protocol Sec. IV-B/C.

## Mang sang máy khác (Ubuntu 20.04 trở lên)

```bash
# trên máy này
scp a2clm.tar.gz user@may-khac:~/

# trên máy kia — MỘT LỆNH DUY NHẤT:
mkdir a2clm && tar xzf a2clm.tar.gz -C a2clm && cd a2clm
bash scripts/setup_ubuntu20.sh
```

`setup_ubuntu20.sh` tự động toàn bộ: tìm/cài Python ≥ 3.10 (Ubuntu 20 mặc
định chỉ có 3.8 — script dùng PPA deadsnakes nếu có sudo, không có sudo thì
dùng `uv` tải Python standalone), tạo venv, cài torch **CUDA nếu máy có GPU**
(tự phát hiện qua `nvidia-smi`, không có thì bản CPU), cài dependency, và
chạy smoke test cuối cùng để xác nhận môi trường sẵn sàng.

Tuỳ chọn: `--cpu` (ép dùng torch CPU), `--no-test` (bỏ qua smoke test).

## Ghi chú cài đặt (khác biệt so với bài báo)

- Meta-graph Fig. 4 chỉ được vẽ dưới dạng hình; ở đây mỗi M_i được mã hoá
  bằng tập quan hệ + đường kính pattern (`schema.py`), lân cận meta-graph
  lấy bằng BFS giới hạn theo quan hệ — xấp xỉ sát ngữ nghĩa Eq. 6.
- Attention giữa các meta-graph (Eq. 11) cài theo dạng semantic attention
  chuẩn HAN vì công thức trong bài không định nghĩa đầy đủ ký hiệu.
- Obfuscation attack thao tác trên graph (thay API độc hại bằng chuỗi API
  tương đương + chèn garbage node) thay vì sửa file PE thật; Eq. 7 (MCS)
  chỉ là mục tiêu thiết kế, không tính trực tiếp vì MCS là NP-hard.
- Negative sampling lấy các mẫu khác family trong batch (giả định chuẩn
  của MoCo/InfoNCE khi self-supervised).
