# DPNSA — Dynamic Prototype Network Based on Sample Adaptation

Cài đặt lại PyTorch của bài báo:

> Chai et al., *Dynamic Prototype Network Based on Sample Adaptation for
> Few-Shot Malware Detection*, IEEE TKDE 35(5), 2023.
> DOI: 10.1109/TKDE.2022.3142820

## Kiến trúc (theo bài báo)

1. **Malware image module** — binary → ảnh grayscale 256×256
   (`scripts/binaries_to_images.py`, phương pháp Nataraj et al.).
2. **Dynamic feature embedding** (Fig. 2) — 3 block
   `conv 3×3 → BatchNorm → ReLU → maxpool 2×2` với số filter 32 → 64 → 128;
   block giữa dùng **dynamic convolution** (attention trên K=4 kernel,
   squeeze-and-excitation, softmax temperature anneal 30 → 1).
3. **Prototype** — trung bình embedding các mẫu support của mỗi lớp.
4. **Dual-sample dynamic activation** (Fig. 3–4, Eq. 1–2) — hyper function
   α(Q′, Pₙ) sinh tham số (aᶜᶠ, bᶜᶠ) theo từng channel;
   kích hoạt `max_f(a·x + b)` cho cả query lẫn prototype, rồi conv + maxpool.
   DS1 chuẩn hóa vector tham số sinh ra trước khi áp dụng, đúng bước
   normalization trong Fig. 4a.
   Hai biến thể: `ds1` (mặc định, tốt hơn theo Table 4) và `ds2`
   (thêm spatial attention).
5. **Metric** — cosine distance, softmax(−d) (Eq. 3), NLL loss (Eq. 4).

Siêu tham số mặc định theo Section 4.1: Adam lr 1e-3, 1000 epoch ×
100 episode, step LR decay mỗi 20 epoch, patience 200, augment lật ngang,
đánh giá 1000 episode (accuracy ± 95% CI).

## Cài đặt (trên máy chạy train — Ubuntu 20.04+)

Copy nguyên thư mục `DPNSA/` sang máy đích (không cần copy `.venv/`), rồi:

```bash
cd DPNSA
./setup.sh          # tự phát hiện GPU, cài torch phù hợp, chạy smoke test
```

Script không cần sudo (dùng `uv` tải Python 3.12 standalone — Ubuntu 20.04
chỉ có sẵn Python 3.8, quá cũ cho torch mới), tự chọn bản torch theo driver
NVIDIA (driver ≥ 525 → CUDA 12; cũ hơn → CUDA 11.8; không GPU → CPU),
và chạy lại lúc nào cũng an toàn. Tuỳ chọn: `--cpu` (ép bản CPU),
`--no-smoke` (bỏ smoke test). Yêu cầu duy nhất: có `curl` và có mạng.

Cài tay (nếu không muốn dùng script): tạo venv Python ≥ 3.10 rồi
`pip install -r requirements.txt` (bản CUDA cụ thể: xem
https://pytorch.org/get-started/locally/).

## Kiểm tra nhanh (không cần dữ liệu thật)

```bash
python scripts/make_fake_dataset.py --root data-fake
python train.py --data-root data-fake --smoke-test --image-size 64 --n-way 3 --k-shot 2 --n-query 3
```

## Chuẩn bị dữ liệu thật

Bài báo dùng **Filtered LargePE** (Tang et al., từ VirusTotal): 100 lớp
train / 58 val / 50 test, 20 mẫu/lớp, ảnh 256×256.

- Nếu đã có ảnh: xếp theo `data/{train,val,test}/<tên_lớp>/*.png`.
- Nếu chỉ có binary: xếp binary theo `raw/{train,val,test}/<tên_lớp>/…` rồi

```bash
python scripts/binaries_to_images.py --in-root raw/train --out-root data/train
# lặp lại cho val, test
```

Lưu ý: chia lớp (không phải chia mẫu) giữa train/val/test — label space
của 3 split phải rời nhau.

## Train

```bash
# 5-way 5-shot (mặc định)
python train.py --data-root data --out runs/5w5s

# 5-way 1-shot
python train.py --data-root data --k-shot 1 --out runs/5w1s

# 10-way 10-shot (bài báo dùng 10 query/lớp cho 10-shot)
python train.py --data-root data --n-way 10 --k-shot 10 --n-query 10 --out runs/10w10s

# Biến thể DS2
python train.py --data-root data --variant ds2 --out runs/5w5s-ds2
```

Thiết bị tự chọn (`cuda` → `mps` → `cpu`); ép bằng `--device cuda:0`.
Checkpoint tốt nhất theo val: `runs/<tên>/best.pt`; log: `history.json`.

## Đánh giá

```bash
python evaluate.py --checkpoint runs/5w5s/best.pt --data-root data \
    --split test --n-way 5 --k-shot 5 --episodes 1000
```

Bài báo train 10-way rồi đánh giá cả 5-way lẫn 10-way — checkpoint dùng
được cho mọi N-way/K-shot vì model không phụ thuộc số lớp.

## Cấu trúc code

```
dpnsa/modules.py     # DynamicConv2d (attention over kernels)
dpnsa/embedding.py   # embedding 3 block (Fig. 2)
dpnsa/activation.py  # dual-sample dynamic activation DS1/DS2 (Fig. 3-4)
dpnsa/model.py       # DPNSA: prototype + activation + cosine metric
dpnsa/data.py        # dataset ảnh malware + episodic sampler
train.py             # vòng train episodic theo Section 4.1
evaluate.py          # accuracy ± 95% CI trên N episode test
scripts/             # binary→ảnh, dataset giả để smoke-test
setup.sh             # cài đặt tự động trên máy mới (Ubuntu 20.04+)
```

Đóng gói mang sang máy khác:

```bash
tar --exclude=.venv --exclude=__pycache__ --exclude=runs -czf dpnsa.tar.gz DPNSA/
# bên máy đích: tar xzf dpnsa.tar.gz && cd DPNSA && ./setup.sh
```
