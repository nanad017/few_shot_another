# Hướng dẫn triển khai cho AI trên máy khác

## 1. Mục tiêu

Tài liệu này dành cho AI/coding agent được chạy trên một máy khác, nơi người dùng sẽ cung cấp source code, dataset động và có thể cung cấp thêm raw PE.

Mục tiêu là:

1. Cài và kiểm tra độc lập ba model:
   - `CLM_sau`: A2-CLM.
   - `Dynamic/DPNSA`: Dynamic Prototype Network Based on Sample Adaptation.
   - `Quantum`: Quantum-enhanced Prototypical Learning.
2. Dùng đúng dataset do người dùng cung cấp.
3. Không random split lại dữ liệu.
4. Không dùng leakage feature.
5. Ghi rõ model nào chạy được exact theo bài báo, model nào chỉ là adaptation do khác domain/input.

AI không được tự ý đổi định nghĩa label, family hoặc split để làm cho quá trình train dễ hơn.

## 2. Quy tắc làm việc bắt buộc

Trước khi sửa code hoặc train thật, AI phải:

1. Kiểm tra `git status` và giữ nguyên mọi thay đổi có sẵn của người dùng.
2. Đọc các README trong `CLM_sau/`, `Dynamic/DPNSA/`, `Quantum/`.
3. Xác nhận phiên bản Python, PyTorch, CUDA và dung lượng disk.
4. Liệt kê dataset thực tế được cung cấp, không đoán tên file.
5. Kiểm tra schema parquet/CSV bằng công cụ đọc dữ liệu; không giả định tên cột event nếu chưa kiểm tra.
6. Tạo một báo cáo data audit trước khi train.
7. Chỉ dùng `train` để fit preprocessing, feature selection, scaler, sensitivity grader hoặc model.
8. Dùng `test` đúng như manifest. Không gọi hàm random split mặc định cho dữ liệu thật.
9. Chạy smoke test nhỏ trước khi chạy full training.
10. Lưu command, config, số lượng mẫu, hash source và kết quả vào thư mục run.

## 3. Cấu trúc thư mục đề nghị trên máy mới

Không hard-code đường dẫn `/home/nad`. Có thể tổ chức như sau:

```text
project_root/
  CLM_sau/
  Dynamic/DPNSA/
  Quantum/
  data/
    dataset_full_x86_x64_20260813_063509Z/
      train_test_view/
      aggregate_features/
      sequence_features/
      event_tables/
    prepared_views/
      static_dynamic_joined_index.csv
      family_labels_from_raw_pe_tree.csv
      family_counts.csv
    raw_pe/
    derived/
      a2clm_reports/
      dpnsa_images/
      static_features/
      joined_features/
  runs/
  reports/
```

Dataset có thể nằm ở vị trí khác. AI phải nhận path từ người dùng hoặc tự tìm bằng `find`, sau đó ghi lại path thực tế trong báo cáo.

## 4. Manifest chuẩn của dataset

File quan trọng nhất là:

```text
prepared_views/static_dynamic_joined_index.csv
```

Manifest này là source of truth để map mẫu:

```text
sha256, split, label, family, arch_group, analysis_id,
behavior_present, api_call_count, ...
```

Dataset family classification có đúng 5 family malware:

```text
Locker
Mediyes
Winwebsec
Zbot
Zeroaccess
```

Manifest hiện có cả benign. AI phải xác định rõ experiment đang chạy là:

- `family classification`: target là 5 family malware; benign phải được loại ra hoặc định nghĩa thành một lớp riêng và ghi rõ.
- `malware detection`: target là `label`, thường gồm benign/malware.

Không được trộn hai bài toán trong cùng một metric.

## 5. Quy tắc leakage

Các cột sau tuyệt đối không được đưa vào feature matrix:

```text
label
label_id
split
source_group
original_filename
original_relative_path
analysis_id
family
sha256
```

Quy tắc sử dụng:

- `family`: chỉ dùng làm target của family classification.
- `label`: chỉ dùng làm target của malware detection.
- `sha256`: chỉ dùng để join/index/audit.
- `analysis_id`: chỉ dùng để join event tables và manifest.
- `split`: chỉ dùng để chọn train/test, không dùng làm feature.
- `arch_group`: chỉ dùng để phân tích hoặc stratification theo yêu cầu; không dùng làm feature nếu chưa có lý do rõ ràng.

Nếu cần dùng một cột thống kê như `api_call_count`, AI phải kiểm tra cột đó có được tạo từ dữ liệu test hoặc từ toàn bộ dataset bằng một phép fit hay không. Nếu có, phải xây lại thống kê chỉ trên train.

## 6. Audit dataset trước khi chạy

AI phải xuất một file, ví dụ:

```text
reports/data_audit.json
```

Audit tối thiểu phải có:

```text
number of rows
number of unique sha256
number of unique analysis_id
train/test counts
label counts by split
family counts by split
missing sha256
duplicate sha256
duplicate analysis_id
raw PE matched/unmatched counts
event rows by table
```

Phải kiểm tra các điều kiện:

```text
sha256 không trùng sai giữa train và test
analysis_id join được với event tables
family tồn tại ở cả split cần đánh giá
không có sample bị gán hai family
raw PE map được về manifest nếu chạy DPNSA/static
```

Nếu phát hiện duplicate hoặc mapping không đầy đủ, AI phải dừng full training và báo rõ tỷ lệ lỗi.

## 7. Cài môi trường

Mỗi model nên dùng virtual environment riêng để tránh xung đột dependency.

### A2-CLM

```bash
cd CLM_sau
bash scripts/setup_ubuntu20.sh --cpu
```

Nếu máy có GPU và muốn dùng CUDA, bỏ `--cpu` sau khi đã xác nhận `nvidia-smi` hoạt động.

Smoke test thủ công:

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_synthetic.py \
  --out /tmp/a2clm_smoke --per-family 3 --benign 4

PYTHONPATH=. .venv/bin/python -m a2clm.train \
  --metadata /tmp/a2clm_smoke/processed/metadata.csv \
  --config configs/smoke.yaml \
  --out /tmp/a2clm_smoke_run \
  --epochs 1 --device cpu
```

### DPNSA

```bash
cd Dynamic/DPNSA
bash setup.sh --cpu
```

### Quantum

```bash
cd Quantum
bash setup_ubuntu.sh --no-demo
```

Nếu setup tự động không phù hợp với máy, tạo Python 3.10+ venv rồi cài `requirements.txt`, sau đó xác nhận các import `torch`, `numpy`, `pandas`, `sklearn`, `catboost`, `PIL`.

## 8. Chuẩn bị data động

### 8.1 Dynamic aggregate

Có thể dùng:

```text
dataset_full_x86_x64_20260813_063509Z/train_test_view/aggregate_features/dynamic_features_with_split.csv
```

File này phải join với:

```text
prepared_views/static_dynamic_joined_index.csv
```

Join ưu tiên theo `sha256`; nếu chỉ có `analysis_id`, phải kiểm tra đó là mapping một-một và ghi lại trong audit.

Sau khi join, tạo các bảng:

```text
derived/dynamic_train.csv
derived/dynamic_test.csv
```

Các bảng này phải giữ split gốc. Không gọi `train_test_split`, `stratified_split` hoặc hàm random split tương tự cho dữ liệu thật.

### 8.2 Event-level data cho A2-CLM

A2-CLM không dùng aggregate count đơn thuần. Model cần graph từ event-level data. AI phải đọc và kiểm tra các bảng:

```text
event_tables/api_events.parquet
event_tables/processes.parquet
event_tables/process_edges.parquet
event_tables/file_events.parquet
event_tables/registry_events.parquet
event_tables/network_flows.parquet
event_tables/dns_events.parquet
event_tables/artifacts.parquet
```

Tạo một report chuẩn cho mỗi sample, ví dụ:

```json
{
  "sample_id": "<sha256 hoặc analysis_id>",
  "events": [
    {
      "src_type": "process",
      "src": "process-1",
      "relation": "call",
      "dst_type": "api",
      "dst": "NtWriteFile",
      "parameters": ["...", "..."]
    }
  ]
}
```

Các loại node được model hỗ trợ:

```text
process, api, file, system, registry, memory, network
```

Các quan hệ chính:

```text
process-fork-process
process-call-api
process-access-file
process-open-system
process-connect-network
process-read-memory
process-set-registry
network-download-file
```

AI phải giữ parameters/path/URL/registry key nếu event table có các trường này, vì A2-CLM dùng chúng cho sensitivity grading. Nếu bảng không có loại entity hoặc relation tương ứng, phải ghi rõ phần bị mất và không tuyên bố exact reproduction.

Tạo metadata cho A2-CLM, tối thiểu:

```text
sample_id,report_path,label,family,split
```

Ví dụ output:

```text
derived/a2clm_metadata.csv
derived/a2clm_reports/<sample_id>.json
```

Model A2-CLM không cần static PE nếu event graph đã đủ.

## 9. Dùng raw PE cho DPNSA

### 9.1 Điều kiện quan trọng

DPNSA theo bài báo dùng:

```text
raw PE bytes -> grayscale malware image 256x256
```

Static feature CSV không thay thế được input này. Vì vậy raw PE mà người dùng cung cấp là phù hợp.

Raw PE phải map được về `sha256` trong manifest. Nếu tên file không phải sha256, AI phải tạo mapping riêng và kiểm tra hash bằng cách hash lại file.

### 9.2 Chuẩn bị thư mục raw

Nếu có thể, tổ chức raw PE theo split và family:

```text
derived/raw_pe/
  train/<family>/*.exe
  test/<family>/*.exe
```

DPNSA paper có meta-validation. Nếu dataset chỉ có train/test, AI không được tự chia random để giả lập val. Có thể dùng một phần train theo policy được người dùng chấp thuận, nhưng phải ghi rõ đó là adaptation.

### 9.3 Chuyển PE thành ảnh

Script có sẵn:

```bash
cd Dynamic/DPNSA

.venv/bin/python scripts/binaries_to_images.py \
  --in-root /path/to/raw_pe/train \
  --out-root /path/to/derived/dpnsa_images/train \
  --size 256

.venv/bin/python scripts/binaries_to_images.py \
  --in-root /path/to/raw_pe/test \
  --out-root /path/to/derived/dpnsa_images/test \
  --size 256
```

Layout cuối:

```text
derived/dpnsa_images/
  train/<class>/*.png
  val/<class>/*.png       # chỉ tạo nếu có policy val hợp lệ
  test/<class>/*.png
```

Không dùng `original_filename` làm feature. Tên file PNG chỉ phục vụ mapping/index.

### 9.4 Chạy DPNSA

```bash
.venv/bin/python train.py \
  --data-root /path/to/derived/dpnsa_images \
  --n-way 5 \
  --k-shot 5 \
  --n-query 15 \
  --image-size 256 \
  --epochs 1000 \
  --episodes-per-epoch 100 \
  --out /path/to/runs/dpnsa_5w5s
```

Nếu chỉ chạy 5 family và mỗi family không đủ số lượng cho episode, AI phải giảm `n-way`, `k-shot`, `n-query` theo số liệu thực tế và ghi rõ thay đổi.

## 10. Chuẩn bị static features từ raw PE

Raw PE có thể dùng để trích xuất static features, nhưng cần phân biệt hai mục đích:

- Với DPNSA: dùng bytes để tạo ảnh; static feature CSV không thay thế ảnh.
- Với Quantum/ProtoNet: static features có thể join với dynamic aggregate để tạo vector đầu vào.
- Với A2-CLM: static features không bắt buộc, vì model chính là event graph.

Nếu repo không có static extractor, AI không được tự tạo một extractor tùy ý rồi gọi đó là exact theo bài báo. AI phải:

1. Kiểm tra có tool/extractor nội bộ nào được cung cấp không.
2. Ghi tên tool, phiên bản, danh sách feature và cách xử lý lỗi.
3. Hash lại raw PE trước khi trích xuất.
4. Map output về `sha256`.
5. Không dùng label/family/split trong static feature vector.
6. Nếu extractor có bước fit hoặc chọn feature, chỉ fit trên train.
7. Lưu schema static features và log số file thành công/thất bại.

Output đề nghị:

```text
derived/static_features/static_features.csv
derived/static_features/extraction_manifest.csv
derived/static_features/extraction_errors.csv
```

Với Windows PE, feature vector static/dynamic sẽ khác bộ Android trong Quantum paper. Vì vậy kết quả phải được gọi là adaptation trên Windows PE, không được báo là tái lập đúng dataset Android của bài báo.

## 11. Quantum/ProtoNet trên data của người dùng

Pipeline đề nghị:

```text
manifest + dynamic aggregate + static features (nếu có)
  -> join theo sha256
  -> chọn train/test theo manifest
  -> loại leakage columns
  -> zero-imputation và z-score fit trên train
  -> CatBoost feature selection fit trên train/validation hợp lệ
  -> episodic ProtoNet
```

Các giá trị `51` và `29` trong bài báo chỉ tương ứng với hai dataset Android gốc. Không được mặc định tuyên bố rằng `top 51` hoặc `top 29` là tối ưu cho Windows PE. AI phải ghi rõ:

- số feature đầu vào ban đầu;
- số feature sau selection;
- phương pháp selection;
- split dùng để fit selection;
- danh sách feature được chọn.

### 11.1 Bắt buộc sửa split trước khi chạy thật

Code hiện có hàm random split trong:

```text
Quantum/src/qproto/data.py
Quantum/scripts/run_experiment.py
```

Đối với dataset này, AI phải thêm loader nhận `split` từ manifest hoặc tạo CSV đã có split và dùng trực tiếp:

```text
X_train, y_train = rows[split == "train"]
X_test, y_test   = rows[split == "test"]
```

Không gọi `stratified_split()` cho experiment chính. Nếu cần validation, phải dùng policy được ghi rõ, ví dụ một phần train cố định theo hash hoặc một file validation do người dùng cung cấp. Không được lấy test để chọn feature hoặc chọn checkpoint.

### 11.2 Cấu hình episode

Thông số theo paper:

```text
n_way = 5
k_shot = 5
q_query = 10
embedding = d -> 512 -> 256 -> 128
dropout = 0.3
episodes = 4000
Adam lr = 1e-3
StepLR step = 1000, gamma = 0.5
```

Nếu một family không đủ `k_shot + q_query` mẫu trong train, AI phải báo lỗi hoặc giảm cấu hình có giải thích; không sample lặp âm thầm cho kết quả chính.

## 12. A2-CLM protocol

A2-CLM paper dùng family-disjoint split, nhưng dataset hiện tại đã có split train/test cố định và nhiều family có mặt ở cả hai split. Với dataset người dùng:

1. Dùng đúng `split=train/test` của manifest.
2. Không gọi `split_families()` hiện có cho experiment chính.
3. Nếu đánh giá family classification, target là `family` với 5 family đã nêu.
4. Nếu đánh giá malware detection, target là `label`.
5. Sensitivity grader chỉ fit trên train.
6. Báo cáo rõ đây là fixed-sample split, không phải family-held-out protocol của paper.

Các điểm kiến trúc đã được chỉnh trong code:

```text
CLM_sau/a2clm/schema.py
CLM_sau/a2clm/graph.py
```

Meta-graph được biểu diễn bằng path có hướng; M5/M6 yêu cầu resource dùng chung bởi hai process.

## 13. Báo cáo kết quả bắt buộc

Mỗi model phải lưu một thư mục run có:

```text
run/
  command.txt
  config.yaml hoặc config.json
  data_audit.json
  feature_schema.json
  selected_features.txt       # nếu có feature selection
  metrics.json
  predictions.csv
  history.json
  checkpoint.pt
  environment.txt
```

`predictions.csv` tối thiểu có:

```text
sha256,analysis_id,split,y_true,y_pred,model
```

Không ghi raw PE vào checkpoint hoặc log nếu không cần thiết.

Metrics cần báo:

- accuracy;
- macro precision/recall/F1;
- per-family metrics;
- confusion matrix;
- số support/query mỗi episode;
- confidence interval nếu đánh giá episodic;
- train/test sample counts;
- unmatched/filtered sample counts.

## 14. Tiêu chí nghiệm thu

AI chỉ được kết luận hoàn tất khi:

1. Ba môi trường import và smoke test thành công, hoặc nêu rõ model nào bị block bởi dependency.
2. Manifest audit không có mapping nghiêm trọng chưa giải thích.
3. Train/test counts đúng manifest.
4. Không có leakage columns trong feature matrix.
5. Không có random split trong experiment chính.
6. A2-CLM có event-to-graph report và report số event bị bỏ qua.
7. DPNSA có ảnh 256×256 map được về raw PE/sha256.
8. Quantum có schema feature và danh sách feature được chọn.
9. Kết quả phân biệt rõ `paper reproduction` và `Windows PE adaptation`.
10. AI cung cấp command tái chạy và đường dẫn output.

## 15. Những việc AI không được làm

- Không random split lại dataset.
- Không dùng `family`, `label`, `sha256`, `analysis_id` làm feature.
- Không lấy test để fit scaler, CatBoost, sensitivity grader hoặc chọn checkpoint.
- Không đổi family name để sửa lỗi parser.
- Không bỏ benign mà không ghi trong protocol.
- Không biến dynamic aggregate thành A2 graph nếu chưa có event-level mapping.
- Không gọi static feature CSV là input exact của DPNSA.
- Không gọi kết quả Windows PE là kết quả tái lập Android paper.
- Không xóa hoặc revert thay đổi có sẵn trong workspace.

## 16. Thứ tự chạy đề nghị

```text
1. Audit manifest và raw PE mapping
2. Smoke test ba model
3. Chuẩn bị dynamic aggregate và fixed train/test views
4. Chuẩn bị event reports cho A2-CLM
5. Chuẩn bị PE images cho DPNSA
6. Trích xuất static features nếu cần cho Quantum
7. Chạy experiment nhỏ để kiểm tra shape/memory/leakage
8. Chạy full training
9. Đánh giá test một lần sau khi khóa config
10. Xuất report và command tái chạy
```

Khi thiếu dữ liệu hoặc schema không đủ, AI phải dừng ở bước tương ứng và báo chính xác file, cột, số dòng hoặc mapping đang thiếu; không tự suy đoán dữ liệu còn thiếu.
