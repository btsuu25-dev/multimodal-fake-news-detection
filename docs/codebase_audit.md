# Codebase Audit – Multimodal Fake News Detection

> **Phiên bản:** 1.0 &nbsp;|&nbsp; **Ngày cập nhật:** 13/09/2026  
> **Trạng thái dự án:** ✅ Hoàn thành toàn bộ (M1–M6)

---

## Mục lục

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Kiến trúc mô hình](#3-kiến-trúc-mô-hình)
4. [Chi tiết từng file](#4-chi-tiết-từng-file)
5. [Luồng dữ liệu (Data Flow)](#5-luồng-dữ-liệu-data-flow)
6. [Kết quả thực nghiệm](#6-kết-quả-thực-nghiệm)
7. [Hướng dẫn chạy lại](#7-hướng-dẫn-chạy-lại)
8. [Các vấn đề đã xử lý](#8-các-vấn-đề-đã-xử-lý)

---

## 1. Tổng quan dự án

| Mục | Thông tin |
|---|---|
| **Bài toán** | Phát hiện tin giả đa phương thức (Multimodal Fake News Detection) |
| **Dataset** | Image Verification Corpus – MediaEval 2015 (11.524 mẫu hợp lệ, 360 ảnh) |
| **Kiến trúc chính** | CrossModalFND = CLIP Encoder + Cross-Attention Fusion + MLP Classifier |
| **Framework** | PyTorch 2.x, Transformers (HuggingFace), scikit-learn |
| **Kết quả tốt nhất** | Test Accuracy **88.16%** · F1 **0.8809** · AUC-ROC **0.9357** |

---

## 2. Cấu trúc thư mục

```
Course_work/
│
├── data/                          # Dữ liệu (bị .gitignore)
│   ├── raw/                       # Dữ liệu gốc tải về
│   │   └── image-verification-corpus/mediaeval2015/
│   ├── processed/
│   │   └── dataset.csv            # CSV đã ghép ảnh+text+nhãn
│   └── splits/
│       ├── train.csv              # 70% ảnh (~9.700 tweets)
│       ├── val.csv                # 15% ảnh (~870 tweets)
│       └── test.csv               # 15% ảnh (~950 tweets)
│
├── src/
│   ├── data/
│   │   ├── dataset.py             # M1 – PyTorch Dataset
│   │   ├── preprocessing.py       # M1 – Tiền xử lý dữ liệu
│   │   └── download_dataset.py    # M1 – Script tải dữ liệu
│   │
│   ├── models/
│   │   ├── image_encoder.py       # M2 – CLIPImageEncoder
│   │   ├── text_encoder.py        # M3 – CLIPTextEncoder
│   │   ├── baselines.py           # M4 – Text/Image/Concat Classifier
│   │   ├── fusion.py              # M5 – CrossAttentionFusion
│   │   └── crossmodal.py          # M5 – CrossModalFND (mô hình chính)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # M6 – Tính Accuracy, F1, AUC-ROC
│   │   ├── visualize.py           # M6 – Vẽ 4 biểu đồ PNG
│   │   └── html_report.py         # M6 – Tạo HTML Dashboard
│   │
│   ├── train.py                   # M5 – Script train CrossModalFND
│   ├── train_baseline.py          # M4 – Script train Baseline models
│   └── evaluate.py                # M6 – Script đánh giá (model-level + full pipeline)
│
├── docs/
│   ├── contracts.md               # Ký kết interface giữa các thành viên
│   ├── plan.md                    # Kế hoạch dự án
│   ├── requirements.md            # Yêu cầu kỹ thuật
│   ├── status.md                  # Trạng thái tiến độ
│   ├── team_assignment.md         # Phân công nhiệm vụ
│   └── codebase_audit.md          # (file này) – Tổng quan codebase
│
├── results/                       # Kết quả (bị .gitignore trừ reports/)
│   ├── training_log.csv           # Log từng epoch của M5
│   ├── plots/                     # 4 file PNG biểu đồ
│   ├── reports/
│   │   ├── baseline_comparison.md # Kết quả M4
│   │   └── final_evaluation.md    # Báo cáo đánh giá M6
│   └── dashboard.html             # HTML Dashboard tương tác
│
├── saved_models/                  # Checkpoints (bị .gitignore)
│   ├── best_model.pt              # CrossModalFND tốt nhất (~589MB)
│   ├── baseline_text.pt           # Text-Only checkpoint
│   ├── baseline_image.pt          # Image-Only checkpoint
│   └── baseline_concat.pt         # Concat checkpoint
│
├── prepare_data.py                # Script chuẩn bị dữ liệu đầu cuối
├── requirements.txt               # Danh sách thư viện
├── .gitignore                     # Loại trừ data, model, kết quả
└── README.md                      # Hướng dẫn sử dụng
```

---

## 3. Kiến trúc mô hình

### 3.1 CrossModalFND (Mô hình chính – M5)

```
Input: (image: PIL.Image, text: str)
         │                    │
         ▼                    ▼
 CLIPImageEncoder      CLIPTextEncoder
 (ViT-B/32, Frozen)   (ViT-B/32, Frozen)
         │                    │
         └────────┬───────────┘
                  ▼
        CrossAttentionFusion
        ┌─────────────────────────┐
        │ Q = image_emb [B, 512]  │
        │ K = text_emb  [B, 512]  │
        │ V = text_emb  [B, 512]  │
        │ MultiheadAttention(8)   │
        │ + LayerNorm + Residual  │
        └────────────┬────────────┘
                     │ [B, 1024]
                     ▼
              MLP Classifier
         Linear → ReLU → Dropout
         Linear → ReLU → Dropout
         Linear → Softmax
                     │
                     ▼
          Output: [REAL, FAKE]
```

**Tham số:**
- Tổng tham số: **154,431,234**
- Có thể học (trainable): **3,153,922** (2.04%)
- Frozen (CLIP): 151,277,312

### 3.2 Baseline Models (M4)

| Mô hình | Đầu vào | Kiến trúc |
|---|---|---|
| `TextOnlyClassifier` | `text_emb [B, 512]` | Linear(512→256) → ReLU → Linear(256→2) |
| `ImageOnlyClassifier` | `img_emb [B, 512]` | Linear(512→256) → ReLU → Linear(256→2) |
| `ConcatClassifier` | `[img_emb; txt_emb] [B, 1024]` | Linear(1024→512) → ReLU → Linear(512→2) |

---

## 4. Chi tiết từng file

### `src/data/dataset.py` — M1
**Người phụ trách:** Thành viên 1

**Lớp:** `FakeNewsDataset(Dataset)`

| Thành phần | Mô tả |
|---|---|
| `__init__(csv_path, transform)` | Đọc CSV, khởi tạo CLIP transform mặc định |
| `__getitem__(idx)` | Trả về `(image_tensor, text_str, label_int)` |
| `__len__()` | Số lượng mẫu |

**Contract:** Trả về tuple `(Tensor[3,224,224], str, int)` — M2/M3/M5 phụ thuộc vào interface này.

---

### `src/data/preprocessing.py` — M1
**Người phụ trách:** Thành viên 1

| Hàm | Chữ ký | Mô tả |
|---|---|---|
| `clean_text(text)` | `str → str` | Xóa URL, @mention, ký tự đặc biệt |
| `build_dataset_csv(tweets_path, images_dir, output_path)` | `str, str, str → int` | Ghép tweets.txt với ảnh, lưu CSV |
| `split_dataset(csv_path, output_dir, ...)` | `str, str → dict` | Chia theo tweet-level (đã deprecated, dùng `prepare_data.py`) |

---

### `src/models/image_encoder.py` — M2
**Người phụ trách:** Thành viên 2

**Lớp:** `CLIPImageEncoder(nn.Module)`

```python
# Input:  image_tensor  [B, 3, 224, 224]
# Output: image_embedding [B, 512]
```

- Model: `openai/clip-vit-base-patch32` (HuggingFace)
- Tất cả tham số **Frozen** (không train)
- Normalize L2 đầu ra

---

### `src/models/text_encoder.py` — M3
**Người phụ trách:** Thành viên 3

**Lớp:** `CLIPTextEncoder(nn.Module)`

```python
# Input:  list[str] (batch of texts)
# Output: text_embedding [B, 512]
```

- Model: `openai/clip-vit-base-patch32` (HuggingFace)
- Tự động tokenize qua `tokenize(texts: list[str]) → dict`
- Tất cả tham số **Frozen**

---

### `src/models/fusion.py` — M5
**Người phụ trách:** Thành viên 5

**Lớp:** `CrossAttentionFusion(nn.Module)`

```python
CrossAttentionFusion(embed_dim=512, num_heads=8, output_dim=1024, dropout=0.1)
# Input:  query [B, 512], key_value [B, 512]
# Output: fused [B, 1024]
```

**Luồng xử lý:**
1. `MultiheadAttention(Q=query, K=kv, V=kv)` → attention output
2. LayerNorm + Residual connection
3. FeedForward (512→1024→1024)
4. LayerNorm + Residual connection

---

### `src/models/crossmodal.py` — M5
**Người phụ trách:** Thành viên 5

**Lớp:** `CrossModalFND(nn.Module)`

```python
# Input:  images [B,3,224,224], texts List[str]
# Output: logits [B, 2]
```

Kết hợp: `CLIPImageEncoder` + `CLIPTextEncoder` + `CrossAttentionFusion` + MLP

---

### `src/train.py` — M5
**Người phụ trách:** Thành viên 5

**Cấu hình mặc định:**

| Hyperparameter | Giá trị |
|---|---|
| EPOCHS | 15 |
| BATCH_SIZE | 8 |
| LR | 2e-4 |
| WEIGHT_DECAY | 1e-4 |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Loss | CrossEntropyLoss |
| Mixed Precision | AMP (tự động bật nếu có GPU) |
| Best model | Lưu theo Val F1 cao nhất |

**Output:** `saved_models/best_model.pt`, `results/training_log.csv`

---

### `src/train_baseline.py` — M4
**Người phụ trách:** Thành viên 4

**Output:** `saved_models/baseline_{text|image|concat}.pt`, `results/reports/baseline_comparison.md`

---

### `src/evaluate.py` — M6
**Người phụ trách:** Thành viên 6

Thực hiện **2 cấp độ đánh giá**:

| Cấp độ | Mô tả |
|---|---|
| **[A] Model-Level** | Load từng checkpoint, chạy trên `test.csv`, so sánh công bằng |
| **[B] Full Pipeline** | Nạp `best_model.pt`, chạy end-to-end, tính đủ Accuracy/F1/AUC/Confusion Matrix |

**Output:** 4 PNG + `final_evaluation.md`

---

### `src/evaluation/metrics.py` — M6

```python
evaluate_model(model, loader, device) → dict
# Trả về: accuracy, f1, auc_roc, conf_matrix, all_labels, all_preds, all_probs
```

### `src/evaluation/visualize.py` — M6

| Hàm | Output |
|---|---|
| `plot_confusion_matrix(conf_matrix, save_path)` | `confusion_matrix.png` |
| `plot_roc_curve(labels, probs, auc, save_path)` | `roc_curve.png` |
| `plot_training_history(log_csv, save_path)` | `training_history.png` |
| `plot_model_comparison(comparison_dict, save_path)` | `model_comparison.png` |

### `src/evaluation/html_report.py` — M6

Tạo file `results/dashboard.html` — self-contained, tương tác với Chart.js.

---

### `prepare_data.py` — Công cụ chung

Script chuẩn bị dữ liệu hoàn chỉnh. Thực hiện:

1. Quét đệ quy toàn bộ ảnh trong thư mục `images/`
2. Đọc `tweets.txt`, ghép với ảnh theo `image_id`
3. Lưu `data/processed/dataset.csv`
4. **Image-level split**: Chia 360 ảnh thành train/val/test trước, sau đó gán tweet theo ảnh → tránh data leakage

---

## 5. Luồng dữ liệu (Data Flow)

```
[tweets.txt] ──┐
               ├──► prepare_data.py ──► dataset.csv ──► train/val/test.csv
[images/]    ──┘                              │
                                              │
                              ┌───────────────▼───────────────┐
                              │         FakeNewsDataset        │
                              │    (image_tensor, text, label) │
                              └───────────────┬───────────────┘
                                              │
                           ┌──────────────────▼──────────────────┐
                           │           CrossModalFND              │
                           │  CLIP Image ──┐                      │
                           │               ├── CrossAttention ── MLP ──► [REAL/FAKE]
                           │  CLIP Text  ──┘                      │
                           └──────────────────┬──────────────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │    evaluate.py (M6)            │
                              │  Model-Level + Full Pipeline   │
                              └───────────────┬───────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │              Output                       │
                         │  final_evaluation.md · 4 PNG · dashboard.html │
                         └──────────────────────────────────────────┘
```

---

## 6. Kết quả thực nghiệm

### 6.1 Quá trình training CrossModalFND

| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|---|---|---|---|---|
| 1 | 0.0349 | 0.6315 | 87.62% | 0.9134 |
| 2 | 0.0056 | 0.4940 | 93.32% | 0.9551 |
| 3 | 0.0042 | 0.4889 | 93.40% | 0.9556 |
| **4** | **0.0039** | **0.5035** | **96.32%** | **0.9758** ← Best |
| 5–15 | ~0.000 | >0.62 | 96.29% | 0.9755 | *(Bão hòa)* |

### 6.2 So sánh mô hình trên tập Test (Image-Level Split)

| Mô hình | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Text-Only | 53.52% | 0.4926 | — |
| Image-Only | 86.50% | 0.8646 | — |
| Concat | 87.96% | 0.8788 | — |
| **CrossModal-FND** | **88.16%** | **0.8809** | **0.9357** |

### 6.3 Confusion Matrix (CrossModalFND trên Test)

```
                 Dự đoán REAL   Dự đoán FAKE
Thực Tế REAL        419             109
Thực Tế FAKE         12             482
```

- **Precision FAKE:** 482 / (109+482) = 81.6%
- **Recall FAKE:** 482 / (12+482) = 97.6% ← Mô hình rất nhạy với FAKE
- **False Positive Rate:** 109/528 = 20.6% ← Điểm cần cải thiện

---

## 7. Hướng dẫn chạy lại

### Yêu cầu
```bash
pip install -r requirements.txt
```

### Bước 1 – Chuẩn bị dữ liệu
```powershell
$env:PYTHONPATH='.'; python prepare_data.py
```

### Bước 2 – Train baseline (M4)
```powershell
$env:PYTHONPATH='.'; python src/train_baseline.py
```

### Bước 3 – Train mô hình chính (M5)
```powershell
$env:PYTHONPATH='.'; python src/train.py
```

### Bước 4 – Đánh giá (M6)
```powershell
$env:PYTHONPATH='.'; python src/evaluate.py
```

### Bước 5 – Tạo HTML Dashboard
```powershell
$env:PYTHONPATH='.'; python src/evaluation/html_report.py
start results/dashboard.html
```

---

## 8. Các vấn đề đã xử lý

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ReduceLROnPlateau verbose=True` | PyTorch ≥ 2.2 xóa tham số này | Xóa `verbose=True` |
| `build_dataset_csv()` thiếu tham số | Gọi không đúng signature | Dùng `prepare_data.py` với đường dẫn đầy đủ |
| 0 ảnh tìm thấy | Ảnh nằm trong thư mục con sự kiện, không phải flat | Dùng `glob` quét đệ quy `**/*.jpg` |
| Data leakage (Accuracy 100%) | Chia theo tweet-level → cùng ảnh ở train và test | Đổi sang **image-level split** |
| Baseline đánh giá bằng random weights | Không lưu checkpoint sau khi train | Thêm `torch.save()` trong `train_baseline.py` |
| UnicodeEncodeError trên Windows | Terminal dùng CP1252 | Đặt `PYTHONIOENCODING=utf-8` |
