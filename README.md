# Anti Fake News – Multimodal Fake News Detection

Hệ thống phát hiện tin tức giả mạo bằng cách phân tích đồng thời **hình ảnh** và **văn bản** sử dụng mô hình học sâu đa phương thức (Multimodal Deep Learning).

**Môn học:** Xử Lý Ảnh & Thị Giác Máy Tính  
**Dataset:** [Image Verification Corpus (MediaEval)](https://github.com/MKLab-ITI/image-verification-corpus)  
**Mô hình tham khảo:** [Results in Engineering, Vol. 26, 2025](https://www.sciencedirect.com/science/article/pii/S2590123025008291)

---

## Kiến Trúc Mô Hình

```
[Hình ảnh] ──► CLIP Image Encoder ──► img_emb [B, 512]
                                                         ├──► Cross-Attention Fusion ──► MLP ──► FAKE / REAL
[Văn bản]  ──► CLIP Text Encoder  ──► txt_emb [B, 512]
```

**Ý tưởng cốt lõi:** Nếu hình ảnh và văn bản mô tả hai chủ đề mâu thuẫn nhau, đó là dấu hiệu mạnh của tin giả.

---

## Cấu Trúc Thư Mục

```
Course_work/
│
├── README.md                        # File này
├── requirements.txt                 # Danh sách thư viện cần cài
├── .gitignore                       # Các file không push lên GitHub
│
├── docs/                            # Tài liệu dự án
│   ├── requirements.md              # Yêu cầu dự án
│   ├── plan.md                      # Kế hoạch thực hiện chi tiết
│   ├── status.md                    # Theo dõi tiến độ
│   ├── team_assignment.md           # Phân công công việc nhóm
│   └── contracts.md                 # Hợp đồng giao diện (Interface Contracts)
│
├── data/                            # Dữ liệu (không push lên GitHub)
│   ├── raw/                         # Dataset gốc sau khi tải về
│   │   └── image-verification-corpus/
│   ├── processed/
│   │   └── dataset.csv              # CSV tổng hợp đã làm sạch
│   └── splits/
│       ├── train.csv                # ~70% dữ liệu (dùng để học)
│       ├── val.csv                  # ~15% dữ liệu (dùng để kiểm tra trong lúc học)
│       └── test.csv                 # ~15% dữ liệu (dùng để chấm điểm cuối)
│
├── src/                             # Mã nguồn chính
│   │
│   ├── data/                        # Xử lý dữ liệu (Thành viên 1)
│   │   ├── download_dataset.py      # Script tải dataset về máy
│   │   ├── dataset.py               # Class FakeNewsDataset cho PyTorch
│   │   └── preprocessing.py         # Hàm resize ảnh, làm sạch văn bản
│   │
│   ├── models/                      # Các mô hình AI
│   │   ├── image_encoder.py         # CLIPImageEncoder – mã hóa hình ảnh (Thành viên 2)
│   │   ├── text_encoder.py          # CLIPTextEncoder – mã hóa văn bản (Thành viên 3)
│   │   ├── baselines.py             # 3 mô hình đơn giản để so sánh (Thành viên 4)
│   │   ├── fusion.py                # CrossAttentionFusion – trộn ảnh và chữ (Thành viên 5)
│   │   └── crossmodal.py            # CrossModalFND – mô hình chính hoàn chỉnh (Thành viên 5)
│   │
│   ├── evaluation/                  # Đánh giá kết quả (Thành viên 6)
│   │   ├── metrics.py               # Tính Accuracy, F1, AUC-ROC
│   │   └── visualize.py             # Vẽ Confusion Matrix, ROC Curve, biểu đồ so sánh
│   │
│   ├── train_baseline.py            # Script train 3 mô hình baseline (Thành viên 4)
│   └── train.py                     # Script train mô hình chính CrossModalFND (Thành viên 5)
│
├── saved_models/                    # Mô hình đã train (không push lên GitHub)
│   └── best_model.pt                # Mô hình tốt nhất sau quá trình huấn luyện
│
└── results/                         # Kết quả sinh ra (không push lên GitHub)
    ├── training_log.csv             # Log Loss và F1 qua từng epoch
    ├── plots/
    │   ├── confusion_matrix.png     # Ma trận nhầm lẫn
    │   ├── roc_curve.png            # Đường cong ROC
    │   ├── training_history.png     # Biểu đồ Loss/F1 qua các epoch
    │   └── model_comparison.png     # So sánh các mô hình
    └── reports/
        ├── baseline_comparison.md   # Kết quả 3 mô hình baseline
        └── final_evaluation.md      # Báo cáo đánh giá tổng hợp
```

---

## Hướng Dẫn Cài Đặt & Chạy

### Yêu cầu
- Python 3.10+
- GPU NVIDIA (khuyến nghị, có thể chạy CPU nhưng chậm hơn)

### Bước 1 – Clone dự án về máy
```bash
git clone <link-repo-github-của-nhóm>
cd Course_work
```

### Bước 2 – Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 3 – Tải dataset
```bash
python src/data/download_dataset.py
```
Dataset sẽ được tải về thư mục `data/raw/` tự động.

## 4. Cấu trúc thư mục

```text
multimodal-fake-news-detection/

├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── contracts.md
│   ├── plan.md
│   ├── requirements.md
│   ├── status.md
│   └── team_assignment.md
│
├── data/
│   └── raw/
│
└── src/
    ├── data/
    │   ├── dataset.py
    │   ├── download_dataset.py
    │   └── preprocessing.py
    │
    ├── models/
    │   ├── baselines.py
    │   ├── image_encoder.py
    │   └── text_encoder.py
    │
    └── train_baseline.py

```
## 5. Cài đặt và chạy

### 5.1. Clone repository

```bash
git clone https://github.com/btsuu25-dev/multimodal-fake-news-detection.git
cd multimodal-fake-news-detection

---
```
## 6. Mô tả các file trong `src/`

### `src/data/dataset.py`

Dùng để đọc và quản lý dữ liệu phục vụ quá trình huấn luyện mô hình.

### `src/data/download_dataset.py`

Dùng để tải Image Verification Corpus và kiểm tra các file dữ liệu cần thiết.

### `src/data/preprocessing.py`

Dùng để tiền xử lý và chuẩn hóa dữ liệu đầu vào trước khi đưa vào mô hình.

### `src/models/image_encoder.py`

Xử lý và mã hóa dữ liệu hình ảnh, tạo biểu diễn đặc trưng hình ảnh phục vụ mô hình.

### `src/models/text_encoder.py`

Xử lý và mã hóa dữ liệu văn bản, tạo biểu diễn đặc trưng văn bản phục vụ mô hình.

### `src/models/baselines.py`

Chứa các mô hình Baseline dùng làm cơ sở so sánh với mô hình Multimodal.

### `src/train_baseline.py`

Thực hiện quá trình huấn luyện và đánh giá các mô hình Baseline.

## 7. Thành viên nhóm

| Thành viên | MSSV | Phân công |
|---|---|---|
| Tô Hoàng Vũ | 056205009808 | M1 – Dữ liệu & Dataset |
| Phạm Anh Tuấn | 058205001597 | M2 – Xử lý ảnh |
| Nguyễn Trọng Vân Khuyên | 068305006610 | M3 – Xử lý văn bản |
| Bùi Trọng Sửu | 024205002460 | M4 – Baseline |
| Võ Duy Khanh | 052205011285 | M5 – Mô hình Multimodal |
| Nguyễn Thành Tài | 052205017040 | M6 – Đánh giá mô hình |
| Nguyễn Gia Hân | 079306010087 | M7 – Báo cáo & Tài liệu |

## 8. Tiến độ dự án

Tiến độ thực hiện của các thành viên được cập nhật tại file `docs/status.md`.

File này được cập nhật định kỳ dựa trên báo cáo của các thành viên trong nhóm.

Trạng thái hoàn thành của từng thành viên sẽ được đánh dấu `✅` khi phần việc đã hoàn thành và được bàn giao.