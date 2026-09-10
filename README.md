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

### Bước 4 – Train mô hình baseline (để so sánh)
```bash
python src/train_baseline.py
```
Kết quả sẽ được lưu vào `results/reports/baseline_comparison.md`.

### Bước 5 – Train mô hình chính CrossModalFND
```bash
python src/train.py
```
Mô hình tốt nhất sẽ được lưu vào `saved_models/best_model.pt`.

---

## Kết Quả Kỳ Vọng

| Mô hình | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Text-Only (Baseline) | ~70% | ~0.68 | ~0.76 |
| Image-Only (Baseline) | ~62% | ~0.60 | ~0.68 |
| Concat (Baseline) | ~75% | ~0.73 | ~0.82 |
| **CrossModal-FND (ours)** | **~83%** | **~0.81** | **~0.90** |

---

## Thông Tin Nhóm

| Thành viên | Phụ trách |
|---|---|
| Thành viên 1 | Dữ liệu – `src/data/` |
| Thành viên 2 | Mã hóa hình ảnh – `src/models/image_encoder.py` |
| Thành viên 3 | Mã hóa văn bản – `src/models/text_encoder.py` |
| Thành viên 4 | Mô hình Baseline – `src/models/baselines.py` |
| Thành viên 5 | Mô hình chính – `src/models/crossmodal.py`, `src/train.py` |
| Thành viên 6 | Đánh giá – `src/evaluation/` |
| Thành viên 7 | Tài liệu – `docs/`, `README.md` |
