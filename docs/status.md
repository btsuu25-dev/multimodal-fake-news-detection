# 📊 Trạng Thái Dự Án (Project Status)
## Cập nhật lần cuối: 2026-09-13

---

## Tiến Độ Tổng Thể

```
[██████████] 100% Hoàn thành ✅
```

| Giai đoạn | Trạng thái | Người phụ trách | Kết quả |
|---|---|---|---|
| Phase 1 – Chuẩn bị dữ liệu | ✅ Hoàn thành | M1 | `train/val/test.csv` (image-level split) |
| Phase 2 – Trích xuất đặc trưng | ✅ Hoàn thành | M2, M3 | `image_encoder.py`, `text_encoder.py` |
| Phase 3 – Mô hình Baseline | ✅ Hoàn thành | M4 | `baseline_*.pt`, `baseline_comparison.md` |
| Phase 4 – Mô hình CrossModal-FND | ✅ Hoàn thành | M5 | `best_model.pt`, Val F1 = 0.9668 |
| Phase 5 – Đánh giá | ✅ Hoàn thành | M6 | Test F1 = 0.9296, AUC = 0.9772 |
| Phase 6 – Audit Codebase | ✅ Hoàn thành | M5+M6 | `codebase_audit.md`, `dashboard.html` |

---

## Kết Quả Cuối Cùng 🏆

| Chỉ số | Giá trị |
|---|---|
| **Test Accuracy** | **92.95%** |
| **Test F1 Score** | **0.9296** |
| **Test AUC-ROC** | **0.9772** |
| Best Val F1 | 0.9668 (Epoch 4/15) |
| Tổng mẫu hợp lệ | 11.524 tweets (360 ảnh) |
| Tập Test | 1.022 tweets (54 ảnh, chưa từng nhìn) |

---

## Đã Hoàn Thành ✅

### Nghiên cứu & Lên kế hoạch
- [x] Nghiên cứu đề bài từ giảng viên
- [x] Tìm hiểu dataset Image Verification Corpus (MediaEval)
- [x] Tìm hiểu bài báo khoa học gợi ý
- [x] Xây dựng Implementation Plan đầy đủ
- [x] Tạo `docs/contracts.md` – ký kết interface giữa các thành viên
- [x] Phân công nhiệm vụ `docs/team_assignment.md`

### Phase 1 – Dữ liệu (M1)
- [x] Tải dataset Image Verification Corpus từ Zenodo
- [x] Giải nén `MediaEval2015_DevSet_Images.rar` (WinRAR)
- [x] Quét ảnh đệ quy, ghép `tweets.txt` với ảnh theo `image_id`
- [x] Xây dựng `data/processed/dataset.csv` (11.524 mẫu)
- [x] **Image-level split** (tránh data leakage): Train/Val/Test theo ảnh, không theo tweet
- [x] `src/data/dataset.py` – PyTorch Dataset chuẩn
- [x] `src/data/preprocessing.py` – Tiền xử lý văn bản

### Phase 2 – Feature Encoder (M2, M3)
- [x] `src/models/image_encoder.py` – CLIPImageEncoder (ViT-B/32, Frozen)
- [x] `src/models/text_encoder.py` – CLIPTextEncoder (ViT-B/32, Frozen)
- [x] Embedding output chuẩn [B, 512] cho cả hai encoder

### Phase 3 – Baseline (M4)
- [x] `src/models/baselines.py` – TextOnly, ImageOnly, ConcatClassifier
- [x] `src/train_baseline.py` – Script train với checkpoint saving
- [x] Lưu `saved_models/baseline_{text|image|concat}.pt`
- [x] Xuất `results/reports/baseline_comparison.md`

### Phase 4 – CrossModal-FND (M5)
- [x] `src/models/fusion.py` – CrossAttentionFusion (8 heads, output 1024)
- [x] `src/models/crossmodal.py` – CrossModalFND (154M params, 2% trainable)
- [x] `src/train.py` – Training pipeline (AMP, GradScaler, ReduceLROnPlateau)
- [x] Fix lỗi `verbose=True` của PyTorch ≥ 2.2
- [x] Lưu `saved_models/best_model.pt` (~589MB)
- [x] Xuất `results/training_log.csv` (15 epochs)

### Phase 5 – Evaluation (M6)
- [x] `src/evaluation/metrics.py` – Accuracy, F1, AUC-ROC, Confusion Matrix
- [x] `src/evaluation/visualize.py` – 4 biểu đồ PNG
- [x] `src/evaluate.py` – Đánh giá 2 cấp: Model-Level + Full Pipeline
- [x] Fix data leakage: Baseline dùng checkpoint thật, không phải random weights
- [x] `src/evaluation/html_report.py` – HTML Dashboard tương tác (Chart.js)
- [x] Xuất `results/dashboard.html` (314KB, self-contained)
- [x] Xuất `results/reports/final_evaluation.md`

### Phase 6 – Audit (M5+M6)
- [x] `docs/codebase_audit.md` – Tổng quan toàn bộ codebase
- [x] Cập nhật `docs/status.md` (file này)
- [x] Cập nhật `docs/plan.md`
- [x] Push toàn bộ lên GitHub (`main` branch)

---

## Vấn Đề Đã Xử Lý ✅

| # | Vấn đề | Cách giải quyết |
|---|---|---|
| 1 | `ReduceLROnPlateau verbose=True` lỗi | Xóa tham số `verbose` (deprecated PyTorch ≥ 2.2) |
| 2 | `build_dataset_csv()` thiếu tham số | Dùng `prepare_data.py` với đường dẫn đầy đủ |
| 3 | Ảnh nằm trong thư mục con sự kiện | Dùng `glob` quét đệ quy `**/*.jpg` |
| 4 | Data leakage → Accuracy 100% | Đổi sang **image-level split** |
| 5 | Baseline đánh giá bằng random weights | Thêm `torch.save()`, load checkpoint khi evaluate |
| 6 | UnicodeEncodeError trên Windows terminal | Đặt `$env:PYTHONIOENCODING='utf-8'` |

---

## Ghi Chú Quan Trọng 📌

> **Lưu ý `.gitignore`**: Các thư mục `data/`, `saved_models/`, `results/plots/` và `results/*.csv`
> bị loại trừ khỏi Git. Đây là chủ ý — chỉ push **code** lên GitHub, không push data và model nặng.
> Khi giảng viên chạy lại sẽ tự sinh ra các file này.

> **Cách chạy lại toàn bộ**: Xem hướng dẫn trong `docs/codebase_audit.md` phần 7.

---

## Nhật Ký Thay Đổi (Changelog)

| Ngày | Cập nhật |
|---|---|
| 2026-09-10 | Khởi tạo dự án, xây dựng Implementation Plan, tạo thư mục `docs/` |
| 2026-09-12 | Hoàn thành M1 (data), M2 (image encoder), M3 (text encoder), M5 (CrossModalFND + train) |
| 2026-09-13 | Hoàn thành M4 (baseline + checkpoint), M6 (evaluation 2 cấp + HTML dashboard), fix data leakage, audit codebase |
