# 🗺️ Kế Hoạch Thực Hiện Chi Tiết (Project Plan)
## Dự Án: Anti Fake News – Multimodal Fake News Detection
## Trạng thái: ✅ **HOÀN THÀNH TOÀN BỘ** (2026-09-13)

---

## Tổng Quan Kiến Trúc Mô Hình

Mô hình có tên **CrossModal-FND** hoạt động theo nguyên lý sau:

```
[Hình ảnh] ──► CLIP Visual Encoder (Frozen) ──► img_emb [512]
                                                              ├──► CrossAttentionFusion ──► MLP ──► [REAL/FAKE]
[Văn bản]  ──► CLIP Text Encoder   (Frozen) ──► txt_emb [512]
```

> **Ý tưởng cốt lõi:** Nếu Ảnh và Chữ mô tả hai chủ đề khác nhau (mâu thuẫn), đó là dấu hiệu của Tin Giả.  
> Cross-Attention học cách "đối chiếu" ảnh và chữ thay vì chỉ ghép thô.

---

## Các Giai Đoạn Thực Hiện (Phases)

### ✅ PHASE 1 – Chuẩn Bị Dữ Liệu (M1)
**Trạng thái: HOÀN THÀNH**

- [x] Tải dataset Image Verification Corpus (MediaEval 2015) từ Zenodo
- [x] Giải nén `MediaEval2015_DevSet_Images.rar` → 360 ảnh
- [x] Quét ảnh đệ quy, ghép với `tweets.txt` → 11.524 mẫu hợp lệ
- [x] Lưu `data/processed/dataset.csv`
- [x] **Image-level split** (70/15/15): 252/54/54 ảnh → tránh data leakage
- [x] `src/data/dataset.py` – PyTorch Dataset trả về `(Tensor[3,224,224], str, int)`
- [x] `src/data/preprocessing.py` – `clean_text()`, `build_dataset_csv()`, `split_dataset()`

**Kết quả bàn giao:** ✅ `data/splits/train.csv` (9.715 dòng), `val.csv` (870 dòng), `test.csv` (939 dòng)

---

### ✅ PHASE 2 – Trích Xuất Đặc Trưng (M2, M3)
**Trạng thái: HOÀN THÀNH**

- [x] `src/models/image_encoder.py` – `CLIPImageEncoder` (ViT-B/32, Frozen, output [B,512])
- [x] `src/models/text_encoder.py` – `CLIPTextEncoder` (ViT-B/32, Frozen, output [B,512])
- [x] Tự động tải weights từ HuggingFace khi chạy lần đầu

**Kết quả bàn giao:** ✅ Cả hai encoder hoạt động, embedding chuẩn 512 chiều

---

### ✅ PHASE 3 – Mô Hình Nền Baseline (M4)
**Trạng thái: HOÀN THÀNH**

- [x] `src/models/baselines.py` – `TextOnlyClassifier`, `ImageOnlyClassifier`, `ConcatClassifier`
- [x] `src/train_baseline.py` – Train 10 epochs, lưu checkpoint tốt nhất
- [x] `saved_models/baseline_text.pt`, `baseline_image.pt`, `baseline_concat.pt`

**Kết quả bàn giao:** ✅

| Mô hình | Test Accuracy | Test F1 |
|---|---|---|
| Text-Only | 53.52% | 0.4926 |
| Image-Only | 86.50% | 0.8646 |
| Concat | 87.96% | 0.8788 |

---

### ✅ PHASE 4 – Mô Hình Chính CrossModal-FND (M5)
**Trạng thái: HOÀN THÀNH**

- [x] `src/models/fusion.py` – `CrossAttentionFusion(embed_dim=512, num_heads=8, output_dim=1024)`
- [x] `src/models/crossmodal.py` – `CrossModalFND` (154M params, 2% trainable)
- [x] `src/train.py` – AMP, GradScaler, ReduceLROnPlateau, lưu best checkpoint theo Val F1
- [x] `saved_models/best_model.pt` (~589MB)

**Cấu hình thực tế đã dùng:**

| Hyperparameter | Giá trị |
|---|---|
| CLIP Variant | `ViT-B/32` |
| Batch Size | `8` |
| Learning Rate | `2e-4` |
| Epochs | `15` |
| Optimizer | `AdamW` (weight_decay=1e-4) |
| Scheduler | `ReduceLROnPlateau` (patience=3, factor=0.5) |
| Loss | `CrossEntropyLoss` |
| Mixed Precision | `torch.amp.GradScaler` |

**Kết quả bàn giao:** ✅

| Epoch | Val F1 | Ghi chú |
|---|---|---|
| 1 | 0.9134 | |
| 4 | **0.9758** | ← Best checkpoint được lưu |
| 15 | 0.9755 | Bão hòa (overfitting nhẹ) |

---

### ✅ PHASE 5 – Đánh Giá Mô Hình (M6)
**Trạng thái: HOÀN THÀNH**

- [x] `src/evaluation/metrics.py` – `evaluate_model()` → Accuracy, F1, AUC-ROC, Confusion Matrix
- [x] `src/evaluation/visualize.py` – 4 hàm vẽ biểu đồ PNG
- [x] `src/evaluate.py` – 2 cấp đánh giá: Model-Level + Full Pipeline
- [x] `src/evaluation/html_report.py` – HTML Dashboard tương tác (Chart.js)

**Kết quả bàn giao:** ✅

| Chỉ số | CrossModal-FND | Concat (Baseline tốt nhất) |
|---|---|---|
| **Test Accuracy** | **88.16%** | 87.96% |
| **Test F1** | **0.8809** | 0.8788 |
| **AUC-ROC** | **0.9357** | — |

Confusion Matrix (CrossModalFND):
```
                 Dự đoán REAL   Dự đoán FAKE
Thực Tế REAL        419             109    (FP = 20.6%)
Thực Tế FAKE         12             482    (FN = 2.4%  ← rất tốt!)
```

---

### ✅ PHASE 6 – Kiểm Tra Chất Lượng Code (Audit)
**Trạng thái: HOÀN THÀNH**

- [x] Tất cả file Python có docstring giải thích bằng tiếng Việt
- [x] `docs/codebase_audit.md` – Tổng quan kiến trúc, luồng dữ liệu, API từng file
- [x] `docs/status.md` – Cập nhật 100% hoàn thành
- [x] Chạy pipeline từ đầu đến cuối không báo lỗi (10/10 module OK)
- [x] Push lên GitHub, tag commit rõ ràng

---

## Thứ Tự Ưu Tiên

```
Phase 1 ✅ → Phase 2 ✅ → Phase 3 ✅ → Phase 4 ✅ → Phase 5 ✅ → Phase 6 ✅
```

---

## Kết Luận

> Mô hình CrossModal-FND với kiến trúc **CLIP + Cross-Attention Fusion** đã vượt qua tất cả baseline
> trên tập test chưa từng nhìn (image-level split). Mô hình đặc biệt nhạy với tin FAKE (Recall FAKE = 97.6%),
> phù hợp với bài toán thực tế nơi bỏ sót tin giả nguy hiểm hơn cảnh báo nhầm.
