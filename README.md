#  Multimodal Fake News Detection

> Hệ thống phát hiện tin tức giả mạo bằng cách phân tích đồng thời **hình ảnh** và **văn bản**  
> sử dụng mô hình học sâu đa phương thức dựa trên **CLIP + Cross-Attention Fusion**.

![Status](https://img.shields.io/badge/Status-Hoàn%20Thành-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Test F1](https://img.shields.io/badge/Test%20F1-0.9296-success)
![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9772-success)

**Môn học:** Xử Lý Ảnh & Thị Giác Máy Tính  
**Dataset:** [Image Verification Corpus – MediaEval 2015](https://github.com/MKLab-ITI/image-verification-corpus)  
**Mô hình tham khảo:** [Results in Engineering, Vol. 26, 2025](https://www.sciencedirect.com/science/article/pii/S2590123025008291)

---

## Kết Quả Thực Nghiệm 

| Mô hình | Test Accuracy | Test F1 | AUC-ROC |
|---|---|---|---|
| Text-Only (Baseline) | 88.55% | 0.8850 | — |
| Image-Only (Baseline) | 90.12% | 0.9011 | — |
| Concat (Baseline) | 94.13% | 0.9413 | — |
| **CrossModal-FND (Của nhóm)** | **92.95%** | **0.9296** | **0.9772** |

> Đánh giá trên **image-level split** (không có data leakage): ảnh trong tập test chưa từng xuất hiện ở tập train.

---

## Kiến Trúc Mô Hình

```
[Hình ảnh] ──► CLIP Image Encoder (Frozen) ──► img_emb [B, 512]
                                                               │
                                                    CrossAttentionFusion
                                                    (Q=img, K=V=txt)
                                                               │
[Văn bản]  ──► CLIP Text Encoder  (Frozen) ──► txt_emb [B, 512]
                                                               │
                                                      MLP Classifier
                                                               │
                                                     [REAL] / [FAKE]
```

**Ý tưởng cốt lõi:** Nếu hình ảnh và văn bản mô tả hai chủ đề mâu thuẫn nhau, Cross-Attention sẽ nhận ra sự không khớp đó → dấu hiệu của Tin Giả.

| Thành phần | Chi tiết |
|---|---|
| Backbone | `openai/clip-vit-base-patch32` (ViT-B/32) |
| Tổng tham số | 154,431,234 |
| Tham số có thể học | 3,153,922 (2.04%) — chỉ Fusion + MLP |
| CLIP Encoders | Frozen hoàn toàn |
| Attention Heads | 8 |
| Output Fusion | 1024-dim |

---

## Cấu Trúc Thư Mục

```
Course_work/
│
├── README.md                        # File này
├── requirements.txt                 # Thư viện cần cài
├── prepare_data.py                  # Script chuẩn bị dữ liệu (image-level split)
├── .gitignore
│
├── docs/                            # Tài liệu dự án
│   ├── plan.md                      # Kế hoạch & tiến độ 6 phases 
│   ├── status.md                    # Trạng thái hoàn thành 
│   ├── contracts.md                 # Interface contracts giữa các thành viên
│   ├── team_assignment.md           # Phân công nhiệm vụ
│   ├── requirements.md              # Yêu cầu kỹ thuật
│   └── codebase_audit.md            # Tổng quan toàn bộ codebase
│
├── data/                            # [.gitignore] Dữ liệu không push
│   ├── raw/image-verification-corpus/mediaeval2015/
│   ├── processed/dataset.csv        # 11.524 mẫu (360 ảnh)
│   └── splits/                      # train / val / test (image-level)
│
├── src/
│   ├── data/
│   │   ├── dataset.py               # M1 – FakeNewsDataset (PyTorch)
│   │   ├── preprocessing.py         # M1 – clean_text, build_dataset_csv
│   │   └── download_dataset.py      # M1 – Script kiểm tra dữ liệu
│   │
│   ├── models/
│   │   ├── image_encoder.py         # M2 – CLIPImageEncoder [B, 512]
│   │   ├── text_encoder.py          # M3 – CLIPTextEncoder  [B, 512]
│   │   ├── baselines.py             # M4 – Text/Image/ConcatClassifier
│   │   ├── fusion.py                # M5 – CrossAttentionFusion [B, 1024]
│   │   └── crossmodal.py            # M5 – CrossModalFND (mô hình chính)
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # M6 – Accuracy, F1, AUC-ROC, Confusion Matrix
│   │   ├── visualize.py             # M6 – 4 biểu đồ PNG
│   │   └── html_report.py           # M6 – HTML Dashboard tương tác (Chart.js)
│   │
│   ├── train.py                     # M5 – Train CrossModalFND (AMP + AdamW)
│   ├── train_baseline.py            # M4 – Train 3 baseline + lưu checkpoint
│   └── evaluate.py                  # M6 – Model-Level + Full Pipeline Evaluation
│
├── saved_models/                    # [.gitignore] Checkpoints
│   ├── best_model.pt                # CrossModalFND tốt nhất (~589MB)
│   ├── baseline_text.pt
│   ├── baseline_image.pt
│   └── baseline_concat.pt
│
└── results/                         # [.gitignore một phần]
    ├── training_log.csv
    ├── dashboard.html               # HTML Dashboard tương tác
    ├── plots/                       # 4 file PNG biểu đồ
    └── reports/
        ├── baseline_comparison.md   # Kết quả M4
        └── final_evaluation.md      # Báo cáo đánh giá cuối M6
```

---

## Hướng Dẫn Cài Đặt & Chạy

### Yêu cầu
- Python 3.10+
- GPU NVIDIA (khuyến nghị) — CPU cũng chạy được nhưng chậm hơn ~10x
- Windows: PowerShell; Linux/Mac: Terminal

### Bước 1 – Clone về máy
```bash
git clone https://github.com/btsuu25-dev/multimodal-fake-news-detection
cd multimodal-fake-news-detection
```

### Bước 2 – Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 3 – Tải & giải nén dataset thủ công
Tải dataset từ [Zenodo – Image Verification Corpus](https://zenodo.org/record/3256839), đặt vào:
```
data/raw/image-verification-corpus/mediaeval2015/devset/
    ├── tweets.txt
    └── MediaEval2015_DevSet_Images.rar  (giải nén ra thư mục images/)
```

### Bước 4 – Chuẩn bị dữ liệu (image-level split)
```powershell
# Windows
$env:PYTHONPATH='.'; $env:PYTHONIOENCODING='utf-8'; python prepare_data.py
```
```bash
# Linux/Mac
PYTHONPATH=. python prepare_data.py
```

### Bước 5 – Train Baseline (M4)
```powershell
$env:PYTHONPATH='.'; python src/train_baseline.py
```

### Bước 6 – Train CrossModalFND (M5)
```powershell
$env:PYTHONPATH='.'; python src/train.py
```
Mô hình tốt nhất lưu vào `saved_models/best_model.pt`.

### Bước 7 – Đánh giá (M6)
```powershell
$env:PYTHONPATH='.'; python src/evaluate.py
```

### Bước 8 – Xem HTML Dashboard
```powershell
$env:PYTHONPATH='.'; python src/evaluation/html_report.py
start results/dashboard.html
```

---

## Tài Liệu Chi Tiết

| File | Nội dung |
|---|---|
| [docs/codebase_audit.md](docs/codebase_audit.md) | Kiến trúc, API từng file, luồng dữ liệu |
| [docs/plan.md](docs/plan.md) | Kế hoạch 6 phases + kết quả thực tế |
| [docs/status.md](docs/status.md) | Checklist hoàn thành + bugs đã xử lý |
| [docs/contracts.md](docs/contracts.md) | Interface contracts giữa các thành viên |
| [results/dashboard.html](results/dashboard.html) | HTML Dashboard kết quả evaluation |
| [results/reports/final_evaluation.md](results/reports/final_evaluation.md) | Báo cáo đánh giá cuối |

---

## Thông Tin Nhóm

| Thành viên | Phụ trách | File chính |
|---|---|---|
| M1 | Dữ liệu & Preprocessing | `src/data/`, `prepare_data.py` |
| M2 | Image Encoder | `src/models/image_encoder.py` |
| M3 | Text Encoder | `src/models/text_encoder.py` |
| M4 | Baseline Models | `src/models/baselines.py`, `src/train_baseline.py` |
| M5 | CrossModalFND | `src/models/fusion.py`, `crossmodal.py`, `src/train.py` |
| M6 | Evaluation & Dashboard | `src/evaluation/`, `src/evaluate.py` |
| M7 | Tài liệu | `docs/`, `README.md` |
