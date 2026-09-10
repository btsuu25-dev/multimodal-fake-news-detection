# 👥 Phân Công Công Việc Nhóm
## Dự Án: Anti Fake News – Multimodal Fake News Detection

---

## Sơ Đồ Phụ Thuộc

```
M1 (Dữ liệu) ──────────────────────────────────────┐
M2 (Mã hóa ảnh) ────────────────────────────────────┤──► M5 (Mô hình chính) ──► M6 (Đánh giá)
M3 (Mã hóa văn bản) ──────────────────────────────┐ │                             ▲
M4 (Baseline) ── cần M2, M3 ──────────────────────┘ │                             │
M7 (Báo cáo) ── không phụ thuộc ai ─────────────────┴─────────────────── dùng output của M6
```

> M1, M2, M3, M7 có thể bắt đầu **ngay lập tức** mà không cần chờ ai.

---

## Thành Viên 1 – Dữ Liệu

### Phụ trách
Tải và chuẩn bị toàn bộ dữ liệu cho dự án.

### Yêu cầu thực hiện

**1. Tải dataset Image Verification Corpus**
- Chạy lệnh: `python src/data/download_dataset.py`
- Dataset sẽ được tải về thư mục `data/raw/`

**2. Giải nén và ghép dữ liệu**
- Giải nén file `.rar` chứa hình ảnh
- Đọc file `mediaeval2015/devset/tweets.txt` (14.484 dòng)
- Mỗi dòng có: `tweet_id`, `image_id`, `label (fake/real)`, `event`
- Tạo file CSV tổng hợp: `data/processed/dataset.csv` với 3 cột: `image_path`, `text`, `label`
- Loại bỏ các dòng không có ảnh tương ứng

**3. Chia tập dữ liệu**
- Chia ngẫu nhiên thành 3 tập (tỷ lệ 70 / 15 / 15):
  - `data/splits/train.csv`
  - `data/splits/val.csv`
  - `data/splits/test.csv`

**4. Viết `src/data/dataset.py`**
- Class `FakeNewsDataset` nhận vào đường dẫn file CSV
- Trả về cặp `(ảnh đã resize 224x224, text, nhãn 0/1)` khi được gọi

### Output bàn giao

| File/Thư mục | Nội dung |
|---|---|
| `data/raw/image-verification-corpus/` | Dataset gốc đã tải về |
| `data/processed/dataset.csv` | CSV tổng hợp đã làm sạch |
| `data/splits/train.csv` | ~10.138 mẫu (70%) |
| `data/splits/val.csv` | ~2.172 mẫu (15%) |
| `data/splits/test.csv` | ~2.172 mẫu (15%) |
| `src/data/dataset.py` | Class `FakeNewsDataset` |
| `src/data/preprocessing.py` | Hàm resize ảnh, làm sạch text |

### Kiểm tra hoàn thành
```python
from src.data.dataset import FakeNewsDataset
dataset = FakeNewsDataset('data/splits/train.csv')
image, text, label = dataset[0]
print(image.shape)  # phải ra: torch.Size([3, 224, 224])
print(label)        # phải ra: 0 hoặc 1
```

---

## Thành Viên 2 – Mã Hóa Hình Ảnh

### Phụ trách
Viết bộ phận chuyển đổi hình ảnh thành vector số bằng mô hình CLIP.

### Yêu cầu thực hiện

**1. Viết `src/models/image_encoder.py`**
- Class `CLIPImageEncoder` dùng `openai/clip-vit-base-patch32` từ HuggingFace
- Hàm `forward(image)`: nhận tensor ảnh `[B, 3, 224, 224]` → trả ra vector `[B, 512]`
- **Bắt buộc Freeze toàn bộ tham số CLIP** (không cho phép thay đổi khi train)
- Hàm `get_embedding_dim()`: trả về số `512`

> Không cần chờ dữ liệu thật — dùng `torch.randn(4, 3, 224, 224)` để test ngay.

### Output bàn giao

| File | Nội dung |
|---|---|
| `src/models/image_encoder.py` | Class `CLIPImageEncoder` hoàn chỉnh |

### Kiểm tra hoàn thành
```python
from src.models.image_encoder import CLIPImageEncoder
import torch

encoder = CLIPImageEncoder()
output = encoder(torch.randn(4, 3, 224, 224))
print(output.shape)  # phải ra: torch.Size([4, 512])
```

---

## Thành Viên 3 – Mã Hóa Văn Bản

### Phụ trách
Viết bộ phận chuyển đổi văn bản thành vector số và tính độ tương đồng ảnh-chữ.

### Yêu cầu thực hiện

**1. Viết `src/models/text_encoder.py`**
- Class `CLIPTextEncoder` dùng `openai/clip-vit-base-patch32` từ HuggingFace
- Hàm `tokenize(texts: list[str])`: nhận list câu thô → trả về tensor token
- Hàm `forward(tokens)`: nhận token → trả ra vector `[B, 512]`
- **Bắt buộc Freeze toàn bộ tham số CLIP**

**2. Viết hàm `compute_similarity(img_emb, txt_emb)`**
- Nhận vào vector ảnh `[B, 512]` và vector chữ `[B, 512]`
- Trả về Cosine Similarity `[B, 1]` (từ -1 đến 1)
- Đây là "manh mối" chính: giá trị thấp → ảnh và chữ không liên quan → nghi ngờ FAKE

> Không cần chờ dữ liệu thật — dùng câu chữ bất kỳ để test.

### Output bàn giao

| File | Nội dung |
|---|---|
| `src/models/text_encoder.py` | Class `CLIPTextEncoder` + hàm `compute_similarity` |

### Kiểm tra hoàn thành
```python
from src.models.text_encoder import CLIPTextEncoder
encoder = CLIPTextEncoder()
tokens = encoder.tokenize(["A man running on the beach"])
output = encoder(tokens)
print(output.shape)  # phải ra: torch.Size([1, 512])
```

---

## Thành Viên 4 – Mô Hình Baseline

### Phụ trách
Xây dựng 3 mô hình đơn giản để làm mốc so sánh với mô hình chính.

### Yêu cầu thực hiện

**1. Viết `src/models/baselines.py`** chứa 3 class:

- `TextOnlyClassifier`: nhận vector chữ `[B, 512]` → phân loại FAKE/REAL
- `ImageOnlyClassifier`: nhận vector ảnh `[B, 512]` → phân loại FAKE/REAL
- `ConcatClassifier`: nhận vector ghép `[B, 1024]` → phân loại FAKE/REAL

Mỗi mô hình dùng cấu trúc đơn giản: `Linear → ReLU → Dropout → Linear`

**2. Viết `src/train_baseline.py`**
- Train và test cả 3 mô hình, mỗi cái 10 epochs, batch_size=8
- In ra Accuracy và F1 của mỗi mô hình
- Lưu kết quả vào `results/reports/baseline_comparison.md`

### Output bàn giao

| File | Nội dung |
|---|---|
| `src/models/baselines.py` | 3 class mô hình baseline |
| `src/train_baseline.py` | Script chạy và so sánh 3 mô hình |
| `results/reports/baseline_comparison.md` | Bảng số liệu kết quả |

### Kết quả kỳ vọng (bảng so sánh)

| Mô hình | Accuracy | F1 |
|---|---|---|
| Text-Only | ~70% | ~0.68 |
| Image-Only | ~62% | ~0.60 |
| Concat | ~75% | ~0.73 |

---

## Thành Viên 5 – Mô Hình Chính

### Phụ trách
Xây dựng mô hình CrossModal-FND hoàn chỉnh và chạy huấn luyện.

> Cần M1 + M2 + M3 hoàn thành trước. Trong khi chờ, có thể viết code với dữ liệu giả.

### Yêu cầu thực hiện

**1. Viết `src/models/fusion.py`**
- Class `CrossAttentionFusion` dùng `nn.MultiheadAttention` (8 heads)
- Nhận vào `img_emb [B, 512]` và `txt_emb [B, 512]`
- Thực hiện cross-attention 2 chiều: ảnh→chữ và chữ→ảnh
- Trả ra vector đã trộn `[B, 2048]`

**2. Viết `src/models/crossmodal.py`**
- Class `CrossModalFND` lắp ráp: encoder ảnh (M2) + encoder chữ (M3) + fusion + MLP classifier
- Luồng xử lý: `ảnh → img_emb` + `chữ → txt_emb` → tính similarity → cross-attention → MLP → `P(FAKE/REAL)`
- Đầu ra cuối: xác suất `[B, 2]`

**3. Viết `src/train.py`**
- Bật Mixed Precision (`torch.cuda.amp`) để tiết kiệm VRAM
- Optimizer: `AdamW(lr=2e-4)`, chạy 15 epochs, batch_size=8
- Mỗi epoch in: `Loss` trên train và `F1` trên val
- Tự động lưu mô hình tốt nhất vào `saved_models/best_model.pt`

### Output bàn giao

| File | Nội dung |
|---|---|
| `src/models/fusion.py` | Class `CrossAttentionFusion` |
| `src/models/crossmodal.py` | Class `CrossModalFND` hoàn chỉnh |
| `src/train.py` | Script huấn luyện chính |
| `saved_models/best_model.pt` | Mô hình tốt nhất sau 15 epochs |
| `results/training_log.csv` | Log Loss và F1 theo từng epoch |

### Kết quả log kỳ vọng
```
Epoch 01/15 | Train Loss: 0.6821 | Val F1: 0.6234
Epoch 02/15 | Train Loss: 0.5934 | Val F1: 0.6891  ← Lưu checkpoint!
...
Epoch 15/15 | Train Loss: 0.2341 | Val F1: 0.8312
```

---

## Thành Viên 6 – Đánh Giá Kết Quả

### Phụ trách
Tính toán chỉ số, vẽ biểu đồ và tổng hợp bảng so sánh tất cả mô hình.

> Cần M4 + M5 hoàn thành trước. Trong khi chờ, có thể viết sẵn code dùng số liệu giả để test.

### Yêu cầu thực hiện

**1. Viết `src/evaluation/metrics.py`**
- Hàm `compute_all_metrics(y_true, y_pred, y_prob)` → trả về dict: `Accuracy, Precision, Recall, F1, AUC-ROC`

**2. Viết `src/evaluation/visualize.py`** với 4 hàm vẽ biểu đồ:
- `plot_confusion_matrix()` → lưu `results/plots/confusion_matrix.png`
- `plot_roc_curve()` → lưu `results/plots/roc_curve.png`
- `plot_training_history()` → lưu `results/plots/training_history.png`
- `plot_model_comparison()` → lưu `results/plots/model_comparison.png`

**3. Tổng hợp bảng so sánh cuối cùng** vào `results/reports/final_evaluation.md`

### Output bàn giao

| File | Nội dung |
|---|---|
| `src/evaluation/metrics.py` | Hàm tính chỉ số |
| `src/evaluation/visualize.py` | Hàm vẽ 4 biểu đồ |
| `results/plots/confusion_matrix.png` | Ma trận nhầm lẫn |
| `results/plots/roc_curve.png` | Đường cong ROC |
| `results/plots/training_history.png` | Biểu đồ Loss/F1 qua 15 epochs |
| `results/plots/model_comparison.png` | Biểu đồ cột so sánh các mô hình |
| `results/reports/final_evaluation.md` | Báo cáo kết quả tổng hợp |

### Kết quả kỳ vọng (bảng so sánh)

| Mô hình | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Text-Only | ~70% | 0.68 | 0.76 |
| Image-Only | ~62% | 0.60 | 0.68 |
| Concat | ~75% | 0.73 | 0.82 |
| **CrossModal-FND** | **~83%** | **0.81** | **0.90** |

---

## Thành Viên 7 – Báo Cáo & Tài Liệu

### Phụ trách
Viết tài liệu và cập nhật tiến độ nhóm.

> Không phụ thuộc ai — có thể bắt đầu ngay và làm song song hoàn toàn.

### Yêu cầu thực hiện

**1. Viết `README.md`**
- Mô tả ngắn gọn dự án là gì
- Hướng dẫn cài đặt từng bước: clone repo → `pip install -r requirements.txt` → chạy lệnh train
- Mô tả từng file trong `src/`

**2. Cập nhật `docs/status.md` thường xuyên**
- Cập nhật ít nhất 2 lần/tuần dựa trên báo cáo của cả nhóm
- Đánh dấu ✅ khi từng thành viên hoàn thành phần của mình

**3. Kiểm tra chất lượng code (sau khi có code từ mọi người)**
- Chạy `black src/` để tự động định dạng code
- Đảm bảo mỗi file Python đều có chú thích (comments)

### Output bàn giao

| File | Nội dung |
|---|---|
| `README.md` | Hướng dẫn cài đặt và chạy |
| `docs/status.md` | Tiến độ được cập nhật đều đặn |
| `requirements.txt` | Danh sách thư viện đầy đủ |

---

### ❌ TUYỆT ĐỐI KHÔNG push lên GitHub

| Loại file | Ví dụ | Lý do |
|---|---|---|
| **Dữ liệu thô** | `data/raw/`, `data/processed/` | Nặng hàng GB, GitHub không cho phép |
| **Mô hình đã train** | `saved_models/*.pt` | File nhị phân nặng, không cần thiết |
| **Kết quả sinh ra** | `results/plots/*.png` | Mỗi người tự chạy sẽ tự có |
| **Môi trường Python** | `venv/`, `__pycache__/` | Mỗi máy cài riêng, dùng chung sẽ lỗi |
| **File hệ điều hành** | `.DS_Store`, `Thumbs.db` | Rác, không liên quan |

> Đã có file `.gitignore` tự động chặn các thứ trên.

---

### ✅ CHỈ push những thứ này

- **Code Python** (`.py`): các file trong `src/`, `docs/`
- **File cấu hình**: `requirements.txt`, `.gitignore`, `README.md`
- **Tài liệu**: `docs/*.md`, `slides/*.pptx`


> [!IMPORTANT]
> **Không dùng `git add .` (thêm tất cả)** vì có thể vô tình push dữ liệu hoặc model nặng lên GitHub.

---

### 🔀 Các bạn tạo nhánh riêng để thực hiện

```bash
# Tạo nhánh riêng cho mình
git checkout -b feature/m2-image-encoder

# Push nhánh riêng lên
git push origin feature/m2-image-encoder
```

Sau khi xong, báo cho trưởng nhóm để **merge** vào nhánh `main`.

