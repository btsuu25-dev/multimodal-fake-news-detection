# 📜 Hợp Đồng Giao Diện (Interface Contracts)
## Dự Án: Anti Fake News – Multimodal Fake News Detection

> **Đây là file quan trọng nhất của dự án.**
> Mỗi thành viên (và AI hỗ trợ) **PHẢI** đọc file này trước khi viết code.
> Tất cả tên class, tên hàm, kiểu dữ liệu đã được thống nhất cố định ở đây.
> **Không được tự ý thay đổi bất kỳ thứ gì trong file này mà không thông báo cho cả nhóm.**

---

## Quy Tắc Chung (Bắt buộc với mọi thành viên)

| # | Quy tắc | Lý do |
|---|---|---|
| 1 | **Chỉ chỉnh sửa file trong phạm vi được phân công** | Tránh ghi đè code của người khác |
| 2 | **Không đổi tên class, hàm, hoặc tham số** đã định nghĩa trong file này | Các module khác phụ thuộc vào tên chính xác này |
| 3 | **Chạy đoạn "Kiểm tra hoàn thành"** trước khi push | Đảm bảo code chạy được |
| 4 | **Cập nhật `requirements.txt`** nếu thêm thư viện mới | Người khác mới clone về sẽ cài đủ |
| 5 | **Không xóa hoặc đổi tên thư mục** trong `src/` | Cấu trúc thư mục là cố định |
| 6 | **Không sửa file của thành viên khác** kể cả khi thấy lỗi | Báo cho người đó tự sửa |

---

## Hợp Đồng Module 1 – Dataset (`src/data/dataset.py`)

**Người phụ trách: Thành viên 1**

```python
from src.data.dataset import FakeNewsDataset
from torch.utils.data import DataLoader

# Cách khởi tạo (KHÔNG ĐƯỢC ĐỔI TÊN CLASS)
dataset = FakeNewsDataset(csv_path: str)

# len() phải hoạt động
len(dataset)  # → int

# Indexing phải trả đúng 3 giá trị này, đúng thứ tự
image, text, label = dataset[0]

# Kiểu dữ liệu bắt buộc:
# image → torch.Tensor, shape = [3, 224, 224], dtype = float32
# text  → str (câu văn tiếng Anh gốc, chưa tokenize)
# label → int (0 = REAL, 1 = FAKE)

# Phải dùng được với DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

---

## Hợp Đồng Module 2 – Image Encoder (`src/models/image_encoder.py`)

**Người phụ trách: Thành viên 2**

```python
from src.models.image_encoder import CLIPImageEncoder
import torch

# Cách khởi tạo (KHÔNG ĐƯỢC ĐỔI TÊN CLASS)
encoder = CLIPImageEncoder()

# Hàm forward (KHÔNG ĐƯỢC ĐỔI TÊN)
img_emb = encoder(image_tensor)
# image_tensor shape → [B, 3, 224, 224]
# img_emb     shape → [B, 512]
# img_emb dtype     → float32

# Hàm lấy số chiều (KHÔNG ĐƯỢC ĐỔI TÊN)
dim = encoder.get_embedding_dim()
# dim → 512 (int)

# Tham số CLIP phải bị freeze
for param in encoder.parameters():
    assert param.requires_grad == False
```

---

## Hợp Đồng Module 3 – Text Encoder (`src/models/text_encoder.py`)

**Người phụ trách: Thành viên 3**

```python
from src.models.text_encoder import CLIPTextEncoder
import torch

# Cách khởi tạo (KHÔNG ĐƯỢC ĐỔI TÊN CLASS)
encoder = CLIPTextEncoder()

# Hàm tokenize (KHÔNG ĐƯỢC ĐỔI TÊN)
tokens = encoder.tokenize(texts)
# texts  → list[str], ví dụ: ["A man running", "Explosion at airport"]
# tokens → dict tensor (output của CLIP tokenizer, trên cùng device)

# Hàm forward (KHÔNG ĐƯỢC ĐỔI TÊN)
txt_emb = encoder(tokens)
# txt_emb shape → [B, 512]
# txt_emb dtype → float32

# Hàm tính Cosine Similarity (KHÔNG ĐƯỢC ĐỔI TÊN)
from src.models.text_encoder import compute_similarity
sim = compute_similarity(img_emb, txt_emb)
# img_emb shape → [B, 512]
# txt_emb shape → [B, 512]
# sim     shape → [B, 1], giá trị trong khoảng [-1, 1]
```

---

## Hợp Đồng Module 4 – Baselines (`src/models/baselines.py`)

**Người phụ trách: Thành viên 4**

```python
from src.models.baselines import TextOnlyClassifier, ImageOnlyClassifier, ConcatClassifier
import torch

# TextOnlyClassifier (KHÔNG ĐƯỢC ĐỔI TÊN CLASS)
model = TextOnlyClassifier()
logits = model(txt_emb)
# txt_emb shape → [B, 512]
# logits  shape → [B, 2]  (logit cho [REAL, FAKE])

# ImageOnlyClassifier (KHÔNG ĐƯỢC ĐỔI TÊN CLASS)
model = ImageOnlyClassifier()
logits = model(img_emb)
# img_emb shape → [B, 512]
# logits  shape → [B, 2]

# ConcatClassifier (KHÔNG ĐƯỢC ĐỔI TÊN CLASS)
model = ConcatClassifier()
combined = torch.cat([img_emb, txt_emb], dim=-1)  # [B, 1024]
logits = model(combined)
# logits shape → [B, 2]
```

---

## Hợp Đồng Module 5 – Mô Hình Chính (`src/models/crossmodal.py`)

**Người phụ trách: Thành viên 5**

```python
from src.models.crossmodal import CrossModalFND
import torch

# Cách khởi tạo (KHÔNG ĐƯỢC ĐỔI TÊN CLASS)
model = CrossModalFND()

# Hàm forward — nhận RAW INPUT, tự xử lý nội bộ (KHÔNG ĐƯỢC ĐỔI TÊN)
logits = model(image_tensor, text_list)
# image_tensor → torch.Tensor, shape [B, 3, 224, 224]
# text_list    → list[str], độ dài B
# logits       → torch.Tensor, shape [B, 2]  (logit cho [REAL, FAKE])

# Lưu và load mô hình
torch.save(model.state_dict(), 'saved_models/best_model.pt')
model.load_state_dict(torch.load('saved_models/best_model.pt'))
```

---

## Hợp Đồng Module 6 – Đánh Giá (`src/evaluation/metrics.py`)

**Người phụ trách: Thành viên 6**

```python
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    plot_model_comparison
)
import numpy as np

# Hàm tính chỉ số (KHÔNG ĐƯỢC ĐỔI TÊN)
results = compute_all_metrics(y_true, y_pred, y_prob)
# y_true → np.ndarray shape [N], giá trị 0 hoặc 1
# y_pred → np.ndarray shape [N], giá trị 0 hoặc 1
# y_prob → np.ndarray shape [N], giá trị float [0,1] (xác suất FAKE)
# results → dict với keys: 'accuracy', 'precision', 'recall', 'f1', 'auc_roc'

# Các hàm vẽ (KHÔNG ĐƯỢC ĐỔI TÊN, lưu file vào đúng thư mục)
plot_confusion_matrix(y_true, y_pred, save_path='results/plots/confusion_matrix.png')
plot_roc_curve(y_true, y_prob, save_path='results/plots/roc_curve.png')
plot_training_history(log_csv_path='results/training_log.csv', save_path='results/plots/training_history.png')
plot_model_comparison(comparison_dict, save_path='results/plots/model_comparison.png')
```

---

## Cấu Trúc Thư Mục Cố Định

Không ai được thêm, xóa, hoặc đổi tên thư mục sau đây:

```
Course_work/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/
├── src/
│   ├── data/
│   │   ├── dataset.py          ← M1
│   │   └── preprocessing.py    ← M1
│   ├── models/
│   │   ├── image_encoder.py    ← M2
│   │   ├── text_encoder.py     ← M3
│   │   ├── baselines.py        ← M4
│   │   ├── fusion.py           ← M5
│   │   └── crossmodal.py       ← M5
│   ├── evaluation/
│   │   ├── metrics.py          ← M6
│   │   └── visualize.py        ← M6
│   ├── train_baseline.py       ← M4
│   └── train.py                ← M5
├── saved_models/
└── results/
    ├── plots/
    └── reports/
```
