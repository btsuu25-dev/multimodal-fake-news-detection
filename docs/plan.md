# 🗺️ Kế Hoạch Thực Hiện Chi Tiết (Project Plan)
## Dự Án: Anti Fake News – Multimodal Fake News Detection

---

## Tổng Quan Kiến Trúc Mô Hình

Mô hình có tên **CrossModal-FND** hoạt động theo nguyên lý sau:

```
[Hình ảnh] ──► CLIP Visual Encoder ──► Vector số (img_emb)
                                                          ├──► Cross-Attention Fusion ──► [REAL / FAKE]
[Văn bản]  ──► CLIP Text Encoder  ──► Vector số (txt_emb)
```

> **Ý tưởng cốt lõi:** Nếu Ảnh và Chữ mô tả hai chủ đề khác nhau hoàn toàn (mâu thuẫn nhau), thì đó là dấu hiệu mạnh của Tin Giả.

---

## Các Giai Đoạn Thực Hiện (Phases)

### 🔵 PHASE 1 – Chuẩn Bị Dữ Liệu
**Mục tiêu:** Có một bộ dữ liệu sạch, sẵn sàng đưa vào mô hình.

**Các bước:**
1. Tải dataset **Image Verification Corpus (MediaEval)** từ đúng danh sách của giảng viên
2. Phân tích dữ liệu (EDA): xem phân phối nhãn REAL/FAKE, độ dài câu văn, kích thước ảnh
3. Làm sạch dữ liệu: giải nén file `.rar`, map ID ảnh với dòng Tweet tương ứng
4. Chia dữ liệu thành 3 phần: **Train (70%) / Val (15%) / Test (15%)**
5. Viết `dataset.py`: lớp Python để đọc và tiền xử lý dữ liệu

**Kết quả bàn giao:** File `dataset.py` hoạt động, dữ liệu đã được chia sẵn

---

### 🔵 PHASE 2 – Trích Xuất Đặc Trưng (Feature Engineering)
**Mục tiêu:** Chuyển đổi Ảnh và Chữ thành các con số mà mô hình hiểu được.

**Các bước:**
1. Cài đặt mô hình CLIP (từ thư viện HuggingFace)
2. Viết `encoders.py`: dùng CLIP để trích xuất đặc trưng ảnh và chữ
3. Chạy thử và kiểm tra kích thước đầu ra (phải là vector 512 chiều)
4. (Nâng cao) Thêm tính năng phân tích cảm xúc văn bản (Sentiment Score)

**Kết quả bàn giao:** File `encoders.py`, kết quả embedding kiểm tra

---

### 🔵 PHASE 3 – Xây Dựng Mô Hình Nền (Baseline Models)
**Mục tiêu:** Có kết quả so sánh để chứng minh mô hình chính của chúng ta tốt hơn.

**Các mô hình Baseline:**

| Mô hình | Mô tả | Lý do làm |
|---|---|---|
| Text-Only | Chỉ dùng chữ, bỏ qua ảnh | Cho thấy chữ một mình có đủ không |
| Image-Only | Chỉ dùng ảnh, bỏ qua chữ | Cho thấy ảnh một mình có đủ không |
| Simple Concat | Ghép số của ảnh + số của chữ | Baseline đơn giản nhất của Multimodal |

**Kết quả bàn giao:** Bảng so sánh Accuracy/F1 của 3 mô hình nền

---

### 🔴 PHASE 4 – Xây Dựng Mô Hình Chính (CrossModal-FND)
**Mục tiêu:** Xây dựng và huấn luyện mô hình đề xuất hoàn chỉnh.

**Các bước:**
1. Viết `fusion.py`: Bộ phận Cross-Attention để đối chiếu Ảnh vs Chữ
2. Viết `crossmodal.py`: Lắp ráp toàn bộ hệ thống
3. Viết `train.py`: Script huấn luyện mô hình (chạy bằng lệnh `python src/train.py`)
4. Chạy huấn luyện với dữ liệu thật, theo dõi Loss giảm dần
5. Lưu mô hình tốt nhất vào thư mục `saved_models/`

**Cấu hình kỹ thuật đã xác định:**

| Hạng mục | Giá trị | Lý do lựa chọn |
|---|---|---|
| **CLIP Variant** | `ViT-B/32` | Nhẹ, phù hợp VRAM hạn chế, embedding 512 chiều |
| **Chiến lược Train** | Freeze Encoder + chỉ train Fusion/Classifier | Tiết kiệm VRAM tối đa, hội tụ nhanh hơn |
| **Batch Size** | `8` | Ổn định, tránh lỗi out-of-memory |
| **Learning Rate** | `2e-4` | Cao hơn vì chỉ train lớp nhỏ, hội tụ nhanh |
| **Epochs** | `15` | Đủ để hội tụ với bộ dữ liệu subset |
| **Optimizer** | `AdamW` | Chuẩn cho các bài toán fine-tuning |
| **Loss Function** | `CrossEntropyLoss` | Phù hợp bài toán phân loại nhị phân |
| **Mixed Precision** | `fp16` (torch AMP) | Giảm ~50% VRAM tiêu thụ, không ảnh hưởng độ chính xác |
| **Dataset** | Image Verification Corpus (MediaEval) | Đúng 100% list của cô, nhẹ (khoảng 50MB), dễ chạy |

**Kết quả bàn giao:** File mô hình đã train (`best_model.pt`), biểu đồ Loss qua các epoch


---

### 🔵 PHASE 5 – Đánh Giá Mô Hình (Evaluation)
**Mục tiêu:** Đo lường chính xác hiệu suất của mô hình, chuẩn bị số liệu cho báo cáo.

**Các chỉ số cần báo cáo:**

| Chỉ số | Ý nghĩa |
|---|---|
| **Accuracy** | Bao nhiêu % tin tức phân loại đúng |
| **Precision** | Trong những cái bị đánh dấu FAKE, bao nhiêu cái thật sự là FAKE |
| **Recall** | Trong tất cả tin FAKE thật sự, ta tìm được bao nhiêu % |
| **F1-Score** | Chỉ số kết hợp Precision và Recall |
| **AUC-ROC** | Chỉ số tổng quát về khả năng phân loại |

**Kết quả bàn giao:** File `eval.py`, bảng số liệu, Confusion Matrix, biểu đồ ROC

---

### 🔵 PHASE 6 – Kiểm Tra Chất Lượng Code (Audit)
**Mục tiêu:** Đảm bảo code sạch sẽ, dễ đọc, đạt chuẩn nộp bài.

**Checklist:**
- [ ] Tất cả file có chú thích (comments) giải thích rõ ràng bằng tiếng Việt
- [ ] Không có số "ma thuật" trong code (ví dụ: thay vì ghi `512` thì phải ghi `Config.EMBEDDING_DIM`)
- [ ] Chạy lệnh `black src/` để tự động định dạng code đẹp
- [ ] Chạy thử pipeline từ đầu đến cuối không báo lỗi
- [ ] Cập nhật `README.md` hướng dẫn chạy

---

## Thứ Tự Ưu Tiên

```
Phase 1 (Dữ liệu) → Phase 2 (Encoder) → Phase 3 (Baseline) 
    → Phase 4 (Mô hình chính) → Phase 5 (Đánh giá) → Phase 6 (Audit)
```
