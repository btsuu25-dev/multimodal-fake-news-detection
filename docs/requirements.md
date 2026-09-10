# 📋 Yêu Cầu Dự Án (Project Requirements)
## Môn: Xử Lý Ảnh & Thị Giác Máy Tính
### Đề tài: Anti Fake News – Multimodal Fake News Detection

---

## 1. Bối Cảnh & Mục Tiêu

Dự án yêu cầu xây dựng một hệ thống AI có khả năng **phát hiện tin tức giả mạo** bằng cách phân tích đồng thời **hình ảnh** và **văn bản** (Multimodal). Đây là bài tập lớn (Coursework) cuối kỳ của bộ môn.

**Tài liệu tham khảo chuyên đề:**
- Nguồn Dataset: https://github.com/thcheung/awesome-fake-news-datasets
- Mô hình tham khảo: https://www.sciencedirect.com/science/article/pii/S2590123025008291
  *(Bài báo: "A systematic review of multimodal fake news detection on social media using deep learning models" - Results in Engineering, 2025)*

---

## 2. Tiêu Chí Đánh Giá (Metrics)

Dự án cần đáp ứng và bám sát **6 tiêu chí** cốt lõi sau:

| # | Tiêu chí | Mô tả |
|---|---|---|
| 1 | **Problem Definition** | Xác định rõ bài toán là gì, đầu vào/đầu ra như thế nào |
| 2 | **Features** | Liệt kê các đặc trưng (features) AI sẽ dùng (in-scope) và không dùng (out-of-scope) |
| 3 | **Solution → Tech → AI** | Đề xuất giải pháp cụ thể, công nghệ sử dụng, kiến trúc mô hình |
| 4 | **Implementation** | Triển khai mã nguồn thực tế, thiết kế cấu trúc thư mục, pipeline hoàn chỉnh |
| 5 | **Evaluation** | Đánh giá kết quả ở cả mức độ mô hình (model-level) và toàn pipeline |
| 6 | **Audit Codebase** | Kiểm tra chất lượng mã nguồn: chuẩn mực, sạch sẽ, có tài liệu kỹ thuật |

---

## 3. Yêu Cầu Kỹ Thuật Bắt Buộc

### 3.1 Về Mã Nguồn (Codebase)
- ✅ **Viết bằng Python Scripts (`.py`)** — Không sử dụng Jupyter Notebook cho quá trình huấn luyện (train model).
- ✅ Mã nguồn phải thực thi mượt mà từ đầu đến cuối, không phát sinh lỗi (bug).
- ✅ Bắt buộc có chú thích (comments) giải thích rõ ràng kiến trúc và luồng dữ liệu.

### 3.2 Về Dữ Liệu (Dataset)
- ✅ Sử dụng ít nhất **1 dataset chuẩn** từ danh sách tài liệu tham khảo đề xuất.
- ✅ Chia tập dữ liệu thành 3 phần tiêu chuẩn: **Train / Validation / Test** (tỷ lệ 70/15/15).

### 3.3 Về Mô Hình AI
- ✅ Kiến trúc bắt buộc là **Đa phương thức (Multimodal)**: xử lý và tổng hợp cả Ảnh (Vision) + Chữ (Text).
- ✅ Phải tiến hành **so sánh đối chiếu** với ít nhất 1 mô hình nền (Baseline) đơn giản hơn.
- ✅ Phải xuất ra kết quả đánh giá bằng các chỉ số đo lường chuẩn: **Accuracy, F1-Score, AUC-ROC**.

### 3.4 Về Báo Cáo Kỹ Thuật
- ✅ Cung cấp sơ đồ khối (architecture diagram) của mô hình.
- ✅ Lập bảng so sánh kết quả thống kê giữa các cấu hình/mô hình.
- ✅ Biểu diễn kết quả trực quan bằng hình ảnh (đồ thị Loss, Confusion Matrix).

---

## 4. Giới Hạn Phạm Vi (Constraints)

- **Không** yêu cầu xử lý các định dạng Video hoặc Audio.
- **Không** yêu cầu thu thập dữ liệu mạng xã hội theo thời gian thực (tương tác, lượt thích, chia sẻ).
- **Không** yêu cầu triển khai (deploy) lên máy chủ production (hệ thống chỉ cần chạy ổn định ở môi trường local).
- Phần giao diện minh họa (nếu có) được phép giới hạn ở mức độ chạy thử trên máy tính cá nhân.

