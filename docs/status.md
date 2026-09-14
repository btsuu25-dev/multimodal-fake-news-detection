# 📊 Trạng Thái Dự Án (Project Status)
## Cập nhật lần cuối: 2026-09-10

---

## Tiến Độ Tổng Thể

```
[██░░░░░░░░] 20% Hoàn thành
```

| Giai đoạn | Trạng thái | Ghi chú |
|---|---|---|
| Phase 1 – Chuẩn bị dữ liệu | ⏳ Chưa bắt đầu | Đang nghiên cứu kế hoạch |
| Phase 2 – Trích xuất đặc trưng | ⏳ Chưa bắt đầu | |
| Phase 3 – Mô hình Baseline | ⏳ Chưa bắt đầu | |
| Phase 4 – Mô hình CrossModal-FND | ⏳ Chưa bắt đầu | |
| Phase 5 – Đánh giá | ⏳ Chưa bắt đầu | |
| Phase 6 – Audit Codebase | ⏳ Chưa bắt đầu | |

---

## Đã Hoàn Thành ✅

- [x] **Nghiên cứu đề bài** từ giảng viên
- [x] **Tìm hiểu dataset** gợi ý (Fakeddit, VERITE, NewsCLIPpings)
- [x] **Tìm hiểu bài báo khoa học** gợi ý (Results in Engineering, 2025)
- [x] **Xây dựng Implementation Plan** đầy đủ 6 tiêu chí
- [x] **Hiểu nguyên lý hoạt động** của mô hình (CLIP + Cross-Attention Fusion)
- [x] **Hiểu khái niệm Feature** (manh mối) trong AI
- [x] **Thống nhất cách triển khai**: Python Scripts (`.py`), không dùng Notebook để train
- [x] **Tạo thư mục `docs/`** với các file tài liệu dự án

---

## Đang Làm 🔄

- [ ] **Đọc và nghiên cứu Implementation Plan** để nắm vững toàn bộ kế hoạch trước khi code

---

## Cần Làm Tiếp Theo 📝

1. **Quyết định dataset**: Fakeddit (lớn, ~5GB ảnh) hay VERITE (nhỏ, dễ chạy thử)?
2. **Kiểm tra môi trường máy tính**: Có GPU không? RAM bao nhiêu GB?
3. **Cài Python và các thư viện** cần thiết (`pip install -r requirements.txt`)
4. **Bắt đầu Phase 1**: Tải và khám phá dữ liệu

---

## Vấn Đề & Quyết Định Đang Chờ ⚠️

| # | Vấn đề | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Chọn dataset nào? | ❓ Chờ quyết định | Fakeddit (to) vs VERITE (nhỏ) |
| 2 | Máy tính có GPU không? | ❓ Chờ xác nhận | Ảnh hưởng đến cấu hình mô hình |
| 3 | Có làm demo app Gradio không? | ❓ Chờ quyết định | Giảng viên có yêu cầu demo không? |

---

## Ghi Chú Quan Trọng 📌

> **Lưu ý về Code**: Tất cả file Python phải được viết dưới dạng script (`.py`), 
> không dùng Jupyter Notebook để train model (theo yêu cầu của giảng viên).

> **Dataset**: Nếu máy tính không đủ mạnh, ưu tiên dùng **VERITE** (~3K mẫu) 
> thay vì Fakeddit (~1 triệu mẫu) để đảm bảo chạy được trong thời gian hạn định.

---

## Nhật Ký Thay Đổi (Changelog)

| Ngày | Cập nhật |
|---|---|
| 2026-09-10 | Khởi tạo dự án, xây dựng Implementation Plan, tạo thư mục `docs/` |
