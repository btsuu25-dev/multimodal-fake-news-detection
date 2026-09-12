"""
dataset.py – PyTorch Dataset cho dự án Anti Fake News
======================================================
Cung cấp class FakeNewsDataset tuân thủ đúng Interface Contract
đã định nghĩa trong docs/contracts.md.

Cách dùng:
    from src.data.dataset import FakeNewsDataset
    from torch.utils.data import DataLoader

    dataset = FakeNewsDataset('data/splits/train.csv')
    image, text, label = dataset[0]
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)

Tác giả: Thành viên 1 (M1 – Dữ Liệu)
"""

import logging
import torch
from torch.utils.data import Dataset
import pandas as pd

# Import hàm tiền xử lý từ cùng package
from src.data.preprocessing import preprocess_image, clean_text, IMAGE_SIZE

# ──────────────────────────────────────────────────────────
# Cấu hình logging
# ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Hằng số
# ──────────────────────────────────────────────────────────
# Tensor ảnh trống dùng làm fallback khi ảnh bị lỗi
# Shape: [3, 224, 224] – khớp đúng với yêu cầu contract
_FALLBACK_IMAGE = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

# Mapping nhãn văn bản → số (phòng trường hợp CSV dùng chữ)
LABEL_MAP = {
    "real": 0,
    "0":    0,
    "fake": 1,
    "1":    1,
}


class FakeNewsDataset(Dataset):
    """
    PyTorch Dataset đọc dữ liệu từ file CSV cho bài toán Fake News Detection.

    ┌─────────────────────────────────────────────────────────┐
    │  Interface Contract (docs/contracts.md – KHÔNG đổi)    │
    │                                                         │
    │  dataset = FakeNewsDataset(csv_path: str)               │
    │  image, text, label = dataset[i]                        │
    │                                                         │
    │  image → torch.Tensor [3, 224, 224], dtype=float32     │
    │  text  → str (câu gốc, CHƯA tokenize)                  │
    │  label → int (0 = REAL, 1 = FAKE)                      │
    └─────────────────────────────────────────────────────────┘

    Cấu trúc file CSV đầu vào (phải có đúng 3 cột này):
        image_path  : Đường dẫn đến file ảnh
        text        : Nội dung văn bản tweet
        label       : Nhãn 0 (REAL) hoặc 1 (FAKE)

    Xử lý ảnh lỗi:
        Nếu ảnh không đọc được → dùng tensor zeros [3, 224, 224] làm fallback.
        Không raise Exception để tránh crash DataLoader.
    """

    def __init__(self, csv_path: str):
        """
        Khởi tạo dataset từ file CSV.

        Args:
            csv_path (str): Đường dẫn đến file CSV (train.csv / val.csv / test.csv).
                            File phải có 3 cột: image_path, text, label.

        Raises:
            FileNotFoundError : Nếu file CSV không tồn tại.
            ValueError        : Nếu CSV thiếu cột bắt buộc.
        """
        # ── Kiểm tra file tồn tại ───────────────────────────
        import os
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")

        # ── Đọc CSV vào DataFrame ────────────────────────────
        self.data = pd.read_csv(csv_path, encoding="utf-8")
        logger.info(f"Đã tải {len(self.data)} mẫu từ: {csv_path}")

        # ── Kiểm tra cột bắt buộc ───────────────────────────
        required_columns = {"image_path", "text", "label"}
        missing = required_columns - set(self.data.columns)
        if missing:
            raise ValueError(
                f"File CSV thiếu cột: {missing}. "
                f"Cột hiện có: {self.data.columns.tolist()}"
            )

        # ── Chuẩn hóa cột label → int ───────────────────────
        # Đảm bảo label luôn là int (0 hoặc 1) dù CSV lưu dạng chữ hay số
        self.data["label"] = self.data["label"].apply(self._parse_label)

        # ── Xóa dòng có label không hợp lệ ──────────────────
        before = len(self.data)
        self.data = self.data[self.data["label"].isin([0, 1])].reset_index(drop=True)
        after = len(self.data)
        if before != after:
            logger.warning(f"Đã loại bỏ {before - after} dòng có label không hợp lệ.")

        # ── Lưu đường dẫn CSV để debug sau ──────────────────
        self.csv_path = csv_path

    def __len__(self) -> int:
        """
        Trả về tổng số mẫu trong dataset.

        Returns:
            int: Số dòng trong file CSV.
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Trả về 1 mẫu dữ liệu theo chỉ số idx.

        Tuân thủ Interface Contract:
            image → torch.Tensor [3, 224, 224], dtype=float32
            text  → str (câu gốc đã clean, CHƯA tokenize)
            label → int (0 = REAL, 1 = FAKE)

        Nếu ảnh lỗi → trả về tensor zeros (không crash DataLoader).

        Args:
            idx (int): Chỉ số của mẫu (0 ≤ idx < len(dataset)).

        Returns:
            tuple: (image_tensor, text_str, label_int)
        """
        row = self.data.iloc[idx]

        # ── Đọc thông tin từ dòng CSV ────────────────────────
        image_path = str(row["image_path"])
        text_raw   = str(row["text"]) if pd.notna(row["text"]) else ""
        label      = int(row["label"])

        # ── Xử lý ảnh ───────────────────────────────────────
        image = preprocess_image(image_path)

        if image is None:
            # Ảnh lỗi hoặc không tồn tại → dùng fallback zeros
            # Ghi log ở level DEBUG để không làm ngập log khi chạy
            logger.debug(f"Dùng fallback cho ảnh lỗi tại idx={idx}: {image_path}")
            image = _FALLBACK_IMAGE.clone()  # Clone để tránh chia sẻ memory

        # ── Xử lý văn bản ───────────────────────────────────
        # Clean text lần nữa phòng trường hợp CSV chưa được clean
        text = clean_text(text_raw)

        # Nếu text rỗng sau khi clean, dùng chuỗi placeholder
        if not text:
            text = "[empty]"
            logger.debug(f"Text rỗng tại idx={idx}, image: {image_path}")

        return image, text, label

    # ──────────────────────────────────────────────────────────
    # Helper methods
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _parse_label(value) -> int:
        """
        Chuyển đổi nhãn từ nhiều định dạng về int (0 hoặc 1).

        Xử lý các trường hợp:
            - Số nguyên: 0, 1
            - Chuỗi số: "0", "1"
            - Chuỗi chữ: "real", "fake", "REAL", "FAKE"
            - Chuỗi không hợp lệ: trả về -1 (sẽ bị lọc bỏ)

        Args:
            value: Giá trị nhãn từ CSV.

        Returns:
            int: 0, 1, hoặc -1 nếu không nhận diện được.
        """
        if pd.isna(value):
            return -1

        value_str = str(value).strip().lower()

        if value_str in LABEL_MAP:
            return LABEL_MAP[value_str]

        # Thử chuyển trực tiếp sang int
        try:
            v = int(float(value_str))
            return v if v in [0, 1] else -1
        except (ValueError, TypeError):
            return -1

    def get_label_distribution(self) -> dict:
        """
        Trả về phân phối nhãn trong dataset (hữu ích để debug).

        Returns:
            dict: {"real": N_real, "fake": N_fake, "total": N_total}
        """
        counts = self.data["label"].value_counts()
        return {
            "real":  int(counts.get(0, 0)),
            "fake":  int(counts.get(1, 0)),
            "total": len(self.data),
        }

    def __repr__(self) -> str:
        dist = self.get_label_distribution()
        return (
            f"FakeNewsDataset(\n"
            f"  file  = '{self.csv_path}'\n"
            f"  total = {dist['total']} mẫu\n"
            f"  real  = {dist['real']}  (label=0)\n"
            f"  fake  = {dist['fake']}  (label=1)\n"
            f")"
        )
