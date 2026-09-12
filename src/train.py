"""
train.py - Script Huấn Luyện Chính CrossModalFND
=================================================
Người phụ trách: Thành viên 5 (M5 – Mô hình chính)

Chạy lệnh:
    python src/train.py

Output:
    saved_models/best_model.pt       ← Mô hình tốt nhất (theo Val F1)
    results/training_log.csv         ← Log Loss & F1 theo từng epoch
"""

import os
import csv
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.data.dataset import FakeNewsDataset
from src.models.crossmodal import CrossModalFND

# ──────────────────────────────────────────────────────────────
# Cấu hình logging
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Siêu tham số (Hyperparameters)
# ──────────────────────────────────────────────────────────────
EPOCHS      = 15
BATCH_SIZE  = 8
LR          = 2e-4          # Learning rate cho AdamW
WEIGHT_DECAY = 1e-4         # L2 regularization
NUM_WORKERS  = 0            # 0 = an toàn trên Windows (tránh lỗi multiprocessing)
PIN_MEMORY   = torch.cuda.is_available()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # Mixed Precision chỉ hoạt động trên GPU

# Đường dẫn dữ liệu và kết quả
CSV_TRAIN   = "data/splits/train.csv"
CSV_VAL     = "data/splits/val.csv"
SAVE_DIR    = "saved_models"
RESULTS_DIR = "results"
MODEL_PATH  = os.path.join(SAVE_DIR, "best_model.pt")
LOG_PATH    = os.path.join(RESULTS_DIR, "training_log.csv")


# ──────────────────────────────────────────────────────────────
# HÀM ĐÁNH GIÁ
# ──────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple:
    """
    Đánh giá mô hình trên tập val/test. Không cập nhật gradient.

    Args:
        model     : CrossModalFND đang đánh giá.
        loader    : DataLoader của tập val hoặc test.
        criterion : Hàm loss (CrossEntropyLoss).

    Returns:
        tuple: (avg_loss, accuracy, f1_score)
    """
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass (không Mixed Precision khi eval để tránh sai số)
            logits = model(images, list(texts))
            loss   = criterion(logits, labels)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


# ──────────────────────────────────────────────────────────────
# HÀM HUẤN LUYỆN MỘT EPOCH
# ──────────────────────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    epoch: int
) -> float:
    """
    Chạy một vòng lặp huấn luyện (1 epoch).

    Args:
        model     : CrossModalFND đang train.
        loader    : DataLoader của tập train.
        optimizer : AdamW optimizer.
        criterion : CrossEntropyLoss.
        scaler    : GradScaler cho Mixed Precision (fp16).
        epoch     : Số epoch hiện tại (dùng để hiển thị).

    Returns:
        float: Loss trung bình trong epoch này.
    """
    model.train()
    # Giữ encoder ở chế độ eval vì chúng đã bị Freeze
    model.image_encoder.eval()
    model.text_encoder.eval()

    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False)

    for images, texts, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # ── Mixed Precision (fp16) – giảm ~50% VRAM ─────────────
        if USE_AMP:
            with autocast():
                logits = model(images, list(texts))
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            # Gradient clipping: tránh gradient quá lớn làm model bất ổn
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU mode – chạy bình thường không có AMP
            logits = model(images, list(texts))
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────
# HÀM MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # ── Tạo thư mục output nếu chưa có ─────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info(f"Device: {DEVICE} | Mixed Precision (AMP): {USE_AMP}")
    logger.info(f"Hyperparameters: EPOCHS={EPOCHS}, BATCH={BATCH_SIZE}, LR={LR}")

    # ── Nạp dữ liệu từ M1 ───────────────────────────────────────
    logger.info("Đang nạp dữ liệu (M1 – FakeNewsDataset)...")
    train_dataset = FakeNewsDataset(CSV_TRAIN)
    val_dataset   = FakeNewsDataset(CSV_VAL)

    logger.info(f"  Train: {len(train_dataset)} mẫu")
    logger.info(f"  Val  : {len(val_dataset)} mẫu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ── Khởi tạo mô hình, optimizer, loss, scaler ───────────────
    logger.info("Đang khởi tạo CrossModalFND (M5)...")
    model = CrossModalFND().to(DEVICE)

    # AdamW chỉ tối ưu các tham số có requires_grad=True
    # (tức là Fusion + MLP, bỏ qua CLIP encoders đã Frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # LR Scheduler: Giảm LR x0.5 nếu Val F1 không cải thiện sau 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=USE_AMP)

    # ── Vòng lặp huấn luyện ─────────────────────────────────────
    best_val_f1  = 0.0
    best_val_acc = 0.0
    log_rows     = []

    print("\n" + "="*60)
    print("  BẮT ĐẦU HUẤN LUYỆN CrossModalFND")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        # Train 1 epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, epoch
        )

        # Đánh giá trên tập val
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

        # Cập nhật learning rate scheduler
        scheduler.step(val_f1)

        # In kết quả epoch
        checkpoint_marker = ""
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            checkpoint_marker = "  ← ✅ Lưu checkpoint!"

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.1f}% | "
            f"Val F1: {val_f1:.4f}"
            f"{checkpoint_marker}"
        )

        # Ghi log vào danh sách để lưu CSV sau
        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(val_f1, 4),
        })

    # ── Lưu log training vào CSV ─────────────────────────────────
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_acc", "val_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    # ── Tóm tắt kết quả ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  KẾT QUẢ HUẤN LUYỆN")
    print("="*60)
    print(f"  Mô hình tốt nhất lưu tại : {MODEL_PATH}")
    print(f"  Log training lưu tại     : {LOG_PATH}")
    print(f"  Val Accuracy tốt nhất    : {best_val_acc*100:.2f}%")
    print(f"  Val F1 tốt nhất          : {best_val_f1:.4f}")
    print("="*60)
    print("\nXONG! Bây giờ M6 có thể chạy đánh giá trên tập test.")


if __name__ == "__main__":
    main()
