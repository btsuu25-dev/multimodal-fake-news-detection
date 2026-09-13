"""
metrics.py - Module 6 (M6): Tính toán các chỉ số đánh giá mô hình
===================================================================
Người phụ trách: Thành viên 6 (M6 – Đánh giá)

Cung cấp hàm tính đầy đủ: Accuracy, F1, AUC-ROC và confusion matrix.

Cách dùng:
    from src.evaluation.metrics import evaluate_model
    results = evaluate_model(model, test_loader, device)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> dict:
    """
    Chạy mô hình trên toàn bộ tập dữ liệu và trả về đầy đủ chỉ số đánh giá.

    Args:
        model  (nn.Module)    : Mô hình đã được nạp trọng số (load state_dict).
        loader (DataLoader)   : DataLoader của tập test.
        device (torch.device) : Thiết bị tính toán (cpu hoặc cuda).

    Returns:
        dict: Bao gồm các chỉ số sau:
            - accuracy   (float): Tỷ lệ dự đoán đúng
            - f1         (float): F1-Score (weighted)
            - auc_roc    (float): Diện tích dưới đường cong ROC
            - conf_matrix (ndarray): Ma trận nhầm lẫn 2x2
            - all_labels  (list): Nhãn thật
            - all_preds   (list): Nhãn dự đoán
            - all_probs   (list): Xác suất dự đoán là FAKE (dùng cho ROC curve)
    """
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []   # Xác suất class FAKE (index 1) cho AUC-ROC

    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images, list(texts))          # [B, 2]
            probs  = torch.softmax(logits, dim=1)        # [B, 2] → xác suất
            preds  = torch.argmax(logits, dim=1)         # [B]

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Xác suất class FAKE

    # ── Tính các chỉ số ────────────────────────────────────────
    accuracy  = accuracy_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    auc_roc   = roc_auc_score(all_labels, all_probs)
    conf_mat  = confusion_matrix(all_labels, all_preds)

    logger.info(f"Accuracy : {accuracy*100:.2f}%")
    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"AUC-ROC  : {auc_roc:.4f}")

    return {
        "accuracy":    accuracy,
        "f1":          f1,
        "auc_roc":     auc_roc,
        "conf_matrix": conf_mat,
        "all_labels":  all_labels,
        "all_preds":   all_preds,
        "all_probs":   all_probs,
    }
