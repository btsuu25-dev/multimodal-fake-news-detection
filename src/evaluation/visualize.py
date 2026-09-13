"""
visualize.py - Module 6 (M6): Vẽ biểu đồ đánh giá
====================================================
Người phụ trách: Thành viên 6 (M6 – Đánh giá)

Cung cấp 4 hàm vẽ biểu đồ chuyên nghiệp:
    1. plot_confusion_matrix()   - Ma trận nhầm lẫn
    2. plot_roc_curve()          - Đường cong ROC / AUC
    3. plot_training_history()   - Biểu đồ Loss/F1 theo epoch
    4. plot_model_comparison()   - So sánh các mô hình
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Không cần màn hình đồ họa (chạy headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve
import logging

logger = logging.getLogger(__name__)

# ── Bảng màu nhất quán toàn module ─────────────────────────────
_COLOR_FAKE  = '#E74C3C'   # Đỏ  – FAKE
_COLOR_REAL  = '#2ECC71'   # Xanh lá – REAL
_COLOR_MODEL = '#3498DB'   # Xanh dương – Đường mô hình
_COLOR_DIAG  = '#E74C3C'   # Đường chéo ngẫu nhiên
_BG_COLOR    = '#F8F9FA'


def _save(fig, save_path: str):
    """Lưu và đóng figure."""
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG_COLOR)
    plt.close(fig)
    logger.info(f"Da luu bieu do: {save_path}")
    print(f"  Saved: {save_path}")


# ── 1. Confusion Matrix ─────────────────────────────────────────
def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: str):
    """
    Vẽ ma trận nhầm lẫn dạng heatmap.

    Args:
        conf_matrix (ndarray): Ma trận 2x2 từ sklearn.metrics.confusion_matrix
        save_path   (str)    : Đường dẫn lưu file PNG
    """
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # Vẽ ô màu
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Nhãn trục
    classes = ['REAL (0)', 'FAKE (1)']
    ax.set_xticks([0, 1]);  ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(classes, fontsize=11, rotation=90, va='center')
    ax.set_xlabel('Nhãn Dự Đoán', fontsize=12, labelpad=10)
    ax.set_ylabel('Nhãn Thực Tế', fontsize=12, labelpad=10)
    ax.set_title('Confusion Matrix – CrossModalFND', fontsize=14, fontweight='bold', pad=15)

    # Điền số vào từng ô
    total = conf_matrix.sum()
    thresh = conf_matrix.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = conf_matrix[i, j]
            pct = 100 * val / total
            ax.text(j, i, f'{val}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold',
                    color='white' if val > thresh else 'black')

    plt.tight_layout()
    _save(fig, save_path)


# ── 2. ROC Curve ────────────────────────────────────────────────
def plot_roc_curve(all_labels: list, all_probs: list, auc_roc: float, save_path: str):
    """
    Vẽ đường cong ROC với diện tích AUC.

    Args:
        all_labels (list) : Nhãn thật (0/1)
        all_probs  (list) : Xác suất dự đoán FAKE từ softmax
        auc_roc    (float): Giá trị AUC đã tính sẵn
        save_path  (str)  : Đường dẫn lưu file PNG
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    # Đường ROC của mô hình
    ax.plot(fpr, tpr, color=_COLOR_MODEL, lw=2.5,
            label=f'CrossModalFND (AUC = {auc_roc:.4f})')
    # Đường ngẫu nhiên (random baseline)
    ax.plot([0, 1], [0, 1], color=_COLOR_DIAG, lw=1.5,
            linestyle='--', label='Random Classifier (AUC = 0.5)')
    # Tô vùng dưới đường ROC
    ax.fill_between(fpr, tpr, alpha=0.1, color=_COLOR_MODEL)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12, labelpad=8)
    ax.set_ylabel('True Positive Rate',  fontsize=12, labelpad=8)
    ax.set_title('ROC Curve – CrossModalFND', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# ── 3. Training History ─────────────────────────────────────────
def plot_training_history(log_csv_path: str, save_path: str):
    """
    Vẽ biểu đồ Loss và F1 theo từng epoch từ file training_log.csv.

    Args:
        log_csv_path (str): Đường dẫn đến results/training_log.csv
        save_path    (str): Đường dẫn lưu file PNG
    """
    df = pd.read_csv(log_csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG_COLOR)
    fig.suptitle('Quá Trình Huấn Luyện CrossModalFND', fontsize=15, fontweight='bold', y=1.02)

    epochs = df['epoch']

    # Biểu đồ Loss (trái)
    ax1.set_facecolor(_BG_COLOR)
    ax1.plot(epochs, df['train_loss'], color=_COLOR_MODEL, lw=2, marker='o',
             markersize=4, label='Train Loss')
    ax1.plot(epochs, df['val_loss'],   color=_COLOR_FAKE,  lw=2, marker='s',
             markersize=4, label='Val Loss',  linestyle='--')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss',  fontsize=11)
    ax1.set_title('Loss theo Epoch', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Biểu đồ F1 (phải)
    ax2.set_facecolor(_BG_COLOR)
    ax2.plot(epochs, df['val_f1'],  color=_COLOR_REAL,  lw=2.5, marker='D',
             markersize=5, label='Val F1')
    ax2.plot(epochs, df['val_acc'], color=_COLOR_MODEL, lw=2, marker='o',
             markersize=4, label='Val Accuracy', linestyle='--')
    # Đánh dấu điểm tốt nhất
    best_idx = df['val_f1'].idxmax()
    ax2.scatter(df.loc[best_idx, 'epoch'], df.loc[best_idx, 'val_f1'],
                color='gold', s=150, zorder=5, edgecolors='black',
                label=f"Best F1={df.loc[best_idx,'val_f1']:.4f} (Epoch {int(df.loc[best_idx,'epoch'])})")
    ax2.set_xlabel('Epoch',    fontsize=11)
    ax2.set_ylabel('Score',    fontsize=11)
    ax2.set_ylim([0.85, 1.01])
    ax2.set_title('F1 & Accuracy theo Epoch', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)


# ── 4. Model Comparison ─────────────────────────────────────────
def plot_model_comparison(comparison_dict: dict, save_path: str):
    """
    Vẽ biểu đồ cột so sánh các mô hình.

    Args:
        comparison_dict (dict): Format:
            {
              "Text-Only":     {"accuracy": 0.682, "f1": 0.761},
              "Image-Only":    {"accuracy": 0.999, "f1": 0.999},
              "Concat":        {"accuracy": 0.998, "f1": 0.998},
              "CrossModal-FND":{"accuracy": 0.998, "f1": 0.997},
            }
        save_path (str): Đường dẫn lưu file PNG
    """
    models     = list(comparison_dict.keys())
    accuracies = [comparison_dict[m]['accuracy'] * 100 for m in models]
    f1_scores  = [comparison_dict[m]['f1'] for m in models]

    x     = np.arange(len(models))
    width = 0.35
    # Highlight mô hình chính bằng màu khác
    bar_colors_acc = [_COLOR_FAKE if m == 'CrossModal-FND' else _COLOR_MODEL for m in models]
    bar_colors_f1  = [_COLOR_FAKE if m == 'CrossModal-FND' else _COLOR_REAL  for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=_BG_COLOR)
    fig.suptitle('So Sánh Hiệu Suất Các Mô Hình', fontsize=15, fontweight='bold', y=1.02)

    def _bar_chart(ax, values, colors, ylabel, title, ylim):
        ax.set_facecolor(_BG_COLOR)
        bars = ax.bar(x, values, width=0.5, color=colors, edgecolor='white', linewidth=1.2)
        # Điền giá trị lên đỉnh mỗi cột
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, rotation=15, ha='right')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title,   fontsize=12, fontweight='bold')
        ax.set_ylim(ylim)
        ax.grid(axis='y', alpha=0.3)
        # Legend
        patch_ours = mpatches.Patch(color=_COLOR_FAKE,  label='CrossModal-FND (Của nhóm)')
        patch_base = mpatches.Patch(color=colors[0], label='Baseline')
        ax.legend(handles=[patch_ours, patch_base], fontsize=9, loc='lower right')

    _bar_chart(ax1, accuracies, bar_colors_acc, 'Accuracy (%)', 'Accuracy (%)', [60, 103])
    _bar_chart(ax2, f1_scores,  bar_colors_f1,  'F1 Score',     'F1 Score',     [0.6, 1.05])

    plt.tight_layout()
    _save(fig, save_path)
