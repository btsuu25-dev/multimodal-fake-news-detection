"""
evaluate.py - Script Đánh Giá Chính (M6)
=========================================
Người phụ trách: Thành viên 6 (M6 – Đánh giá)

Thực hiện đầy đủ 2 cấp độ đánh giá:
  [A] MODEL-LEVEL : Đánh giá từng mô hình riêng lẻ trên test.csv
                    (Text-Only, Image-Only, Concat, CrossModalFND)
  [B] FULL PIPELINE: Đánh giá đầu cuối CrossModalFND kèm biểu đồ chi tiết

Chạy lệnh: python src/evaluate.py

Output sinh ra:
    results/plots/confusion_matrix.png
    results/plots/roc_curve.png
    results/plots/training_history.png
    results/plots/model_comparison.png
    results/reports/final_evaluation.md
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataset import FakeNewsDataset
from src.models.image_encoder import CLIPImageEncoder
from src.models.text_encoder import CLIPTextEncoder
from src.models.baselines import TextOnlyClassifier, ImageOnlyClassifier, ConcatClassifier
from src.models.crossmodal import CrossModalFND
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    plot_model_comparison,
)

# ── Cấu hình logging ───────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Đường dẫn ─────────────────────────────────────────────────
CSV_TEST    = "data/splits/test.csv"
MODEL_PATH  = "saved_models/best_model.pt"
LOG_CSV     = "results/training_log.csv"
PLOTS_DIR   = "results/plots"
REPORTS_DIR = "results/reports"

BATCH_SIZE = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# [A] MODEL-LEVEL EVALUATION
# Đánh giá từng mô hình trên cùng 1 tập test – so sánh công bằng
# ══════════════════════════════════════════════════════════════

def evaluate_baseline_on_test(
    baseline_model: nn.Module,
    mode: str,
    test_loader: DataLoader,
    image_encoder: nn.Module,
    text_encoder: nn.Module,
) -> dict:
    """
    Chạy 1 mô hình Baseline trên tập test và trả về Accuracy + F1.

    Args:
        baseline_model : TextOnlyClassifier / ImageOnlyClassifier / ConcatClassifier
        mode           : 'text' | 'image' | 'concat'
        test_loader    : DataLoader tập test
        image_encoder  : CLIPImageEncoder đã khởi tạo (Frozen)
        text_encoder   : CLIPTextEncoder đã khởi tạo (Frozen)

    Returns:
        dict: {"accuracy": float, "f1": float}
    """
    baseline_model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Trích xuất embedding (dùng encoder đã Frozen)
            img_emb = image_encoder(images)
            tokens  = text_encoder.tokenize(list(texts))
            tokens  = {k: v.to(DEVICE) for k, v in tokens.items()}
            txt_emb = text_encoder(tokens)

            if mode == 'text':
                logits = baseline_model(txt_emb)
            elif mode == 'image':
                logits = baseline_model(img_emb)
            else:  # concat
                combined = torch.cat([img_emb, txt_emb], dim=-1)
                logits   = baseline_model(combined)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average='weighted', zero_division=0),
    }


def run_model_level_evaluation(test_loader: DataLoader) -> dict:
    """
    [A] MODEL-LEVEL: Khởi tạo và đánh giá tất cả mô hình trên test.csv.
    Đây là đánh giá công bằng vì tất cả dùng chung 1 tập test chưa nhìn thấy.

    Returns:
        dict: Kết quả của tất cả mô hình
    """
    print("\n" + "="*60)
    print("  [A] MODEL-LEVEL EVALUATION")
    print("  Danh gia tung mo hinh rieng le tren test.csv")
    print("="*60)

    # Khởi tạo shared encoders (dùng chung cho tất cả baseline)
    print("  Dang khoi tao CLIP Encoders (dung chung)...")
    image_encoder = CLIPImageEncoder().to(DEVICE)
    text_encoder  = CLIPTextEncoder().to(DEVICE)
    image_encoder.eval()
    text_encoder.eval()

    results = {}

    # Danh sách baseline cần đánh giá
    baselines = [
        (TextOnlyClassifier(),  'text',   'Text-Only'),
        (ImageOnlyClassifier(), 'image',  'Image-Only'),
        (ConcatClassifier(),    'concat', 'Concat'),
    ]

    for model_obj, mode, name in baselines:
        model_obj = model_obj.to(DEVICE)

        # Load checkpoint đã train (nếu có), nếu không thì dùng random weights
        ckpt_path = os.path.join("saved_models", f"baseline_{mode}.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model_obj.load_state_dict(state)
            print(f"  Dang danh gia: {name} (da load: {ckpt_path})")
        else:
            print(f"  [CANH BAO] Khong tim thay {ckpt_path}, dung random weights cho {name}")

        metrics = evaluate_baseline_on_test(
            model_obj, mode, test_loader, image_encoder, text_encoder
        )
        results[name] = metrics
        print(f"    Accuracy={metrics['accuracy']*100:.2f}%  F1={metrics['f1']:.4f}")

    return results



# ══════════════════════════════════════════════════════════════
# [B] FULL PIPELINE EVALUATION
# Đánh giá đầu cuối CrossModalFND – từ raw image+text → logits
# ══════════════════════════════════════════════════════════════

def run_full_pipeline_evaluation(test_loader: DataLoader) -> dict:
    """
    [B] FULL PIPELINE: Nạp best_model.pt, chạy toàn bộ pipeline
    (DataLoader → Encoder → Fusion → MLP → prediction) trên test.csv.

    Returns:
        dict: Đầy đủ chỉ số (accuracy, f1, auc_roc, conf_matrix, probs, ...)
    """
    print("\n" + "="*60)
    print("  [B] FULL PIPELINE EVALUATION")
    print("  Chay CrossModalFND end-to-end tren test.csv")
    print("="*60)

    # Nạp mô hình đã train
    model = CrossModalFND().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Da nap: {MODEL_PATH}")

    # Chạy đánh giá đầy đủ
    metrics = evaluate_model(model, test_loader, DEVICE)

    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════
# LƯU BÁO CÁO CUỐI
# ══════════════════════════════════════════════════════════════

def save_final_report(pipeline_metrics: dict, model_level: dict, save_path: str):
    """Xuất báo cáo đánh giá cuối dạng Markdown với đầy đủ 2 cấp độ."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = pipeline_metrics['conf_matrix']
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    # Gộp kết quả model-level + full pipeline để so sánh
    all_results = {
        **model_level,
        "CrossModal-FND": {
            "accuracy": pipeline_metrics['accuracy'],
            "f1":       pipeline_metrics['f1'],
            "auc_roc":  pipeline_metrics['auc_roc'],
        }
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# Bao Cao Danh Gia Cuoi - CrossModalFND\n\n")

        # ── A. Model-level ───────────────────────────────────
        f.write("## [A] Model-Level Evaluation\n\n")
        f.write("Danh gia tung mo hinh rieng le tren cung 1 tap test (1730 mau):\n\n")
        f.write("| Mo Hinh | Accuracy | F1 (weighted) | AUC-ROC |\n")
        f.write("|---|---|---|---|\n")
        for name, vals in all_results.items():
            auc_str = f"{vals.get('auc_roc', '-'):.4f}" if 'auc_roc' in vals else "-"
            marker = " **(Cua nhom)**" if name == "CrossModal-FND" else ""
            f.write(
                f"| {name}{marker} | {vals['accuracy']*100:.2f}% "
                f"| {vals['f1']:.4f} | {auc_str} |\n"
            )

        # ── B. Full Pipeline ─────────────────────────────────
        f.write("\n## [B] Full Pipeline Evaluation (CrossModalFND)\n\n")
        f.write("Ket qua end-to-end: Raw Image + Raw Text → Prediction\n\n")
        f.write(f"| Chi So   | Gia Tri |\n")
        f.write(f"|----------|---------|\n")
        f.write(f"| Accuracy | {pipeline_metrics['accuracy']*100:.2f}% |\n")
        f.write(f"| F1 Score | {pipeline_metrics['f1']:.4f} |\n")
        f.write(f"| AUC-ROC  | {pipeline_metrics['auc_roc']:.4f} |\n\n")

        f.write("### Confusion Matrix\n\n```\n")
        f.write(f"                 Du Doan REAL  Du Doan FAKE\n")
        f.write(f"Thuc Te REAL        {tn:5d}         {fp:5d}\n")
        f.write(f"Thuc Te FAKE        {fn:5d}         {tp:5d}\n```\n\n")

        f.write("## Bieu Do Tham Khao\n\n")
        f.write("- `results/plots/confusion_matrix.png`\n")
        f.write("- `results/plots/roc_curve.png`\n")
        f.write("- `results/plots/training_history.png`\n")
        f.write("- `results/plots/model_comparison.png`\n")

    print(f"  Saved: {save_path}")
    return all_results


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(PLOTS_DIR,   exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    logger.info(f"Device: {DEVICE}")

    # Nạp dữ liệu test 1 lần, dùng chung cho cả 2 cấp độ
    print("\n=== NAP DU LIEU TEST ===")
    test_dataset = FakeNewsDataset(CSV_TEST)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Test set: {len(test_dataset)} mau")

    # ── [A] Model-level ──────────────────────────────────────
    model_level_results = run_model_level_evaluation(test_loader)

    # ── [B] Full Pipeline ────────────────────────────────────
    pipeline_metrics = run_full_pipeline_evaluation(test_loader)

    # ── Lưu báo cáo ──────────────────────────────────────────
    print("\n=== LUU BAO CAO VA VE BIEU DO ===")
    all_results = save_final_report(
        pipeline_metrics,
        model_level_results,
        save_path=os.path.join(REPORTS_DIR, "final_evaluation.md")
    )

    # ── Vẽ biểu đồ ───────────────────────────────────────────
    plot_confusion_matrix(
        pipeline_metrics['conf_matrix'],
        save_path=os.path.join(PLOTS_DIR, "confusion_matrix.png")
    )
    plot_roc_curve(
        pipeline_metrics['all_labels'],
        pipeline_metrics['all_probs'],
        pipeline_metrics['auc_roc'],
        save_path=os.path.join(PLOTS_DIR, "roc_curve.png")
    )
    plot_training_history(
        log_csv_path=LOG_CSV,
        save_path=os.path.join(PLOTS_DIR, "training_history.png")
    )
    plot_model_comparison(
        all_results,
        save_path=os.path.join(PLOTS_DIR, "model_comparison.png")
    )

    # ── Tóm tắt ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HOAN THANH DAY DU 2 CAP DO DANH GIA!")
    print("="*60)
    print("  [A] Model-Level (tren test.csv):")
    for name, vals in model_level_results.items():
        print(f"      {name:15s}: Acc={vals['accuracy']*100:.2f}%  F1={vals['f1']:.4f}")
    print(f"      {'CrossModal-FND':15s}: Acc={pipeline_metrics['accuracy']*100:.2f}%  "
          f"F1={pipeline_metrics['f1']:.4f}  AUC={pipeline_metrics['auc_roc']:.4f}")
    print("  [B] Full Pipeline: OK - Xem bieu do trong results/plots/")
    print(f"  Bao cao: results/reports/final_evaluation.md")
    print("="*60)


if __name__ == "__main__":
    main()
