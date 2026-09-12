import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Import các baseline models do M4 (chúng ta) viết
from src.models.baselines import TextOnlyClassifier, ImageOnlyClassifier, ConcatClassifier

# Chú ý: Các module dưới đây do M1, M2, M3 phụ trách.
# Việc dùng try...except giúp chúng ta viết code ngay bây giờ
# mà không bị crash ứng dụng khi các file kia chưa tồn tại.
try:
    from src.data.dataset import FakeNewsDataset
    from src.models.image_encoder import CLIPImageEncoder
    from src.models.text_encoder import CLIPTextEncoder
    MODULES_READY = True
except ImportError as e:
    print(f"⚠️ Ghi chú: Chờ M1, M2, M3 nộp bài để có thể chạy. (Lỗi thiếu: {e})")
    MODULES_READY = False

# ==========================================
# CẤU HÌNH SIÊU THAM SỐ (HYPERPARAMETERS)
# ==========================================
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results/reports"
CSV_TRAIN_PATH = "data/splits/train.csv"
CSV_VAL_PATH = "data/splits/val.csv"

# ==========================================
# CÁC HÀM XỬ LÝ CHÍNH
# ==========================================

def evaluate(model, loader, image_encoder, text_encoder, mode, device):
    """
    Đánh giá mô hình trên tập validation.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, texts, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Trích xuất đặc trưng từ Image
            img_emb = image_encoder(images)
            
            # Trích xuất đặc trưng từ Text (tuân thủ theo hợp đồng M3)
            tokens = text_encoder.tokenize(texts)
            if isinstance(tokens, dict):
                tokens = {k: v.to(device) for k, v in tokens.items()}
            txt_emb = text_encoder(tokens)
            
            # Chạy qua Baseline Model
            if mode == 'text':
                logits = model(txt_emb)
            elif mode == 'image':
                logits = model(img_emb)
            elif mode == 'concat':
                combined = torch.cat([img_emb, txt_emb], dim=-1)
                logits = model(combined)
            
            # Tính nhãn dự đoán (argmax)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Tính Accuracy và F1 bằng thư viện scikit-learn
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, f1


def train_model(model, model_name, mode, train_loader, val_loader, image_encoder, text_encoder, device):
    """
    Vòng lặp huấn luyện đầy đủ cho một mô hình.
    """
    print(f"\n" + "="*40)
    print(f"🚀 ĐANG HUẤN LUYỆN: {model_name}")
    print("="*40)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        # tqdm giúp tạo thanh progress bar đẹp mắt trên terminal
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS}", leave=False)
        for images, texts, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Tắt tính gradient cho Encoder để tiết kiệm bộ nhớ (Frozen theo yêu cầu)
            with torch.no_grad():
                img_emb = image_encoder(images)
                
                tokens = text_encoder.tokenize(texts)
                if isinstance(tokens, dict):
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                txt_emb = text_encoder(tokens)
            
            # Feed forward
            if mode == 'text':
                logits = model(txt_emb)
            elif mode == 'image':
                logits = model(img_emb)
            elif mode == 'concat':
                combined = torch.cat([img_emb, txt_emb], dim=-1)
                logits = model(combined)
                
            loss = criterion(logits, labels)
            
            # Lan truyền ngược & Cập nhật tham số
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Cuối mỗi epoch, đánh giá mô hình
        val_acc, val_f1 = evaluate(model, val_loader, image_encoder, text_encoder, mode, device)
        avg_loss = total_loss / len(train_loader)
        
        # In kết quả (Tuân thủ rule trong team_assignment.md)
        print(f"[{model_name}] Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.4f}")
        
        # Ghi nhận kết quả tốt nhất
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            
    return best_acc, best_f1


def save_results(results):
    """
    Xuất kết quả ra file markdown theo định dạng bảng yêu cầu.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "baseline_comparison.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 📊 Báo Cáo Kết Quả Baseline Models\n\n")
        f.write("| Mô hình | Accuracy | F1 |\n")
        f.write("|---|---|---|\n")
        
        # Điền số liệu cho từng mô hình
        for model_name, metrics in results.items():
            acc = metrics['acc'] * 100
            f1 = metrics['f1']
            f.write(f"| {model_name} | ~{acc:.1f}% | ~{f1:.4f} |\n")
            
    print(f"\n✅ Đã lưu file báo cáo kết quả: {report_path}")


def main():
    if not MODULES_READY:
        print("❌ Dừng chạy: Cần có code của M1, M2, M3 mới có thể Train thật.")
        return
        
    print("--- NẠP DỮ LIỆU & ENCODER (M1, M2, M3) ---")
    image_encoder = CLIPImageEncoder().to(DEVICE)
    text_encoder = CLIPTextEncoder().to(DEVICE)
    
    # Bật chế độ eval cho 2 encoder để chắc chắn chúng không bị đổi tham số (Frozen)
    image_encoder.eval()
    text_encoder.eval()
    
    # Khởi tạo data loaders
    train_dataset = FakeNewsDataset(CSV_TRAIN_PATH)
    val_dataset = FakeNewsDataset(CSV_VAL_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Danh sách các baseline model cần train
    models_to_train = [
        (TextOnlyClassifier(), "Text-Only", "text"),
        (ImageOnlyClassifier(), "Image-Only", "image"),
        (ConcatClassifier(), "Concat", "concat")
    ]
    
    results = {}
    
    # Bắt đầu duyệt qua và train từng model một
    for model, name, mode in models_to_train:
        acc, f1 = train_model(model, name, mode, train_loader, val_loader, image_encoder, text_encoder, DEVICE)
        results[name] = {'acc': acc, 'f1': f1}
        
    # In báo cáo markdown
    save_results(results)


if __name__ == "__main__":
    main()
