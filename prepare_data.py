"""
prepare_data.py - Script chuẩn bị dữ liệu hoàn chỉnh
======================================================
Chạy lệnh: python prepare_data.py
"""
import os
import glob
from src.data.preprocessing import split_dataset, clean_text
import pandas as pd
from tqdm import tqdm

# ── Đường dẫn ─────────────────────────────────────────────────
TWEETS_PATH  = 'data/raw/image-verification-corpus/mediaeval2015/devset/tweets.txt'
IMAGES_ROOT  = 'data/raw/image-verification-corpus/mediaeval2015/devset/images'
CSV_OUT      = 'data/processed/dataset.csv'
SPLITS_DIR   = 'data/splits'

# ── BƯỚC 1: Xây dựng bảng tra cứu ảnh (image_id → full path) ─
print("=== BƯỚC 1: Quét toàn bộ ảnh trong thư mục ===")
image_map = {}
for ext in ['jpg', 'jpeg', 'png', 'gif']:
    for path in glob.glob(os.path.join(IMAGES_ROOT, '**', f'*.{ext}'), recursive=True):
        # Key = tên file không có extension (ví dụ: boston_fake_01)
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        image_map[stem] = path

print(f"Tìm thấy: {len(image_map)} ảnh")
if len(image_map) == 0:
    print("LỖI: Không tìm thấy ảnh nào! Kiểm tra lại thư mục images/")
    exit(1)

# In mẫu để kiểm tra
sample_keys = list(image_map.keys())[:3]
for k in sample_keys:
    print(f"  Mẫu: {k} → {image_map[k]}")

# ── BƯỚC 2: Đọc tweets.txt và ghép với ảnh ────────────────────
print("\n=== BƯỚC 2: Đọc tweets.txt và ghép ảnh ===")
try:
    df = pd.read_csv(TWEETS_PATH, sep='\t', encoding='utf-8', dtype=str)
except Exception:
    df = pd.read_csv(TWEETS_PATH, sep='\t', encoding='latin-1', dtype=str)

df.columns = [c.strip().lower().replace(' ', '_').replace('(s)', '') for c in df.columns]
print(f"Đọc được {len(df)} dòng, cột: {df.columns.tolist()}")

# Xác định tên cột image_id và label
img_col   = next((c for c in df.columns if 'image' in c), None)
label_col = next((c for c in df.columns if 'label' in c), None)
text_col  = next((c for c in df.columns if 'text' in c or 'tweet' in c.lower()), None)

print(f"Cột ảnh: '{img_col}' | Cột nhãn: '{label_col}' | Cột text: '{text_col}'")

records = []
skipped = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Ghép ảnh với tweet"):
    # Lấy image_id và chuẩn hóa về lowercase
    raw_img_id = str(row.get(img_col, '')).strip().lower()
    
    # Tìm ảnh khớp trong image_map
    found_path = image_map.get(raw_img_id)
    if not found_path:
        skipped += 1
        continue
    
    # Xử lý nhãn
    raw_label = str(row.get(label_col, '')).strip().lower()
    if 'fake' in raw_label:
        label = 1
    elif 'real' in raw_label:
        label = 0
    else:
        skipped += 1
        continue
    
    # Làm sạch văn bản
    text = clean_text(str(row.get(text_col, '')))
    if not text:
        text = '[no text]'
    
    records.append({
        'image_path': found_path,
        'text':       text,
        'label':      label,
    })

print(f"\nKết quả: {len(records)} mẫu hợp lệ | Bỏ qua: {skipped}")

if len(records) == 0:
    print("LỖI: Không ghép được ảnh nào! Kiểm tra tên cột image_id trong tweets.txt")
    exit(1)

# ── BƯỚC 3: Lưu dataset.csv ───────────────────────────────────
print("\n=== BƯỚC 3: Lưu dataset.csv ===")
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
df_out = pd.DataFrame(records)
df_out.to_csv(CSV_OUT, index=False, encoding='utf-8')
dist = df_out['label'].value_counts()
print(f"Đã lưu: {CSV_OUT}")
print(f"  REAL (0): {dist.get(0, 0)} mẫu")
print(f"  FAKE (1): {dist.get(1, 0)} mẫu")

# ── BƯỚC 4: Chia train/val/test ───────────────────────────────
print("\n=== BƯỚC 4: Chia train/val/test ===")
result = split_dataset(CSV_OUT, SPLITS_DIR)
print('Train:', result['train'], '| Val:', result['val'], '| Test:', result['test'])

print("\n✅ XONG! Dữ liệu sẵn sàng để train.")
print("Chạy tiếp: python src/train.py")
