"""
preprocessing.py - Cac ham tien xu ly du lieu cho du an Anti Fake News
==========================================================================
Module nay cung cap:
  - preprocess_image()  : Doc va chuan hoa anh thanh Tensor [3, 224, 224]
  - clean_text()        : Lam sach van ban tweet
  - build_dataset_csv() : Ghep tweets.txt voi anh -> CSV tong hop
  - split_dataset()     : Chia CSV thanh train / val / test

Tac gia: Thanh vien 1 (M1 - Du Lieu)
"""

import os
import re
import logging
import pandas as pd
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------------------------------------------
# Cau hinh logging de theo doi loi trong qua trinh xu ly
# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# Hang so co dinh - theo docs/contracts.md
# ----------------------------------------------------------
IMAGE_SIZE    = 224   # Kich thuoc anh dau vao (224x224 pixel)
EMBEDDING_DIM = 512   # So chieu embedding CLIP ViT-B/32
RANDOM_SEED   = 42    # Seed co dinh de ket qua reproducible

# Gia tri mean/std chuan CLIP ViT-B/32 (khac voi ImageNet thong thuong)
NORMALIZE_MEAN = [0.48145466, 0.4578275,  0.40821073]
NORMALIZE_STD  = [0.26862954, 0.26130258, 0.27577711]

# ----------------------------------------------------------
# Transform pipeline dung torchvision
# ----------------------------------------------------------
_image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize ve 224x224
    transforms.ToTensor(),                        # PIL -> Tensor [3,H,W], float32 trong [0,1]
    transforms.Normalize(                         # Normalize theo chuan CLIP
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD
    ),
])


# ==========================================================
# HAM 1: Tien xu ly anh
# ==========================================================
def preprocess_image(image_path: str):
    """
    Doc va chuan hoa mot anh tu duong dan tren disk.

    Luong xu ly:
        1. Kiem tra file ton tai
        2. Mo anh bang PIL
        3. Chuyen sang RGB (tranh loi RGBA / grayscale / palette)
        4. Ap dung transform: Resize(224x224) -> ToTensor -> Normalize
        5. Tra ve Tensor [3, 224, 224], dtype=float32

    Args:
        image_path (str): Duong dan den file anh.

    Returns:
        torch.Tensor shape [3, 224, 224] dtype float32 neu thanh cong.
        None neu file khong ton tai hoac anh bi loi/corrupt.
    """
    # Buoc 1: Kiem tra file co ton tai khong
    if not os.path.exists(image_path):
        logger.warning(f"Anh khong ton tai: {image_path}")
        return None

    try:
        # Buoc 2: Mo anh bang PIL
        img = Image.open(image_path)

        # Buoc 3: Chuyen sang RGB
        # CLIP chi nhan RGB (3 kenh mau)
        # Mot so anh tweet co the la RGBA (4 kenh) hoac grayscale (1 kenh)
        img = img.convert("RGB")

        # Buoc 4: Ap dung transform pipeline (resize + normalize)
        tensor = _image_transform(img)  # -> [3, 224, 224], float32

        return tensor

    except UnidentifiedImageError:
        logger.warning(f"Anh bi loi (corrupt): {image_path}")
        return None
    except Exception as e:
        logger.warning(f"Khong doc duoc anh {image_path}: {e}")
        return None


# ==========================================================
# HAM 2: Lam sach van ban tweet
# ==========================================================
def clean_text(text: str) -> str:
    """
    Lam sach mot chuoi van ban tweet.

    Cac buoc lam sach:
        1. Xoa URL (http://, https://, www.)
        2. Decode HTML entities (&amp; -> &, &lt; -> <, v.v.)
        3. Xoa ky tu dieu khien (newline, tab, carriage return)
        4. Xoa khoang trang thua (dau/cuoi va giua tu)

    Luu y:
        - GIU NGUYEN chu hoa/thuong (CLIP Tokenizer tu xu ly)
        - KHONG xoa stopwords (CLIP can du ngu canh cau)

    Args:
        text (str): Van ban tweet goc.

    Returns:
        str: Van ban da duoc lam sach.
    """
    if not isinstance(text, str):
        return ""

    # Buoc 1: Xoa URL
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # Buoc 2: Decode HTML entities thong dung
    html_entities = {
        "&amp;":  "&",
        "&lt;":   "<",
        "&gt;":   ">",
        "&quot;": '"',
        "&#39;":  "'",
        "&nbsp;": " ",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)

    # Buoc 3: Xoa ky tu dieu khien (newline, tab, carriage return)
    text = re.sub(r"[\r\n\t]", " ", text)

    # Buoc 4: Xoa khoang trang thua
    text = re.sub(r"\s+", " ", text)

    # Buoc 5: Trim dau va cuoi chuoi
    text = text.strip()

    return text


# ==========================================================
# HAM 3: Ghep tweets.txt + anh -> CSV tong hop
# ==========================================================
def build_dataset_csv(tweets_path: str, images_dir: str, output_path: str) -> int:
    """
    Doc file tweets.txt, map voi anh thuc te, tao CSV tong hop.

    Dataset Image Verification Corpus (MediaEval 2015) - format thuc te:
        - File tweets.txt: tab-separated, 14.483 dong du lieu + 1 dong header
        - Cac cot: tweetId | tweetText | userId | imageId(s) | username | timestamp | label
        - Cot imageId(s): ten file anh KHONG co extension (vi du: sandyA_fake_46)
        - Cot label: chuoi "fake" hoac "real"
        - Thu muc anh: images/ voi file dang sandyA_fake_46.jpg (hoac .png)

    Logic xu ly:
        1. Doc tweets.txt vao DataFrame (tab-separated, encoding latin-1)
        2. Xay dung bang map {ten_anh_stem: duong_dan_day_du} tu thu muc images/
        3. Voi moi dong: tim file anh tuong ung qua image_map
        4. Neu anh ton tai -> them vao danh sach ket qua
        5. Neu khong co anh -> BO QUA (theo yeu cau docs/team_assignment.md)
        6. Chuyen label: 'fake' -> 1, 'real' -> 0
        7. Lam sach text bang clean_text()
        8. Luu CSV voi 3 cot: image_path, text, label

    Args:
        tweets_path (str): Duong dan den file tweets.txt
        images_dir  (str): Duong dan den thu muc chua anh da giai nen (thu muc images/)
        output_path (str): Duong dan luu file CSV output

    Returns:
        int: So dong hop le da ghi vao CSV
    """
    logger.info(f"Doc file: {tweets_path}")

    # -- Buoc 1: Doc tweets.txt ------------------------------------
    # Format thuc te: tab-separated, encoding latin-1
    # (file goc co ky tu tieng Tay Ban Nha va dau dac biet)
    try:
        df_raw = pd.read_csv(
            tweets_path,
            sep="\t",
            encoding="utf-8",     # File goc la UTF-8 (kiem tra thuc te: "utf-8" cho text dung)
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as e:
        logger.error(f"Khong doc duoc tweets.txt: {e}")
        raise

    logger.info(f"Doc duoc {len(df_raw)} dong tu tweets.txt")
    logger.info(f"Cac cot: {df_raw.columns.tolist()}")

    # -- Buoc 2: Chuan hoa ten cot ---------------------------------
    # Ten cot thuc te: tweetId, tweetText, userId, imageId(s), username, timestamp, label
    df_raw.columns = [col.strip().lower().replace("(s)", "s") for col in df_raw.columns]

    # Mapping ten cot thuc te -> ten chuan noi bo
    COLUMN_MAP = {
        "tweetid":   "tweet_id",
        "tweettext": "text",
        "userid":    "user_id",
        "imageids":  "image_id",   # imageId(s) -> imageids -> image_id
        "imageid":   "image_id",
        "username":  "username",
        "timestamp": "timestamp",
        "label":     "label",
    }
    df_raw = df_raw.rename(columns=COLUMN_MAP)

    # Kiem tra cot bat buoc
    for required_col in ["image_id", "text", "label"]:
        if required_col not in df_raw.columns:
            logger.error(f"Thieu cot '{required_col}'. Cot hien co: {df_raw.columns.tolist()}")
            raise ValueError(f"tweets.txt thieu cot: {required_col}")

    logger.info(f"Cot sau chuan hoa: {df_raw.columns.tolist()}")

    # -- Buoc 3: Xay dung bang map anh de tim kiem nhanh -----------
    # Lap bang do {ten_khong_extension: duong_dan_day_du}
    # Hieu qua hon la goi os.path.exists() tung dong (14.000+ lan)
    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
    image_map = {}  # {stem_lowercase: full_path}

    for fname in os.listdir(images_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMG_EXTENSIONS:
            image_map[stem.lower()] = os.path.join(images_dir, fname)

    logger.info(f"Tim thay {len(image_map)} anh trong thu muc: {images_dir}")

    # -- Buoc 4: Duyet tung dong va map anh -------------------------
    records = []
    skipped_no_image  = 0
    skipped_bad_label = 0

    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Dang xu ly tweets"):
        image_id  = str(row.get("image_id", "")).strip()
        label_raw = str(row.get("label",    "")).strip().lower()
        text_raw  = str(row.get("text",     "")).strip()

        # Bo qua dong thieu image_id
        if not image_id or image_id == "nan":
            skipped_no_image += 1
            continue

        # Tim anh trong image_map (khong phan biet hoa thuong)
        image_path = image_map.get(image_id.lower())

        # Khong co anh -> BO QUA (theo yeu cau team_assignment.md)
        if image_path is None:
            skipped_no_image += 1
            continue

        # Chuyen label: 'fake' -> 1, 'real' -> 0
        if "fake" in label_raw:
            label_int = 1
        elif "real" in label_raw:
            label_int = 0
        else:
            skipped_bad_label += 1
            continue

        # Lam sach van ban tweet
        text_clean = clean_text(text_raw)
        if not text_clean:
            text_clean = "[no text]"

        records.append({
            "image_path": image_path,
            "text":       text_clean,
            "label":      label_int,
        })

    # -- Buoc 5: Thong ke va luu CSV --------------------------------
    logger.info(f"Ket qua xu ly:")
    logger.info(f"  Hop le        : {len(records)} dong")
    logger.info(f"  Bo (no image) : {skipped_no_image} dong")
    logger.info(f"  Bo (bad label): {skipped_bad_label} dong")

    df_output = pd.DataFrame(records, columns=["image_path", "text", "label"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_output.to_csv(output_path, index=False, encoding="utf-8")

    # In phan phoi nhan de kiem tra can bang
    label_dist = df_output["label"].value_counts()
    logger.info(f"Phan phoi nhan:")
    logger.info(f"  REAL (0): {label_dist.get(0, 0)} mau")
    logger.info(f"  FAKE (1): {label_dist.get(1, 0)} mau")
    logger.info(f"Da luu CSV: {output_path}")

    return len(df_output)


# ==========================================================
# HAM 4: Chia train / val / test
# ==========================================================
def split_dataset(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = RANDOM_SEED,
) -> dict:
    """
    Chia dataset.csv thanh 3 tap train / val / test theo ty le co dinh.

    Ky thuat:
        - Dung stratified split de giu ty le FAKE/REAL deu trong ca 3 tap
        - random_state=seed dam bao ket qua co the tai lap (reproducible)

    Ty le mac dinh: 70% train / 15% val / 15% test

    Args:
        csv_path   (str)  : Duong dan den dataset.csv da tao
        output_dir (str)  : Thu muc de luu 3 file CSV dau ra
        train_ratio(float): Ty le train (mac dinh 0.70)
        val_ratio  (float): Ty le val   (mac dinh 0.15)
        seed       (int)  : Random seed  (mac dinh 42)

    Returns:
        dict: {"train": N_train, "val": N_val, "test": N_test}
    """
    logger.info(f"Doc dataset tu: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Tong so mau: {len(df)}")

    test_ratio = 1.0 - train_ratio - val_ratio

    # -- Buoc 1: Tach phan train (70%) ra truoc ---------------------
    df_train, df_temp = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df["label"],
        random_state=seed,
    )

    # -- Buoc 2: Chia phan con lai thanh val / test (15%/15%) -------
    val_size_relative = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_size_relative),
        stratify=df_temp["label"],
        random_state=seed,
    )

    # -- Buoc 3: Luu 3 file CSV -------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path   = os.path.join(output_dir, "val.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    df_train.to_csv(train_path, index=False, encoding="utf-8")
    df_val.to_csv(val_path,     index=False, encoding="utf-8")
    df_test.to_csv(test_path,   index=False, encoding="utf-8")

    logger.info(f"train.csv : {len(df_train)} mau -> {train_path}")
    logger.info(f"val.csv   : {len(df_val)}   mau -> {val_path}")
    logger.info(f"test.csv  : {len(df_test)}  mau -> {test_path}")

    return {
        "train": len(df_train),
        "val":   len(df_val),
        "test":  len(df_test),
    }
