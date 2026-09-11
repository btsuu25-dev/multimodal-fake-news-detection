import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_URL = "https://github.com/MKLab-ITI/image-verification-corpus.git"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "image-verification-corpus"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SPLITS_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


def download_dataset():
    """Download Image Verification Corpus if it does not exist."""

    if DATASET_DIR.exists():
        print(f"Dataset already exists: {DATASET_DIR}")
        return

    print("Downloading Image Verification Corpus...")

    subprocess.run(
        [
            "git",
            "clone",
            REPO_URL,
            str(DATASET_DIR),
        ],
        check=True,
    )

    print("Download completed.")


def find_image(image_root: Path, image_id: str):
    """
    Find image by image ID.

    Supports jpg/jpeg/png/gif/bmp/webp.
    """

    image_id = str(image_id).strip()

    if not image_id:
        return None

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
    ]

    # Tìm trực tiếp
    for ext in extensions:
        candidate = image_root / f"{image_id}{ext}"

        if candidate.exists():
            return candidate

    # Tìm recursive
    for path in image_root.rglob("*"):
        if path.is_file() and path.stem == image_id:
            return path

    return None


def build_dataset_csv():
    """
    Build:

    data/processed/dataset.csv

    Columns:
        image_path
        text
        label

    Label:
        REAL = 0
        FAKE = 1
    """

    tweets_file = (
        DATASET_DIR
        / "mediaeval2015"
        / "devset"
        / "tweets.txt"
    )

    image_root = (
        DATASET_DIR
        / "mediaeval2015"
        / "devset"
        / "MediaEval2015_DevSet_Images"
    )

    if not tweets_file.exists():
        raise FileNotFoundError(
            f"Cannot find tweets.txt: {tweets_file}"
        )

    if not image_root.exists():
        raise FileNotFoundError(
            f"Cannot find image folder: {image_root}"
        )

    print(f"Reading: {tweets_file}")
    print(f"Images: {image_root}")

    # MediaEval tweets.txt dùng tab
    df = pd.read_csv(
        tweets_file,
        sep="\t",
        dtype=str,
    )

    print(f"Total tweets: {len(df)}")

    records = []

    for _, row in df.iterrows():

        tweet_text = row.get("tweetText", "")
        image_ids = row.get("imageId(s)", "")
        label = str(row.get("label", "")).strip().lower()

        # Chuyển label
        if label == "real":
            numeric_label = 0
        elif label == "fake":
            numeric_label = 1
        else:
            continue

        # Có thể có nhiều image ID
        image_id_list = str(image_ids).split(",")

        image_path = None

        for image_id in image_id_list:

            image_path = find_image(
                image_root,
                image_id,
            )

            if image_path is not None:
                break

        # Nếu không tìm thấy ảnh thì bỏ sample
        if image_path is None:
            continue

        records.append({
            "image_path": str(image_path),
            "text": tweet_text,
            "label": numeric_label,
        })

    result = pd.DataFrame(records)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        PROCESSED_DIR
        / "dataset.csv"
    )

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8",
    )

    print("\n===== DATASET CREATED =====")
    print(f"Output: {output_file}")
    print(f"Samples: {len(result)}")

    print("\nLabel distribution:")
    print(result["label"].value_counts())

    return result


def split_dataset():
    """
    Split dataset:

    Train = 70%
    Validation = 15%
    Test = 15%
    """

    input_file = (
        PROCESSED_DIR
        / "dataset.csv"
    )

    df = pd.read_csv(input_file)

    # 70% train
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"],
    )

    # 15% validation
    # 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    SPLITS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        SPLITS_DIR / "train.csv",
        index=False,
        encoding="utf-8",
    )

    val_df.to_csv(
        SPLITS_DIR / "val.csv",
        index=False,
        encoding="utf-8",
    )

    test_df.to_csv(
        SPLITS_DIR / "test.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n===== DATA SPLIT =====")

    print(
        f"Train: {len(train_df)} "
        f"({len(train_df) / len(df):.2%})"
    )

    print(
        f"Validation: {len(val_df)} "
        f"({len(val_df) / len(df):.2%})"
    )

    print(
        f"Test: {len(test_df)} "
        f"({len(test_df) / len(df):.2%})"
    )


if __name__ == "__main__":

    download_dataset()

    build_dataset_csv()

    split_dataset()