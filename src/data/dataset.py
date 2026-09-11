from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from src.data.preprocessing import (
    clean_text,
    load_and_preprocess_image,
)


class FakeNewsDataset(Dataset):
    """
    Dataset for multimodal fake news detection.

    Returns:
        image: Tensor [3, 224, 224]
        text: str
        label: int

    Labels:
        0 = REAL
        1 = FAKE
    """

    def __init__(self, csv_path: str):

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}"
            )

        self.data = pd.read_csv(
            self.csv_path
        )

        required_columns = {
            "image_path",
            "text",
            "label",
        }

        missing_columns = (
            required_columns
            - set(self.data.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

        # Remove invalid rows
        self.data = (
            self.data
            .dropna(
                subset=[
                    "image_path",
                    "text",
                    "label",
                ]
            )
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image = load_and_preprocess_image(
            row["image_path"]
        )

        text = clean_text(
            row["text"]
        )

        label = int(
            row["label"]
        )

        return image, text, label