import re
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def clean_text(text: str) -> str:
    """Clean tweet text."""

    if not isinstance(text, str):
        return ""

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    # Xử lý HTML entities phổ biến
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")

    return text


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """
    Load image and convert to:
    [3, 224, 224], float32
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image = IMAGE_TRANSFORM(image)

    return image