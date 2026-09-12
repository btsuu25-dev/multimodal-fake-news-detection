"""
 Mã Hóa Văn Bản & Tính Độ Tương Đồng
Phụ trách: Viết bộ phận chuyển đổi văn bản thành vector số và tính độ
tương đồng ảnh-chữ (cosine similarity), dùng chung backbone CLIP với
CLIPImageEncoder (M2) để đảm bảo 2 vector nằm chung không gian embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# nằm chung một không gian embedding.
CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"


class CLIPTextEncoder(nn.Module):
    """
    Bộ mã hóa văn bản dùng CLIP (openai/clip-vit-base-patch32).

    - tokenize(texts): list[str] -> dict tensor token (input_ids, attention_mask)
    - forward(tokens): dict tensor token -> vector [B, 512]
    - Toàn bộ tham số CLIP được freeze (không train lại).
    """

    def __init__(self, checkpoint: str = CLIP_CHECKPOINT, device: str = None):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # lại từ_pretrained ở đây vì transformers sẽ cache checkpoint.)
        self.clip_model = CLIPModel.from_pretrained(checkpoint)
        self.tokenizer = CLIPTokenizer.from_pretrained(checkpoint)

        # Bắt buộc freeze toàn bộ tham số CLIP.
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        self.to(self.device)

    def tokenize(self, texts: list):
        """
        Nhận list câu thô -> trả về tensor token đã pad/truncate sẵn,
        sẵn sàng đưa vào forward().

        Args:
            texts (list[str]): danh sách câu văn bản thô.

        Returns:
            dict[str, torch.Tensor]: {"input_ids": ..., "attention_mask": ...}
        """
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,  # giới hạn context length gốc của CLIP
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        return tokens

    @torch.no_grad()
    def forward(self, tokens: dict):
        """
        Nhận token (output của tokenize) -> trả ra vector đặc trưng văn bản.

        Args:
            tokens (dict[str, torch.Tensor]): output của self.tokenize().

        Returns:
            torch.Tensor: vector đặc trưng, shape [B, 512].
        """
        text_features = self.clip_model.get_text_features(
            input_ids=tokens["input_ids"],
            attention_mask=tokens.get("attention_mask"),
        )
        return text_features

    def get_embedding_dim(self) -> int:
        """Trả về số chiều của vector embedding (đồng bộ với M2)."""
        return 512

    def train(self, mode: bool = True):
        """
        Override để CLIP luôn ở eval() dù model cha (fusion model) gọi
        .train(). Bắt buộc vì CLIP đã bị freeze — nếu không override,
        khi ghép vào model tổng và gọi model.train() lúc training,
        PyTorch sẽ tự cascade .train() xuống submodule này.
        """
        super().train(mode)
        self.clip_model.eval()
        return self


def compute_similarity(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
    """
    Tính Cosine Similarity giữa vector ảnh và vector chữ.

    Đây là "manh mối" chính của hệ thống: giá trị thấp (gần -1) nghĩa là
    ảnh và chữ không liên quan -> nghi ngờ FAKE. Giá trị cao (gần 1)
    nghĩa là ảnh và chữ khớp nhau -> nghiêng về REAL.

    Args:
        img_emb (torch.Tensor): vector ảnh, shape [B, 512].
        txt_emb (torch.Tensor): vector chữ, shape [B, 512].

    Returns:
        torch.Tensor: cosine similarity, shape [B, 1], giá trị trong [-1, 1].
    """
    img_norm = F.normalize(img_emb, p=2, dim=-1)
    txt_norm = F.normalize(txt_emb, p=2, dim=-1)

    similarity = torch.sum(img_norm * txt_norm, dim=-1, keepdim=True)
    return similarity


if __name__ == "__main__":
    # Test nhanh
    encoder = CLIPTextEncoder()
    tokens = encoder.tokenize(["A man running on the beach"])
    output = encoder(tokens)
    print(output.shape)  

    # Test 
    fake_img_emb = torch.randn(1, 512).to(encoder.device)
    sim = compute_similarity(fake_img_emb, output)
    print(sim.shape)  
    print(sim)         
