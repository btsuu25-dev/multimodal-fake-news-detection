"""
crossmodal.py - Mô Hình Chính CrossModalFND
============================================
Người phụ trách: Thành viên 5 (M5 – Mô hình chính)

Lắp ráp toàn bộ hệ thống từ các module của M1, M2, M3:

    [Ảnh]  → CLIPImageEncoder (M2) → img_emb [B, 512]  ─────────────────────────┐
                                                          ├─► CrossAttentionFusion ─┤
    [Chữ]  → CLIPTextEncoder  (M3) → txt_emb [B, 512]  ─┤     fused [B, 1024]    │
                                    → sim_score [B, 1]  ─┘                         │
                                                                                    ▼
                                           combined [B, 2049] → MLP → logits [B, 2]

Interface Contract (docs/contracts.md – KHÔNG ĐỔI):
    model  = CrossModalFND()
    logits = model(image_tensor, text_list)
    # image_tensor → torch.Tensor, shape [B, 3, 224, 224]
    # text_list    → list[str], độ dài B
    # logits       → torch.Tensor, shape [B, 2]
"""

import torch
import torch.nn as nn
import logging

from src.models.image_encoder import CLIPImageEncoder
from src.models.text_encoder import CLIPTextEncoder, compute_similarity
from src.models.fusion import CrossAttentionFusion

logger = logging.getLogger(__name__)

# ── Kích thước vector từng phần (theo contracts.md) ───────────
_IMG_DIM   = 512    # img_emb
_TXT_DIM   = 512    # txt_emb
_FUSED_DIM = 1024   # cross_t2i [512] + cross_i2t [512]
_SIM_DIM   = 1      # cosine similarity score
_COMBINED_DIM = _IMG_DIM + _TXT_DIM + _FUSED_DIM + _SIM_DIM  # = 2049


class CrossModalFND(nn.Module):
    """
    Mô hình phát hiện tin giả đa phương thức (CrossModal Fake News Detector).

    Quy trình xử lý (forward pass):
        1. img_emb   ← CLIPImageEncoder(image_tensor)          [B, 512]
        2. txt_emb   ← CLIPTextEncoder(text_list)              [B, 512]
        3. sim_score ← compute_similarity(img_emb, txt_emb)    [B, 1]
        4. fused     ← CrossAttentionFusion(img_emb, txt_emb)  [B, 1024]
        5. combined  ← cat([img_emb, txt_emb, fused, sim])     [B, 2049]
        6. logits    ← MLP(combined)                           [B, 2]

    Tham số có thể học (trainable parameters):
        - CrossAttentionFusion (attention heads + LayerNorm)
        - MLP Classifier (Linear layers)
        ✗ CLIPImageEncoder (FROZEN – không thay đổi)
        ✗ CLIPTextEncoder  (FROZEN – không thay đổi)
    """

    def __init__(self, num_heads: int = 8, dropout: float = 0.3):
        """
        Khởi tạo CrossModalFND.

        Args:
            num_heads (int): Số attention heads trong CrossAttentionFusion. Mặc định 8.
            dropout   (float): Tỷ lệ Dropout trong MLP classifier. Mặc định 0.3.
        """
        super().__init__()

        # ── Bộ mã hóa (Encoders) – Frozen, do M2 và M3 cung cấp ──
        self.image_encoder = CLIPImageEncoder()   # M2
        self.text_encoder  = CLIPTextEncoder()    # M3

        # Đảm bảo encoder luôn ở chế độ eval khi inference
        self.image_encoder.eval()
        self.text_encoder.eval()

        # ── Bộ trộn Cross-Attention – Trainable ───────────────────
        self.fusion = CrossAttentionFusion(
            embed_dim=_IMG_DIM,
            num_heads=num_heads
        )

        # ── Bộ phân loại MLP – Trainable ──────────────────────────
        # Input: combined [B, 2049] → Output: logits [B, 2]
        self.classifier = nn.Sequential(
            nn.Linear(_COMBINED_DIM, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        # In thống kê số tham số khi khởi tạo
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            f"CrossModalFND khởi tạo xong.\n"
            f"  Tổng tham số : {total:,}\n"
            f"  Có thể học   : {trainable:,}  ({100*trainable/total:.1f}%)\n"
            f"  Frozen (CLIP): {total - trainable:,}"
        )

    def forward(self, image_tensor: torch.Tensor, text_list: list) -> torch.Tensor:
        """
        Chạy forward pass cho toàn bộ mô hình.

        Args:
            image_tensor (torch.Tensor): Tensor ảnh đã resize, shape [B, 3, 224, 224].
                                          Thường đến từ FakeNewsDataset (M1).
            text_list    (list[str])   : Danh sách câu văn bản thô, độ dài B.
                                          Thường đến từ FakeNewsDataset (M1).

        Returns:
            torch.Tensor: Logits dự đoán, shape [B, 2].
                          logits[:, 0] = điểm cho REAL
                          logits[:, 1] = điểm cho FAKE
        """
        device = image_tensor.device

        # ── Bước 1: Trích xuất embedding ảnh (M2) ─────────────
        # Encoder đã frozen → dùng no_grad để tiết kiệm bộ nhớ
        with torch.no_grad():
            img_emb = self.image_encoder(image_tensor)       # [B, 512]

        # ── Bước 2: Trích xuất embedding văn bản (M3) ─────────
        with torch.no_grad():
            tokens = self.text_encoder.tokenize(text_list)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            txt_emb = self.text_encoder(tokens)              # [B, 512]

        # ── Bước 3: Tính điểm tương đồng ảnh-chữ ─────────────
        # Giá trị thấp → ảnh và chữ mâu thuẫn → dấu hiệu FAKE
        sim_score = compute_similarity(img_emb, txt_emb)    # [B, 1]

        # ── Bước 4: Cross-Attention Fusion ─────────────────────
        fused = self.fusion(img_emb, txt_emb)               # [B, 1024]

        # ── Bước 5: Ghép toàn bộ đặc trưng ────────────────────
        combined = torch.cat(
            [img_emb, txt_emb, fused, sim_score], dim=-1
        )                                                    # [B, 2049]

        # ── Bước 6: Phân loại qua MLP ──────────────────────────
        logits = self.classifier(combined)                   # [B, 2]
        return logits
