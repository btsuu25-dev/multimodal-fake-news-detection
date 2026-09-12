"""
fusion.py - Module Cross-Attention Fusion
=========================================
Người phụ trách: Thành viên 5 (M5 – Mô hình chính)

Module này xây dựng bộ phận "trộn" thông tin giữa ảnh và chữ theo cơ chế
Cross-Attention hai chiều:
  - Chiều 1 (T→I): Văn bản hỏi Ảnh  → "chữ đang hỏi về chi tiết gì trong ảnh?"
  - Chiều 2 (I→T): Ảnh hỏi Văn bản  → "ảnh đang minh họa điều gì trong chữ?"

Nếu hai chiều cho ra kết quả mâu thuẫn nhau → dấu hiệu mạnh của FAKE news.

Interface Contract (docs/contracts.md):
    fusion = CrossAttentionFusion()
    fused  = fusion(img_emb, txt_emb)
    # img_emb shape → [B, 512]
    # txt_emb shape → [B, 512]
    # fused   shape → [B, 1024]
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Bộ phận trộn thông tin ảnh-chữ bằng Cross-Attention hai chiều.

    Cơ chế hoạt động:
        1. Mỗi embedding được "mở rộng" thành dạng chuỗi [B, 1, 512]
           để tương thích với MultiheadAttention.
        2. Thực hiện attention hai chiều:
           - T→I: text làm Query,  image làm Key & Value
           - I→T: image làm Query, text làm Key & Value
        3. Áp dụng LayerNorm để ổn định giá trị.
        4. Ghép (concatenate) hai output: [B, 512] + [B, 512] → [B, 1024]

    Args:
        embed_dim (int): Số chiều embedding đầu vào/ra. Mặc định 512 (CLIP ViT-B/32).
        num_heads (int): Số attention heads. Mặc định 8.
        dropout (float): Tỷ lệ Dropout trong attention layer. Mặc định 0.1.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        # ── Chiều 1: Text (Q) → Image (K, V) ─────────────────
        # "Văn bản hỏi: Trong ảnh có chi tiết nào liên quan đến tôi không?"
        self.txt_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True   # Input format: [B, seq_len, embed_dim]
        )

        # ── Chiều 2: Image (Q) → Text (K, V) ─────────────────
        # "Ảnh hỏi: Văn bản có từ nào mô tả tôi không?"
        self.img_to_txt_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # ── LayerNorm để ổn định sau mỗi lần attention ────────
        self.norm_t2i = nn.LayerNorm(embed_dim)
        self.norm_i2t = nn.LayerNorm(embed_dim)

        logger.info(
            f"CrossAttentionFusion khởi tạo: embed_dim={embed_dim}, "
            f"num_heads={num_heads}, output_dim={embed_dim * 2}"
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện cross-attention hai chiều giữa embedding ảnh và văn bản.

        Args:
            img_emb (torch.Tensor): Embedding hình ảnh, shape [B, 512]
            txt_emb (torch.Tensor): Embedding văn bản,  shape [B, 512]

        Returns:
            torch.Tensor: Tensor đã trộn, shape [B, 1024]
                          = concat(cross_T→I [B,512], cross_I→T [B,512])
        """
        # MultiheadAttention yêu cầu input dạng [B, seq_len, embed_dim]
        # Vì mỗi ảnh/chữ là 1 vector đơn, ta thêm chiều seq_len = 1
        img_seq = img_emb.unsqueeze(1)   # [B, 1, 512]
        txt_seq = txt_emb.unsqueeze(1)   # [B, 1, 512]

        # ── Chiều 1: Text hỏi Image ────────────────────────
        # Query = txt, Key = img, Value = img
        cross_t2i, _ = self.txt_to_img_attn(
            query=txt_seq,
            key=img_seq,
            value=img_seq
        )
        # Bỏ chiều seq_len, áp LayerNorm với residual connection
        cross_t2i = self.norm_t2i(cross_t2i.squeeze(1) + txt_emb)   # [B, 512]

        # ── Chiều 2: Image hỏi Text ────────────────────────
        # Query = img, Key = txt, Value = txt
        cross_i2t, _ = self.img_to_txt_attn(
            query=img_seq,
            key=txt_seq,
            value=txt_seq
        )
        cross_i2t = self.norm_i2t(cross_i2t.squeeze(1) + img_emb)   # [B, 512]

        # ── Ghép hai chiều lại ─────────────────────────────
        fused = torch.cat([cross_t2i, cross_i2t], dim=-1)            # [B, 1024]
        return fused
