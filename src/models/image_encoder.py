"""
image_encoder.py - Module 2 (M2) Image Encoder
===================================================================
Người phụ trách: Thành viên 2

Module này chứa class CLIPImageEncoder để trích xuất đặc trưng hình ảnh
thành các vector embedding 512 chiều sử dụng pre-trained CLIP.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection
import logging

logger = logging.getLogger(__name__)

class CLIPImageEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Khởi tạo Image Encoder với pre-trained CLIP.
        
        Args:
            model_name (str): Tên mô hình trên HuggingFace.
                              Mặc định: 'openai/clip-vit-base-patch32'
        """
        super().__init__()
        
        logger.info(f"Đang tải mô hình CLIP Image Encoder: {model_name}...")
        
        # Sử dụng CLIPVisionModelWithProjection để có trực tiếp lớp chiếu (projection layer) 
        # giúp output ra đúng số chiều của không gian chung CLIP (512 chiều cho ViT-B/32).
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        
        # YÊU CẦU QUAN TRỌNG: Đóng băng toàn bộ tham số của mô hình CLIP (freeze).
        # Điều này đảm bảo khi huấn luyện, phần pre-trained CLIP không bị cập nhật gradient.
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Xác định số chiều trả về (theo chuẩn ViT-B/32 là 512)
        self.embedding_dim = self.model.config.projection_dim
        
        logger.info(f"Khởi tạo xong. Embedding dim: {self.embedding_dim}")

    def get_embedding_dim(self) -> int:
        """
        Hàm trả về số chiều embedding theo yêu cầu của Hợp Đồng (Contract).
        
        Returns:
            int: Số chiều (ví dụ 512)
        """
        return self.embedding_dim

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Trích xuất embedding từ ảnh đầu vào.
        
        Args:
            image_tensor (torch.Tensor): Tensor hình ảnh với shape [B, 3, 224, 224]
            
        Returns:
            torch.Tensor: Vector embedding [B, 512], dtype float32
        """
        # Đưa tensor vào CLIP model thông qua tham số pixel_values
        outputs = self.model(pixel_values=image_tensor)
        
        # Lấy image_embeds (kết quả sau khi đi qua projection layer của CLIP)
        img_emb = outputs.image_embeds
        
        # Đảm bảo output có dtype là float32 theo đúng Hợp Đồng
        return img_emb.to(torch.float32)
