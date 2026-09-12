"""
text_encoder.py - Module 3 (M3) Text Encoder
===================================================================
Người phụ trách: Thành viên 3

Module này chứa class CLIPTextEncoder để mã hóa văn bản thành vector
và hàm tính độ tương đồng (Cosine Similarity) giữa ảnh và chữ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
import logging

logger = logging.getLogger(__name__)

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Khởi tạo Text Encoder với pre-trained CLIP.
        
        Args:
            model_name (str): Tên mô hình trên HuggingFace.
        """
        super().__init__()
        
        logger.info(f"Đang tải CLIP Tokenizer và Text Model: {model_name}...")
        
        # 1. Khởi tạo Tokenizer để xử lý raw text
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # 2. Khởi tạo Text Model (dùng bản có Projection để ra đúng 512 chiều)
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name)
        
        # 3. YÊU CẦU BẮT BUỘC: Đóng băng toàn bộ tham số của mô hình CLIP (freeze).
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.embedding_dim = self.model.config.projection_dim
        logger.info(f"Khởi tạo xong Text Encoder. Embedding dim: {self.embedding_dim}")

    def tokenize(self, texts: list[str]):
        """
        Tokenize danh sách các câu văn bản thành tensor để đưa vào mô hình.
        
        Args:
            texts (list[str]): Danh sách các câu văn bản. Ví dụ: ["A dog", "A cat"]
            
        Returns:
            dict: Dictionary chứa các tensor 'input_ids' và 'attention_mask'.
        """
        # Gọi tokenizer từ thư viện transformers
        # padding=True: Đệm (pad) các câu ngắn cho bằng câu dài nhất trong batch
        # truncation=True: Cắt bớt nếu câu dài hơn giới hạn tối đa của CLIP (77 tokens)
        # return_tensors="pt": Trả về tensor của PyTorch
        tokens = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        # Chuyển đổi BatchEncoding sang dict chuẩn
        return dict(tokens)

    def forward(self, tokens) -> torch.Tensor:
        """
        Trích xuất embedding từ tokens đầu vào.
        
        Args:
            tokens (dict): Kết quả trả về từ hàm tokenize(), chứa 'input_ids' và 'attention_mask'
            
        Returns:
            torch.Tensor: Vector embedding văn bản có shape [B, 512], dtype float32
        """
        # Giải nén dictionary tokens thành các tham số truyền vào hàm forward của mô hình
        outputs = self.model(**tokens)
        
        # text_embeds là kết quả sau khi đi qua lớp projection (chiếu về 512 chiều)
        txt_emb = outputs.text_embeds
        
        # Đảm bảo output có dtype là float32 theo đúng Hợp Đồng
        return txt_emb.to(torch.float32)


def compute_similarity(img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
    """
    Tính độ tương đồng Cosine (Cosine Similarity) giữa embedding của ảnh và chữ.
    
    Args:
        img_emb (torch.Tensor): Tensor embedding hình ảnh, shape [B, 512]
        txt_emb (torch.Tensor): Tensor embedding văn bản, shape [B, 512]
        
    Returns:
        torch.Tensor: Tensor độ tương đồng có shape [B, 1], giá trị từ -1 đến 1.
                      Giá trị thấp nghĩa là không liên quan (dấu hiệu của FAKE news).
    """
    # Tính cosine similarity dọc theo chiều đặc trưng (dim=1)
    # Kết quả của F.cosine_similarity có shape [B]
    sim = F.cosine_similarity(img_emb, txt_emb, dim=1)
    
    # Định hình lại (reshape) từ [B] thành [B, 1] cho đúng với yêu cầu của Hợp Đồng
    return sim.unsqueeze(1)
