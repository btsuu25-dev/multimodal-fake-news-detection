import torch
import torch.nn as nn

class TextOnlyClassifier(nn.Module):
    """
    Baseline model 1: Text-Only Classifier
    Nhận input là text embedding từ CLIP, phân loại thành 2 class (REAL/FAKE)
    """
    def __init__(self):
        super(TextOnlyClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, txt_emb):
        # txt_emb shape: [B, 512]
        logits = self.net(txt_emb)
        # logits shape: [B, 2]
        return logits

class ImageOnlyClassifier(nn.Module):
    """
    Baseline model 2: Image-Only Classifier
    Nhận input là image embedding từ CLIP, phân loại thành 2 class (REAL/FAKE)
    """
    def __init__(self):
        super(ImageOnlyClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, img_emb):
        # img_emb shape: [B, 512]
        logits = self.net(img_emb)
        # logits shape: [B, 2]
        return logits

class ConcatClassifier(nn.Module):
    """
    Baseline model 3: Concat Classifier
    Nhận input là vector được nối (concatenate) từ image và text embeddings, 
    phân loại thành 2 class (REAL/FAKE)
    """
    def __init__(self):
        super(ConcatClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, combined):
        # combined shape: [B, 1024]
        logits = self.net(combined)
        # logits shape: [B, 2]
        return logits
