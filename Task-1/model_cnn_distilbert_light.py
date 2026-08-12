"""
Model 1 LIGHT: MobileNet + DistilBERT (Fast Version)
Lightweight alternative to CNN + BioBERT
Trains in 3-5 hours instead of 20 hours
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class DistilBERTTextEncoder(nn.Module):
    """Lightweight text encoder using DistilBERT"""

    def __init__(self):
        super().__init__()
        self.model_name = "distilbert-base-uncased"
        self.encoder = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.hidden_dim = 768

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token


class MobileNetImageEncoder(nn.Module):
    """Lightweight image encoder using MobileNetV2"""

    def __init__(self):
        super().__init__()
        # MobileNetV2 is 10x smaller than ResNet50
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # Remove classification head
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])
        self.hidden_dim = 1280  # MobileNetV2 output channels

    def forward(self, images):
        features = self.mobilenet(images)
        return torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)


class CNNDistilBERTLight(nn.Module):
    """
    Lightweight Model 1: MobileNet + DistilBERT
    - MobileNet instead of ResNet50 (10x smaller)
    - DistilBERT instead of BioBERT (faster, lighter)
    - Simple fusion instead of attention (faster)
    """

    def __init__(self, num_classes, device='cuda'):
        super().__init__()
        self.device = device

        # Lightweight encoders
        self.image_encoder = MobileNetImageEncoder().to(device)
        self.text_encoder = DistilBERTTextEncoder().to(device)

        # Dimensions
        image_dim = self.image_encoder.hidden_dim  # 1280
        text_dim = self.text_encoder.hidden_dim    # 768

        # Simple fusion layer (faster than attention)
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

        logger.info(f"✓ CNN-DistilBERT Light model initialized with {num_classes} classes")

    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: (B, 3, 224, 224)
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        """
        # Image features
        image_features = self.image_encoder(images)  # (B, 1280)

        # Text features
        text_features = self.text_encoder(input_ids, attention_mask)  # (B, 768)

        # Concatenate
        combined = torch.cat([image_features, text_features], dim=1)  # (B, 2048)

        # Fusion
        fused = self.fusion(combined)  # (B, 256)

        # Classification
        logits = self.classifier(fused)  # (B, num_classes)

        return logits, fused

    def get_model_info(self):
        """Return model architecture info"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "Model": "CNN-DistilBERT-Light",
            "Total Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Image Encoder": "MobileNetV2 (lightweight)",
            "Text Encoder": "DistilBERT (fast)",
            "Fusion Strategy": "Concatenation + MLP",
            "Output Classes": self.classifier.out_features,
        }


def create_light_model(num_classes, device='cuda'):
    """Factory function"""
    model = CNNDistilBERTLight(num_classes=num_classes, device=device)
    return model
