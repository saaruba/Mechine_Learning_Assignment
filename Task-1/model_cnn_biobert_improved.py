"""
Model 1: CNN + BioBERT for Medical VQA (Improved Implementation)

Key Improvements:
- BioBERT for medical text understanding (not generic BERT)
- Attention-based multi-modal fusion
- Professional architecture with ablation support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


class ResNet50ImageEncoder(nn.Module):
    """ResNet-50 based image encoder"""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, images):
        features = self.feature_extractor(images)
        features = features.squeeze(-1).squeeze(-1)
        projected = self.projection(features)
        return projected


class BioBERTTextEncoder(nn.Module):
    """BioBERT encoder for medical text"""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.model_name = "dmis-lab/biobert-v1.1"
        self.hidden_dim = hidden_dim

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.encoder = AutoModel.from_pretrained(self.model_name)
        biobert_dim = self.encoder.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(biobert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_output)
        return projected

    def tokenize(self, texts, max_length=64):
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )


class MultimodalAttentionFusion(nn.Module):
    """Attention-based multi-modal fusion"""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, image_features, text_features):
        combined = torch.cat([image_features, text_features], dim=1)
        attention_weight = self.attention(combined)
        attended_image = image_features * attention_weight
        fused = torch.cat([attended_image, text_features], dim=1)
        fused_features = self.fusion_gate(fused)
        return fused_features


class CNNBioBERT(nn.Module):
    """Complete CNN + BioBERT VQA Model"""

    def __init__(self, num_classes: int, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.image_encoder = ResNet50ImageEncoder(hidden_dim=hidden_dim)
        self.text_encoder = BioBERTTextEncoder(hidden_dim=hidden_dim)
        self.fusion = MultimodalAttentionFusion(hidden_dim=hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        logger.info(f"✓ CNN + BioBERT model initialized with {self._count_parameters():,} parameters")

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        fused_features = self.fusion(image_features, text_features)
        logits = self.classifier(fused_features)

        return logits, {
            'image_features': image_features,
            'text_features': text_features,
            'fused_features': fused_features
        }

    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        return {
            'model_name': 'CNN + BioBERT',
            'num_parameters': self._count_parameters(),
            'image_encoder': 'ResNet-50 (ImageNet pretrained)',
            'text_encoder': 'BioBERT (PubMed pretrained)',
            'fusion_type': 'Attention-based',
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
        }


def create_model(num_classes, device='cuda'):
    """Create and move model to device"""
    model = CNNBioBERT(num_classes=num_classes, hidden_dim=512)
    model = model.to(device)
    return model
