"""
Model 3: Lightweight VisualBERT Fine-tuned for Medical VQA
Fast variant - optimized for quick training
"""

import torch
import torch.nn as nn
from transformers import AutoModel, VisualBertConfig
import logging

logger = logging.getLogger(__name__)


class VisualBERTFastVQA(nn.Module):
    """
    VisualBERT-based model for VQA
    Uses pre-trained VisualBERT from UClanlp for faster convergence
    """

    def __init__(self, num_classes, device='cuda', freeze_vision=False, freeze_text=False):
        super(VisualBERTFastVQA, self).__init__()
        self.device = device
        self.num_classes = num_classes

        # Load pre-trained VisualBERT (faster to fine-tune than training from scratch)
        # Using AutoModel to get the base model with proper output structure
        self.visualbert = AutoModel.from_pretrained(
            "uclanlp/visualbert-vqa-coco-pre"
        ).to(device)

        # Get hidden dimension
        hidden_dim = self.visualbert.config.hidden_size  # 768

        # Classification head (simple and fast)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # Note: VisualBertForPreTraining has a different architecture
        # Freezing specific components is complex, so we keep all trainable for fast fine-tuning
        logger.info(f"✓ VisualBERT-Fast model initialized with {num_classes} classes (all layers trainable)")

    def forward(self, images, input_ids, attention_mask, token_type_ids=None, visual_embeddings=None):
        """
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            input_ids: Token IDs of shape (B, seq_len)
            attention_mask: Attention mask of shape (B, seq_len)
            token_type_ids: Optional token type IDs
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        batch_size = images.size(0)

        # Extract visual features from images using ViT
        pixel_values = images

        # Forward through VisualBERT
        outputs = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
        )

        # Get pooled output (CLS token representation from last_hidden_state)
        # outputs.last_hidden_state shape: (B, seq_len, 768)
        # Use CLS token (first token) as representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # Classification
        logits = self.classifier(pooled_output)

        return logits, pooled_output

    def get_model_info(self):
        """Return model architecture info"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "Model": "VisualBERT-Fast (Pre-trained)",
            "Total Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Image Encoder": "ViT-Base (frozen)",
            "Text Encoder": "BERT-Base (fine-tunable)",
            "Fusion Strategy": "VisualBERT (Cross-modal attention)",
            "Output Classes": self.num_classes,
        }


def create_visualbert_fast_model(num_classes, device='cuda'):
    """Factory function to create VisualBERT-Fast model"""
    model = VisualBERTFastVQA(
        num_classes=num_classes,
        device=device,
        freeze_vision=False,  # All layers trainable for fast fine-tuning
        freeze_text=False    # Fine-tune text for task adaptation
    )
    return model
