"""
Model 2: Lightweight CLIP Fine-tuned for Medical VQA
Fast variant - optimized for quick training
"""

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import logging

logger = logging.getLogger(__name__)


class CLIPFastVQA(nn.Module):
    """
    CLIP-based model for VQA
    Uses smaller ViT-B/32 model for faster training
    """

    def __init__(self, num_classes, device='cuda', freeze_vision=False, freeze_text=False):
        super(CLIPFastVQA, self).__init__()
        self.device = device
        self.num_classes = num_classes

        # Load CLIP model (ViT-B/32 is lighter than ViT-L/14)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze encoders if specified
        if freeze_vision:
            for param in self.clip_model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")

        if freeze_text:
            for param in self.clip_model.text_model.parameters():
                param.requires_grad = False
            logger.info("Text encoder frozen")

        # Classification head
        clip_dim = self.clip_model.projection_dim  # 512
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        logger.info(f"✓ CLIP-Fast model initialized with {num_classes} classes")

    def forward(self, images, question_texts):
        """
        Args:
            images: Tensor of shape (B, 3, 224, 224)
            question_texts: List of question strings
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        batch_size = images.size(0)

        # Process images through CLIP vision encoder
        with torch.no_grad():
            image_features = self.clip_model.vision_model(images)
            image_features = self.clip_model.visual_projection(image_features.pooler_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Process text through CLIP text encoder
        text_inputs = self.processor(text=question_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.text_model(**text_inputs)
            text_features = self.clip_model.text_projection(text_features.pooler_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Combine image and text features (element-wise addition)
        combined_features = image_features + text_features  # (B, 512)

        # Classification
        logits = self.classifier(combined_features)

        return logits, combined_features

    def get_model_info(self):
        """Return model architecture info"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "Model": "CLIP-ViT-Base/32 (Fast)",
            "Total Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Image Encoder": "Vision Transformer (ViT-B/32)",
            "Text Encoder": "CLIP Text Model",
            "Fusion Strategy": "Feature Addition + Classification Head",
            "Output Classes": self.num_classes,
        }


def create_clip_fast_model(num_classes, device='cuda'):
    """Factory function to create CLIP-Fast model"""
    model = CLIPFastVQA(num_classes=num_classes, device=device, freeze_vision=False, freeze_text=False)
    return model
