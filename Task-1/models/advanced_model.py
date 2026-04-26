import torch
import torch.nn as nn
import torchvision.models as models


class AdvancedVQAModel(nn.Module):
    """
    Advanced VQA model:
    - Image encoder: pretrained ResNet18 without final FC
    - Question encoder: Embedding + GRU
    - Projection to shared space
    - Fusion via element-wise multiplication
    - Classifier for answer prediction
    """

    def __init__(self, vocab_size: int, num_answers: int) -> None:
        super().__init__()

        # Image encoder -> [B, 512]
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Question encoder -> [B, 256]
        self.embedding = nn.Embedding(vocab_size, 300)
        self.gru = nn.GRU(
            input_size=300,
            hidden_size=256,
            batch_first=True,
        )

        # Feature projections to shared 256-dim space.
        self.image_proj = nn.Linear(512, 256)
        self.question_proj = nn.Linear(256, 256)

        # Classifier on fused feature.
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_answers),
        )

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        # Image features: [B, 3, 224, 224] -> [B, 512]
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)

        # Question features: [B, seq_len] -> [B, 256]
        embedded = self.embedding(questions)
        _, hidden = self.gru(embedded)
        question_features = hidden[-1]

        # Project both modalities to shared space.
        image_proj = self.image_proj(image_features)
        question_proj = self.question_proj(question_features)

        # Improved fusion: element-wise multiplication.
        fused = image_proj * question_proj

        logits = self.classifier(fused)
        return logits
