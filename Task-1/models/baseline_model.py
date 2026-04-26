import torch
import torch.nn as nn
import torchvision.models as models


class BaselineVQAModel(nn.Module):
    """
    Baseline VQA model:
    - Image encoder: pretrained ResNet18 without final FC
    - Question encoder: Embedding + LSTM
    - Fusion: concatenation
    - Classifier: MLP over fused features
    """

    def __init__(self, vocab_size: int, num_answers: int) -> None:
        super().__init__()

        # Image encoder: ResNet18 -> 512-dim feature.
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Question encoder.
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=256,
            batch_first=True,
        )

        # Project image features to question feature size for interaction fusion.
        self.image_fusion_proj = nn.Linear(512, 256)

        # Fusion(256 + 256 + 256 = 768) + classifier.
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_answers),
        )

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        # Image features: [B, 3, 224, 224] -> [B, 512] -> [B, 256] for fusion.
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.image_fusion_proj(image_features)

        # Question features: [B, seq_len] -> [B, 256]
        embedded = self.embedding(questions)
        _, (hidden, _) = self.lstm(embedded)
        question_features = hidden[-1]

        # Concatenate image, question, and multiplicative interaction.
        fused = torch.cat(
            [image_features, question_features, image_features * question_features],
            dim=1,
        )
        logits = self.classifier(fused)
        return logits
