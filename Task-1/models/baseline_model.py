import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaselineVQAModel(nn.Module):
    """
    Baseline VQA model with:
    - ResNet18 visual encoder (fine-tune layer3/layer4)
    - GRU question encoder
    - Higher-capacity fusion head
    """

    def __init__(self, vocab_size: int, num_answers: int) -> None:
        super().__init__()

        try:
            self.cnn = models.resnet18(weights="DEFAULT")
        except Exception:
            self.cnn = models.resnet18(pretrained=True)

        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.layer3.parameters():
            param.requires_grad = True
        for param in self.cnn.layer4.parameters():
            param.requires_grad = True

        self.image_encoder = nn.Sequential(*list(self.cnn.children())[:-1])  # [B, 512, 1, 1]

        # Stronger question encoder.
        self.embedding = nn.Embedding(vocab_size, 300)
        self.gru = nn.GRU(300, 512, batch_first=True)

        # Gated fusion + higher-capacity classifier.
        self.gate_fc = nn.Linear(512 + 512, 512)
        self.fc1 = nn.Linear(512 + 512, 1024)
        # LayerNorm is stable even when batch size is 1.
        self.bn1 = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_answers)

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images).flatten(1)  # [B, 512]

        embedded = self.embedding(questions)
        _, hidden = self.gru(embedded)
        question_features = hidden[-1]  # [B, 512]

        # Normalize features before fusion.
        image_features = F.normalize(image_features, dim=1)
        question_features = F.normalize(question_features, dim=1)

        gate = torch.sigmoid(self.gate_fc(torch.cat([image_features, question_features], dim=1)))
        fused = gate * image_features + (1.0 - gate) * question_features
        combined = torch.cat([fused, question_features], dim=1)

        combined = self.fc1(combined)
        combined = self.bn1(combined)
        combined = torch.relu(combined)
        combined = self.dropout(combined)
        logits = self.fc2(combined)
        return logits
