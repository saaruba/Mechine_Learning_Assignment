import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdvancedVQAModel(nn.Module):
    """
    Advanced VQA model tuned for stronger performance:
    - ResNet18 visual encoder (fine-tune layer3/layer4)
    - GRU question encoder
    - Gated attention fusion (question-conditioned)
    - High-capacity classifier head
    """

    def __init__(self, vocab_size: int, num_answers: int) -> None:
        super().__init__()

        try:
            self.cnn = models.resnet18(weights="DEFAULT")
        except Exception:
            self.cnn = models.resnet18(pretrained=True)

        # Freeze early visual layers and fine-tune deeper layers.
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

        # Question-conditioned gating over image features.
        self.att_fc = nn.Linear(512, 512)

        # Higher-capacity classification head.
        self.fc1 = nn.Linear(1024, 1024)
        self.bn1 = nn.LayerNorm(1024)  # Robust for small/variable batch sizes.
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, num_answers)

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images).flatten(1)  # [B, 512]

        embedded = self.embedding(questions)
        _, hidden = self.gru(embedded)
        question_features = hidden[-1]  # [B, 512]

        # Normalize both modalities before fusion to stabilize optimization.
        image_features = F.normalize(image_features, dim=1)
        question_features = F.normalize(question_features, dim=1)

        # Gated attention fusion.
        attention = torch.sigmoid(self.att_fc(question_features))
        attended_image = image_features * attention
        combined = torch.cat((attended_image, question_features), dim=1)  # [B, 1024]

        combined = self.fc1(combined)
        combined = self.bn1(combined)
        combined = torch.relu(combined)
        combined = self.dropout(combined)
        output = self.fc2(combined)
        return output
