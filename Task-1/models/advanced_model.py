import torch
import torch.nn as nn
import torchvision.models as models


class AdvancedVQAModel(nn.Module):
    """
    Advanced VQA model with:
    - ResNet18 visual encoder (fine-tune layer4 only)
    - GRU question encoder
    - Lightweight feature projections
    - Dropout + MLP fusion head
    """

    def __init__(self, vocab_size: int, num_answers: int) -> None:
        super().__init__()

        try:
            self.cnn = models.resnet18(weights="DEFAULT")
        except Exception:
            self.cnn = models.resnet18(pretrained=True)

        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.layer4.parameters():
            param.requires_grad = True

        self.image_encoder = nn.Sequential(*list(self.cnn.children())[:-1])  # [B, 512, 1, 1]

        self.embedding = nn.Embedding(vocab_size, 256)
        self.gru = nn.GRU(256, 512, batch_first=True)

        self.image_proj = nn.Linear(512, 512)
        self.question_proj = nn.Linear(512, 512)

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_answers)

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images).flatten(1)  # [B, 512]
        image_features = torch.relu(self.image_proj(image_features))

        embedded = self.embedding(questions)
        _, hidden = self.gru(embedded)
        question_features = hidden.squeeze(0)  # [B, 512]
        question_features = torch.relu(self.question_proj(question_features))

        combined = torch.cat((image_features, question_features), dim=1)
        combined = self.dropout(combined)
        combined = self.fc1(combined)
        combined = torch.relu(combined)
        logits = self.fc2(combined)
        return logits
