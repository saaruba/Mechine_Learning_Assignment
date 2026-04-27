import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AttentionVQAModel(nn.Module):
    """
    Attention-based VQA model with:
    - ResNet18 spatial encoder (fine-tune layer3/layer4)
    - GRU question encoder
    - Question-guided spatial attention
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

        self.image_encoder = nn.Sequential(*list(self.cnn.children())[:-2])  # [B, 512, 7, 7]

        # Stronger question encoder.
        self.embedding = nn.Embedding(vocab_size, 300)
        self.gru = nn.GRU(300, 512, batch_first=True)

        self.image_proj = nn.Linear(512, 512)
        self.question_proj = nn.Linear(512, 512)

        # Gated fusion + higher-capacity classifier.
        self.gate_fc = nn.Linear(512 + 512, 512)
        self.fc1 = nn.Linear(512 + 512, 1024)
        # LayerNorm is stable even when batch size is 1.
        self.bn1 = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_answers)

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        image_map = self.image_encoder(images)  # [B, 512, H, W]
        bsz, channels, height, width = image_map.shape
        image_tokens = image_map.view(bsz, channels, height * width).permute(0, 2, 1)  # [B, N, 512]

        embedded = self.embedding(questions)
        _, hidden = self.gru(embedded)
        question_features = hidden[-1]  # [B, 512]

        attn_scores = torch.tanh(
            self.image_proj(image_tokens) + self.question_proj(question_features).unsqueeze(1)
        )  # [B, N, 512]
        attn_scores = attn_scores.sum(dim=-1)  # [B, N]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N]

        image_features = torch.bmm(attn_weights.unsqueeze(1), image_tokens).squeeze(1)  # [B, 512]

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
