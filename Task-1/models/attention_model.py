import torch
import torch.nn as nn
import torchvision.models as models


class AttentionVQAModel(nn.Module):
    """
    Attention-based VQA model:
    - Image encoder: pretrained ResNet18 backbone (spatial map kept)
    - Question encoder: Embedding + LSTM
    - Attention over spatial image features conditioned on question
    - Fusion + classifier
    """

    def __init__(self, vocab_size: int, num_answers: int) -> None:
        super().__init__()

        # Image encoder: keep spatial features [B, 512, H, W].
        resnet = models.resnet18(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Question encoder.
        self.embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=256,
            batch_first=True,
        )

        # Attention projections.
        self.image_proj = nn.Linear(512, 256)
        self.question_proj = nn.Linear(256, 256)
        self.image_fusion_proj = nn.Linear(512, 256)

        # Classifier on fused feature [256 + 256 + 256 = 768].
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_answers),
        )

    def forward(self, images: torch.Tensor, questions: torch.Tensor) -> torch.Tensor:
        # Image feature map: [B, 3, 224, 224] -> [B, 512, H, W]
        image_map = self.image_encoder(images)
        bsz, channels, height, width = image_map.shape

        # Flatten spatial dims: [B, 512, H, W] -> [B, N, 512], where N = H * W
        image_flat = image_map.view(bsz, channels, height * width).permute(0, 2, 1)

        # Question features: [B, seq_len] -> [B, 256]
        embedded = self.embedding(questions)
        _, (hidden, _) = self.lstm(embedded)
        question_features = hidden[-1]

        # Attention:
        # image_proj: [B, N, 256], question_proj: [B, 1, 256]
        image_proj = self.image_proj(image_flat)
        question_proj = self.question_proj(question_features).unsqueeze(1)

        score = torch.tanh(image_proj + question_proj)         # [B, N, 256]
        score = score.sum(dim=-1)                              # [B, N]
        attention_weights = torch.softmax(score, dim=1)        # [B, N]

        # Weighted sum over spatial locations -> attended image feature [B, 512]
        attended_image = torch.bmm(attention_weights.unsqueeze(1), image_flat).squeeze(1)
        attended_image = self.image_fusion_proj(attended_image)  # [B, 256]

        # Fuse image, question, and multiplicative interaction.
        fused = torch.cat(
            [attended_image, question_features, attended_image * question_features],
            dim=1,
        )  # [B, 768]
        logits = self.classifier(fused)                                 # [B, num_answers]
        return logits
