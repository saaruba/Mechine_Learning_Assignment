import json
import pickle
import re
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models.advanced_model import AdvancedVQAModel


def tokenize(text: str):
    """Match training tokenization behavior."""
    return re.findall(r"[a-z0-9']+", text.lower().strip())


def load_artifacts(artifacts_dir: Path):
    vocab_path = artifacts_dir / "vocab.pkl"
    idx_to_answer_path = artifacts_dir / "idx_to_answer.json"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
    if not idx_to_answer_path.exists():
        raise FileNotFoundError(f"Missing answer map file: {idx_to_answer_path}")

    with open(vocab_path, "rb") as f:
        vocab_payload = pickle.load(f)

    with open(idx_to_answer_path, "r", encoding="utf-8") as f:
        idx_to_answer = json.load(f)

    word_to_idx = vocab_payload["word_to_idx"]
    max_question_len = int(vocab_payload["max_question_len"])

    return word_to_idx, max_question_len, idx_to_answer


def encode_question(question: str, word_to_idx: dict, max_question_len: int) -> torch.Tensor:
    tokens = tokenize(question)
    unk_id = word_to_idx.get("<unk>", 1)
    pad_id = word_to_idx.get("<pad>", 0)

    ids = [word_to_idx.get(tok, unk_id) for tok in tokens[:max_question_len]]
    if len(ids) < max_question_len:
        ids.extend([pad_id] * (max_question_len - len(ids)))

    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, seq_len]


def load_model(checkpoint_path: Path, vocab_size: int, num_answers: int, device: torch.device):
    model = AdvancedVQAModel(vocab_size=vocab_size, num_answers=num_answers)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir = Path("outputs_checkpoints")
    checkpoint_path = artifacts_dir / "AdvancedVQAModel_best.pt"

    word_to_idx, max_question_len, idx_to_answer = load_artifacts(artifacts_dir)
    model = load_model(
        checkpoint_path=checkpoint_path,
        vocab_size=len(word_to_idx),
        num_answers=len(idx_to_answer),
        device=device,
    )

    # Inference transform: deterministic image preprocessing used in evaluation.
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Sample questions:")
    print('- "What is in the image?"')
    print('- "What object is visible?"')
    print('- "What medical condition is shown?"')

    image_path = input("Enter image path: ").strip()
    question = input("Enter your question: ").strip()

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    question_tensor = encode_question(question, word_to_idx, max_question_len).to(device)  # [1, seq_len]

    with torch.no_grad():
        output = model(image_tensor, question_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    answer = idx_to_answer[predicted_class]
    print("Predicted Answer:", answer)


if __name__ == "__main__":
    main()
