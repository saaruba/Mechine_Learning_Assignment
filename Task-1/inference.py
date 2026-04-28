import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from models.advanced_model import AdvancedVQAModel
from models.attention_model import AttentionVQAModel
from models.baseline_model import BaselineVQAModel

VALID_ORGANS = [
    "liver",
    "kidney",
    "spleen",
    "pancreas",
    "heart",
    "brain",
    "head",
    "lung",
    "chest",
    "abdomen",
]
DETECTION_CONFIDENCE_THRESHOLD = 0.10
FINAL_ANSWER_CONFIDENCE_THRESHOLD = 0.40

DEFAULT_ARTIFACTS_DIR = Path("outputs_checkpoints")
MODEL_REGISTRY = {
    "baseline": (BaselineVQAModel, "BaselineVQAModel_best.pt"),
    "attention": (AttentionVQAModel, "AttentionVQAModel_best.pt"),
    "advanced": (AdvancedVQAModel, "AdvancedVQAModel_best.pt"),
}
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower().strip())


def normalize_label(text: str) -> str:
    return text.strip().lower()


def improve_question(question: str) -> str:
    normalized = question.strip().lower()
    vague_patterns = (
        r"^what is in (the )?image\??$",
        r"^what is this\??$",
        r"^what object is visible\??$",
        r"^what is visible\??$",
    )
    for pattern in vague_patterns:
        if re.match(pattern, normalized):
            return "Which organ is visible in this medical image?"
    return question.strip()


def load_artifacts(artifacts_dir: Path) -> Tuple[Dict[str, int], int, List[str]]:
    vocab_path = artifacts_dir / "vocab.pkl"
    idx_to_answer_path = artifacts_dir / "idx_to_answer.json"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
    if not idx_to_answer_path.exists():
        raise FileNotFoundError(f"Missing answer map file: {idx_to_answer_path}")

    with open(vocab_path, "rb") as f:
        vocab_payload = pickle.load(f)
    with open(idx_to_answer_path, "r", encoding="utf-8") as f:
        idx_to_answer_raw = json.load(f)

    if not isinstance(vocab_payload, dict) or "word_to_idx" not in vocab_payload or "max_question_len" not in vocab_payload:
        raise ValueError("Invalid vocab.pkl format.")

    if isinstance(idx_to_answer_raw, dict):
        idx_to_answer = [idx_to_answer_raw[str(i)] for i in range(len(idx_to_answer_raw))]
    elif isinstance(idx_to_answer_raw, list):
        idx_to_answer = idx_to_answer_raw
    else:
        raise ValueError("Invalid idx_to_answer.json format.")

    return vocab_payload["word_to_idx"], int(vocab_payload["max_question_len"]), idx_to_answer


def load_model(
    model_name: str,
    checkpoint_path: Path,
    vocab_size: int,
    num_answers: int,
    device: torch.device,
) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {', '.join(MODEL_REGISTRY.keys())}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model_cls = MODEL_REGISTRY[model_name][0]
    model = model_cls(vocab_size=vocab_size, num_answers=num_answers)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    CLAHE-enhanced preprocessing for medical scans while keeping model input format unchanged.
    """
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_file}")

    image_bgr = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read image file: {image_file}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    rgb_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

    pil_image = Image.fromarray(rgb_image)
    return IMAGE_TRANSFORM(pil_image).unsqueeze(0).to(device)


def encode_question(
    question: str,
    word_to_idx: Dict[str, int],
    max_question_len: int,
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenize(question)
    unk_id = word_to_idx.get("<unk>", 1)
    pad_id = word_to_idx.get("<pad>", 0)

    token_ids = [word_to_idx.get(tok, unk_id) for tok in tokens[:max_question_len]]
    if len(token_ids) < max_question_len:
        token_ids.extend([pad_id] * (max_question_len - len(token_ids)))

    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)


def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    question_tensor: torch.Tensor,
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    with torch.no_grad():
        logits = model(image_tensor, question_tensor)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, probabilities.shape[1]), dim=1)
    return list(zip(top_indices.squeeze(0).tolist(), top_probs.squeeze(0).tolist()))


def is_valid_organ(answer: str, valid_organs: Sequence[str]) -> bool:
    normalized = normalize_label(answer)
    valid_set = {organ.lower() for organ in valid_organs}
    if normalized in valid_set:
        return True
    return any(organ in normalized for organ in valid_set)


def extract_detected_organs(
    top_answers: Sequence[Tuple[str, float]],
    valid_organs: Sequence[str],
    threshold: float = DETECTION_CONFIDENCE_THRESHOLD,
) -> List[str]:
    valid_set = {organ.lower() for organ in valid_organs}
    detected: List[str] = []

    for answer, prob in top_answers:
        if prob < threshold:
            continue
        normalized = normalize_label(answer)

        matched = None
        if normalized in valid_set:
            matched = normalized
        else:
            for organ in valid_set:
                if organ in normalized:
                    matched = organ
                    break

        if matched is not None and matched not in detected:
            detected.append(matched)

    return detected


def apply_context_filter(detected_organs: Sequence[str]) -> List[str]:
    organs = [normalize_label(organ) for organ in detected_organs]
    if "chest" in organs:
        allowed = {"chest", "lung", "heart"}
        return [organ for organ in organs if organ in allowed]
    if "brain" in organs or "head" in organs:
        allowed = {"brain", "head"}
        return [organ for organ in organs if organ in allowed]
    return organs


def get_confidence_for_answer(answer: str, top_answers: Sequence[Tuple[str, float]]) -> float:
    target = normalize_label(answer)
    for candidate, prob in top_answers:
        if normalize_label(candidate) == target:
            return float(prob)
    for candidate, prob in top_answers:
        if target in normalize_label(candidate) or normalize_label(candidate) in target:
            return float(prob)
    return 0.0


def refine_answer(
    top_answers: Sequence[Tuple[str, float]],
    detected_organs: Sequence[str],
    valid_organs: Sequence[str],
    min_confidence: float = FINAL_ANSWER_CONFIDENCE_THRESHOLD,
) -> Tuple[str, float]:
    """
    Final answer logic:
    - Prefer detected organ if available.
    - Otherwise fallback to top prediction.
    - If confidence < threshold, return "uncertain".
    """
    if not top_answers:
        return "uncertain", 0.0

    top_answer, top_conf = top_answers[0]

    if detected_organs:
        final_answer = detected_organs[0]
        final_conf = get_confidence_for_answer(final_answer, top_answers)
        if final_conf <= 0.0:
            final_conf = float(top_conf)
    else:
        final_answer = normalize_label(top_answer)
        final_conf = float(top_conf)

    if not is_valid_organ(final_answer, valid_organs):
        return "uncertain", final_conf
    if final_conf < min_confidence:
        return "uncertain", final_conf
    return final_answer, final_conf


def organ_to_region(organ: str) -> str:
    name = normalize_label(organ)
    if name == "lung":
        return "chest"
    if name in {"brain", "head"}:
        return "head"
    if name == "heart":
        return "center_chest"
    if name == "chest":
        return "chest"
    if name in {"liver", "kidney", "spleen", "pancreas", "abdomen"}:
        return "abdomen"
    return "unknown"


def detect_region(image_bgr) -> List[Dict[str, object]]:
    """
    Region detection using blur + Canny + contour filtering.
    Returns top valid regions by area (max 3).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours_result = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]

    h, w = gray.shape
    image_area = float(h * w)
    min_area = max(250.0, image_area * 0.0015)
    max_area = image_area * 0.55

    regions: List[Dict[str, object]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = float(bw) / float(max(bh, 1))
        if aspect_ratio < 0.35 or aspect_ratio > 3.2:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        regions.append(
            {
                "contour": contour,
                "area": area,
                "bbox": (x, y, bw, bh),
                "center": (int(cx), int(cy)),
                "radius": max(int(radius), 8),
            }
        )

    regions.sort(key=lambda r: float(r["area"]), reverse=True)
    return regions[:3]


def should_localize(final_answer: str, detected_organs: Sequence[str]) -> bool:
    if normalize_label(final_answer) == "uncertain":
        return False

    predicted_region = organ_to_region(final_answer)
    if predicted_region == "unknown":
        return False

    if not detected_organs:
        return True

    detected_region = organ_to_region(detected_organs[0])
    return detected_region == predicted_region


def detect_and_visualize_organs(
    image_path: str,
    final_answer: str,
    detected_organs: Sequence[str],
) -> str:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image for visualization: {image_path}")

    regions = detect_region(image)
    can_draw = should_localize(final_answer, detected_organs) and len(regions) > 0

    if can_draw:
        best_region = regions[0]
        center = best_region["center"]
        radius = best_region["radius"]
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.putText(
            image,
            normalize_label(final_answer),
            (center[0] - radius, center[1] - radius - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            image,
            "Low confidence localization",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output_path = Path("output_visualization.png")
    cv2.imwrite(str(output_path), image)
    return str(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SLAKE VQA inference CLI")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="advanced", help="Model to load")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional custom checkpoint path")
    parser.add_argument("--artifacts-dir", type=str, default=str(DEFAULT_ARTIFACTS_DIR), help="Directory with vocab and answer maps")
    parser.add_argument("--image", type=str, default="", help="Optional image path for non-interactive use")
    parser.add_argument("--question", type=str, default="", help="Optional question for non-interactive use")
    parser.add_argument("--top-k", type=int, default=3, help="Number of predictions to display")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir = Path(args.artifacts_dir)

    try:
        word_to_idx, max_question_len, idx_to_answer = load_artifacts(artifacts_dir)

        model_checkpoint = Path(args.checkpoint) if args.checkpoint else artifacts_dir / MODEL_REGISTRY[args.model][1]
        model = load_model(
            model_name=args.model,
            checkpoint_path=model_checkpoint,
            vocab_size=len(word_to_idx),
            num_answers=len(idx_to_answer),
            device=device,
        )

        print(f"Using model: {args.model}", flush=True)
        print(f"Device: {device}", flush=True)

        image_path = args.image.strip() if args.image else input("Enter image path: ").strip()
        question_input = args.question.strip() if args.question else input("Enter your question: ").strip()

        if not image_path:
            raise ValueError("Image path is empty. Please provide a valid file path.")
        if not question_input:
            raise ValueError("Question is empty. Please enter a question.")

        improved_question = improve_question(question_input)
        if improved_question != question_input:
            print(f"\nRefined question: {improved_question}", flush=True)

        image_tensor = preprocess_image(image_path, device)
        question_tensor = encode_question(improved_question, word_to_idx, max_question_len, device)

        raw_top = predict(model, image_tensor, question_tensor, top_k=max(1, args.top_k))
        top_answers: List[Tuple[str, float]] = []
        for class_idx, prob in raw_top:
            answer = str(idx_to_answer[class_idx]) if 0 <= class_idx < len(idx_to_answer) else "unknown"
            top_answers.append((answer, float(prob)))

        print("\nTop Predictions:", flush=True)
        for rank, (answer, prob) in enumerate(top_answers, start=1):
            print(f"{rank}. {answer} ({prob:.2f})", flush=True)

        detected_organs = extract_detected_organs(top_answers, VALID_ORGANS, DETECTION_CONFIDENCE_THRESHOLD)
        detected_organs = apply_context_filter(detected_organs)
        print(f"\nDetected organs: {detected_organs}", flush=True)

        final_answer, final_conf = refine_answer(
            top_answers=top_answers,
            detected_organs=detected_organs,
            valid_organs=VALID_ORGANS,
            min_confidence=FINAL_ANSWER_CONFIDENCE_THRESHOLD,
        )
        print(f"\nFinal Answer: {final_answer}", flush=True)
        print(f"Confidence: {final_conf:.2f}", flush=True)

        vis_path = detect_and_visualize_organs(
            image_path=image_path,
            final_answer=final_answer,
            detected_organs=detected_organs,
        )
        print(f"\nVisualization saved at: {vis_path}", flush=True)

    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Inference error: {exc}", flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
