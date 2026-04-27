import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.advanced_model import AdvancedVQAModel
from models.attention_model import AttentionVQAModel
from models.baseline_model import BaselineVQAModel
from evaluation.evaluate import evaluate_model
from training.train import train_model
from visualization.plots import save_model_comparison, save_training_curves


@dataclass
class SlakeVQAMetadata:
    """Container for fitted preprocessing state from the train split only."""

    word_to_idx: Dict[str, int]
    idx_to_word: List[str]
    answer_to_idx: Dict[str, int]
    idx_to_answer: List[str]
    max_question_len: int


class SlakeVQAProcessor:
    """
    Fits vocab + answer label space on training data, then encodes any split.

    Notes:
    - Vocabulary is built from train questions only.
    - Answer classes are top-K frequent train answers only.
    - Rare answers are ignored via sample filtering in dataset creation.
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, top_k_answers: int = 50, max_question_len: Optional[int] = 20) -> None:
        self.top_k_answers = top_k_answers
        self.max_question_len = max_question_len
        self._fitted = False
        self.metadata: Optional[SlakeVQAMetadata] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower().strip()
        # Keep alphanumerics and apostrophes, split on everything else.
        return re.findall(r"[a-z0-9']+", text)

    def fit(self, train_entries: Sequence[dict]) -> SlakeVQAMetadata:
        question_counter: Counter = Counter()
        answer_counter: Counter = Counter()
        max_len_seen = 0

        for entry in train_entries:
            q_tokens = self.tokenize(entry["question"])
            question_counter.update(q_tokens)
            max_len_seen = max(max_len_seen, len(q_tokens))

            answer = str(entry["answer"]).strip().lower()
            if answer:
                answer_counter[answer] += 1

        idx_to_word = [self.PAD_TOKEN, self.UNK_TOKEN] + sorted(question_counter.keys())
        word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

        top_answers = answer_counter.most_common(self.top_k_answers)
        idx_to_answer = [answer for answer, _ in top_answers]
        answer_to_idx = {answer: idx for idx, answer in enumerate(idx_to_answer)}

        max_question_len = self.max_question_len if self.max_question_len is not None else max_len_seen

        self.metadata = SlakeVQAMetadata(
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            answer_to_idx=answer_to_idx,
            idx_to_answer=idx_to_answer,
            max_question_len=max_question_len,
        )
        self._fitted = True
        return self.metadata

    def ensure_fitted(self) -> SlakeVQAMetadata:
        if not self._fitted or self.metadata is None:
            raise RuntimeError("SlakeVQAProcessor must be fit() on train data before encoding.")
        return self.metadata

    def encode_question(self, question: str) -> torch.Tensor:
        metadata = self.ensure_fitted()
        tokens = self.tokenize(question)
        unk_id = metadata.word_to_idx[self.UNK_TOKEN]
        pad_id = metadata.word_to_idx[self.PAD_TOKEN]

        encoded = [metadata.word_to_idx.get(token, unk_id) for token in tokens[: metadata.max_question_len]]
        if len(encoded) < metadata.max_question_len:
            encoded.extend([pad_id] * (metadata.max_question_len - len(encoded)))

        return torch.tensor(encoded, dtype=torch.long)

    def encode_answer(self, answer: str) -> Optional[int]:
        metadata = self.ensure_fitted()
        normalized = str(answer).strip().lower()
        return metadata.answer_to_idx.get(normalized)


class SlakeVQADataset(Dataset):
    """
    PyTorch Dataset for SLAKE VQA.

    Returns:
      (image_tensor, question_tensor, answer_label)
    where:
      - image_tensor: float tensor [3, 224, 224]
      - question_tensor: long tensor [max_question_len]
      - answer_label: long scalar tensor
    """

    def __init__(
        self,
        entries: Sequence[dict],
        images_root: str,
        processor: SlakeVQAProcessor,
        image_transform: Optional[transforms.Compose] = None,
        drop_rare_answers: bool = True,
        image_lookup: Optional[Dict[str, str]] = None,
    ) -> None:
        self.images_root = images_root
        self.processor = processor
        self.image_transform = image_transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.image_lookup = image_lookup if image_lookup is not None else self._build_image_lookup(self.images_root)
        self.samples: List[Tuple[str, torch.Tensor, int]] = []

        for entry in entries:
            answer_idx = self.processor.encode_answer(entry["answer"])
            if answer_idx is None and drop_rare_answers:
                continue
            if answer_idx is None:
                continue

            image_path = self._resolve_image_path(entry["img_name"])
            if image_path is None:
                continue

            question_tensor = self.processor.encode_question(str(entry["question"]))
            self.samples.append((image_path, question_tensor, answer_idx))

        # Cached labels for class-imbalance handling in training.
        self.labels: List[int] = [sample[2] for sample in self.samples]

    @staticmethod
    def _build_image_lookup(images_root: str) -> Dict[str, str]:
        """
        Build lookup once for efficient path resolution.
        Supports both relative-path keys and filename keys.
        """
        lookup: Dict[str, str] = {}
        images_root = os.path.abspath(images_root)

        for root, _, files in os.walk(images_root):
            for file_name in files:
                full_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(full_path, images_root).replace("\\", "/")

                # Exact relative key: e.g. xmlab123/image.jpg
                lookup.setdefault(rel_path, full_path)
                # Basename key: e.g. image.jpg
                lookup.setdefault(file_name, full_path)

        return lookup

    def _resolve_image_path(self, img_name: str) -> Optional[str]:
        key = str(img_name).replace("\\", "/").lstrip("./")

        # 1) exact key in precomputed lookup
        if key in self.image_lookup:
            return self.image_lookup[key]

        # 2) direct join (handles path-like img_name)
        direct = os.path.join(self.images_root, key)
        if os.path.isfile(direct):
            return direct

        # 3) basename fallback
        basename = os.path.basename(key)
        return self.image_lookup.get(basename)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, question_tensor, answer_idx = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)
        answer_label = torch.tensor(answer_idx, dtype=torch.long)

        return image_tensor, question_tensor, answer_label


def _load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_slake_dataloaders(
    data_root: str = "data/Slake/Slake1.0",
    batch_size: int = 32,
    num_workers: int = 4,
    max_question_len: Optional[int] = 20,
    top_k_answers: int = 50,
    pin_memory: Optional[bool] = None,
) -> Tuple[Dict[str, DataLoader], SlakeVQAMetadata, SlakeVQAProcessor]:
    """
    Build train/validation/test dataloaders for SLAKE VQA.

    Expected layout under data_root:
      - train.json
      - validate.json
      - test.json
      - imgs/
    """

    train_json = os.path.join(data_root, "train.json")
    val_json = os.path.join(data_root, "validate.json")
    test_json = os.path.join(data_root, "test.json")
    images_root = os.path.join(data_root, "imgs")

    train_entries = _load_json(train_json)
    val_entries = _load_json(val_json)
    test_entries = _load_json(test_json)

    processor = SlakeVQAProcessor(top_k_answers=top_k_answers, max_question_len=max_question_len)
    metadata = processor.fit(train_entries)
    image_lookup = SlakeVQADataset._build_image_lookup(images_root)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SlakeVQADataset(
        entries=train_entries,
        images_root=images_root,
        processor=processor,
        image_transform=train_transform,
        drop_rare_answers=True,
        image_lookup=image_lookup,
    )
    val_dataset = SlakeVQADataset(
        entries=val_entries,
        images_root=images_root,
        processor=processor,
        image_transform=eval_transform,
        drop_rare_answers=True,
        image_lookup=image_lookup,
    )
    test_dataset = SlakeVQADataset(
        entries=test_entries,
        images_root=images_root,
        processor=processor,
        image_transform=eval_transform,
        drop_rare_answers=True,
        image_lookup=image_lookup,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        ),
        "validation": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        ),
    }

    return loaders, metadata, processor


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batch_size = 16

    loaders, metadata, _ = build_slake_dataloaders(
        data_root="data/Slake/Slake1.0",
        batch_size=train_batch_size,
        num_workers=2,
        max_question_len=20,
        top_k_answers=50,
    )

    baseline_model = BaselineVQAModel(
        vocab_size=len(metadata.idx_to_word),
        num_answers=len(metadata.idx_to_answer),
    ).to(device)

    images, questions, labels = next(iter(loaders["train"]))
    images = images.to(device)
    questions = questions.to(device)
    outputs = baseline_model(images, questions)
    print("Output shape:", outputs.shape)

    attention_model = AttentionVQAModel(
        vocab_size=len(metadata.idx_to_word),
        num_answers=len(metadata.idx_to_answer),
    ).to(device)
    outputs_attn = attention_model(images, questions)
    print("Attention Output shape:", outputs_attn.shape)

    advanced_model = AdvancedVQAModel(
        vocab_size=len(metadata.idx_to_word),
        num_answers=len(metadata.idx_to_answer),
    ).to(device)
    outputs_adv = advanced_model(images, questions)
    print("Advanced Output shape:", outputs_adv.shape)

    baseline_history = train_model(
        baseline_model, loaders["train"], loaders["validation"], device, num_epochs=15
    )
    print("Baseline training done")
    attention_history = train_model(
        attention_model, loaders["train"], loaders["validation"], device, num_epochs=15
    )
    print("Attention training done")
    advanced_history = train_model(
        advanced_model, loaders["train"], loaders["validation"], device, num_epochs=15
    )
    print("Advanced training done")

    print("\nEvaluating Baseline Model")
    baseline_metrics = evaluate_model(baseline_model, loaders["test"], device)

    print("\nEvaluating Attention Model")
    attention_metrics = evaluate_model(attention_model, loaders["test"], device)

    print("\nEvaluating Advanced Model")
    advanced_metrics = evaluate_model(advanced_model, loaders["test"], device)

    save_training_curves(baseline_history, "Baseline")
    save_training_curves(attention_history, "Attention")
    save_training_curves(advanced_history, "Advanced")
    save_model_comparison(
        {
            "Baseline": baseline_metrics,
            "Attention": attention_metrics,
            "Advanced": advanced_metrics,
        }
    )

    print("Vocab size:", len(metadata.idx_to_word))
    print("Answer classes:", len(metadata.idx_to_answer))
    for split, loader in loaders.items():
        print(split, "num_batches:", len(loader), "num_samples:", len(loader.dataset))

 
