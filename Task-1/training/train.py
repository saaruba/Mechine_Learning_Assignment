from collections import Counter
import json
import os
import pickle
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def _extract_labels(train_loader) -> List[int]:
    """
    Prefer cached dataset labels for efficient class-weight computation.
    Falls back to dataset.samples, then dataloader iteration if needed.
    """
    train_dataset = train_loader.dataset
    labels = getattr(train_dataset, "labels", None)
    if labels is not None and len(labels) > 0:
        return [int(x) for x in labels]

    samples = getattr(train_dataset, "samples", None)
    if samples is not None and len(samples) > 0:
        return [int(sample[2]) for sample in samples]

    collected = []
    for _, _, batch_labels in train_loader:
        collected.extend(batch_labels.tolist())
    return [int(x) for x in collected]


def _save_label_maps(train_loader, output_dir: str = "outputs_checkpoints") -> None:
    """
    Save vocab and answer mappings so inference can use exactly the same indices.
    """
    dataset = train_loader.dataset
    processor = getattr(dataset, "processor", None)
    if processor is None:
        return

    metadata = getattr(processor, "metadata", None)
    if metadata is None:
        return

    os.makedirs(output_dir, exist_ok=True)

    vocab_payload = {
        "word_to_idx": metadata.word_to_idx,
        "idx_to_word": metadata.idx_to_word,
        "max_question_len": metadata.max_question_len,
    }
    with open(os.path.join(output_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab_payload, f)

    with open(os.path.join(output_dir, "answer_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(metadata.answer_to_idx, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "idx_to_answer.json"), "w", encoding="utf-8") as f:
        json.dump(metadata.idx_to_answer, f, ensure_ascii=False, indent=2)


def train_model(model, train_loader, val_loader, device, num_epochs: int = 20) -> Dict[str, List[float]]:
    """
    Reusable training loop for SLAKE VQA models.

    Returns:
      {
        "train_loss": [...],
        "val_loss": [...],
        "val_accuracy": [...]
      }
    """
    labels = _extract_labels(train_loader)
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = max(labels) + 1 if labels else 1

    # Dynamic class weights for imbalance handling.
    # If a class index is absent in the current train split, fallback to 1 to avoid divide-by-zero.
    weights = torch.tensor(
        [total / max(label_counts.get(i, 1), 1) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )
    # Less aggressive schedule for smoother training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    os.makedirs("outputs_checkpoints", exist_ok=True)
    best_model_path = os.path.join("outputs_checkpoints", f"{model.__class__.__name__}_best.pt")
    _save_label_maps(train_loader, output_dir="outputs_checkpoints")

    best_val_loss = float("inf")
    patience = 3
    counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        train_progress = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Train {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for i, (images, questions, labels_batch) in train_progress:
            images = images.to(device, non_blocking=True)
            questions = questions.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                outputs = model(images, questions)
                loss = criterion(outputs, labels_batch)

            scaler.scale(loss).backward()

            # Gradient clipping for stability.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            batch_size = labels_batch.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples += batch_size

            train_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
            if i % 100 == 0 or i == len(train_loader):
                print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)

        train_loss = running_train_loss / max(train_samples, 1)

        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        correct = 0

        with torch.no_grad():
            for images, questions, labels_batch in val_loader:
                images = images.to(device, non_blocking=True)
                questions = questions.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    outputs = model(images, questions)
                    loss = criterion(outputs, labels_batch)

                batch_size = labels_batch.size(0)
                running_val_loss += loss.item() * batch_size
                val_samples += batch_size

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels_batch).sum().item()

        val_loss = running_val_loss / max(val_samples, 1)
        val_accuracy = correct / max(val_samples, 1)

        # Step scheduler once per epoch for cosine annealing.
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered", flush=True)
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}",
                    flush=True,
                )
                break

        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}",
            flush=True,
        )

    return history
