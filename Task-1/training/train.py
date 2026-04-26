import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def _compute_class_weights(train_loader, device) -> Optional[torch.Tensor]:
    samples = getattr(train_loader.dataset, "samples", None)
    if not samples:
        return None

    labels = [int(sample[2]) for sample in samples]
    if not labels:
        return None

    num_classes = max(labels) + 1
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()

    class_weights = torch.zeros_like(counts)
    nonzero_mask = counts > 0
    class_weights[nonzero_mask] = 1.0 / counts[nonzero_mask]

    # Normalize non-zero weights to keep average scale stable.
    class_weights[nonzero_mask] = class_weights[nonzero_mask] / class_weights[nonzero_mask].mean()
    return class_weights.to(device)


def train_model(model, train_loader, val_loader, device, num_epochs: int = 5) -> Dict[str, List[float]]:
    """
    Reusable training loop for SLAKE VQA models.

    Returns history dict with:
    - train_loss
    - val_loss
    - val_accuracy
    """
    # Freeze visual backbone so only fusion/classifier/question parts are trained.
    if hasattr(model, "image_encoder"):
        for param in model.image_encoder.parameters():
            param.requires_grad = False

    class_weights = _compute_class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    outputs_dir = "outputs_checkpoints"
    os.makedirs(outputs_dir, exist_ok=True)
    best_model_path = os.path.join(outputs_dir, f"{model.__class__.__name__}_best.pt")

    best_val_loss = float("inf")
    early_stop_patience = 3
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}", flush=True)
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        train_progress = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )

        for i, (images, questions, labels) in train_progress:
            images = images.to(device, non_blocking=True)
            questions = questions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, questions)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples += batch_size

            train_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if i % 100 == 0 or i == len(train_loader):
                print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)

        epoch_train_loss = running_train_loss / max(train_samples, 1)

        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        correct = 0

        with torch.no_grad():
            for images, questions, labels in val_loader:
                images = images.to(device, non_blocking=True)
                questions = questions.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images, questions)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                running_val_loss += loss.item() * batch_size
                val_samples += batch_size

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()

        epoch_val_loss = running_val_loss / max(val_samples, 1)
        epoch_val_accuracy = correct / max(val_samples, 1)
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_accuracy)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_accuracy:.4f}",
            flush=True,
        )

        if epochs_without_improvement >= early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1} "
                f"(no val loss improvement for {early_stop_patience} epochs).",
                flush=True,
            )
            break

    return history
