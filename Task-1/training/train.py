#this is the file that we are going to us for training 
import os 

from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_model(model, train_loader, val_loader, device, num_epochs: int = 10) -> Dict[str, List[float]]:
    """
    Reusable training loop for SLAKE VQA models.

    Returns:
      {
        "train_loss": [...],
        "val_loss": [...],
        "val_accuracy": [...]
      }
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=1,
        factor=0.5,
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    os.makedirs("outputs_checkpoints", exist_ok=True)
    best_model_path = os.path.join("outputs_checkpoints", f"{model.__class__.__name__}_best.pt")

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

        for i, (images, questions, labels) in train_progress:
            images = images.to(device, non_blocking=True)
            questions = questions.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(device.type == "cuda")):
                outputs = model(images, questions)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
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
            for images, questions, labels in val_loader:
                images = images.to(device, non_blocking=True)
                questions = questions.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(enabled=(device.type == "cuda")):
                    outputs = model(images, questions)
                    loss = criterion(outputs, labels)

                batch_size = labels.size(0)
                running_val_loss += loss.item() * batch_size
                val_samples += batch_size

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()

        val_loss = running_val_loss / max(val_samples, 1)
        val_accuracy = correct / max(val_samples, 1)
        scheduler.step(val_loss)

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
