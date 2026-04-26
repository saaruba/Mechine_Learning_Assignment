import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, val_loader, device, num_epochs: int = 5) -> Dict[str, List[float]]:
    """
    Reusable training loop for SLAKE VQA models.

    Returns history dict with:
    - train_loss
    - val_loss
    - val_accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = -1.0
    outputs_dir = os.path.join("outputs", "checkpoints")
    os.makedirs(outputs_dir, exist_ok=True)
    best_model_path = os.path.join(outputs_dir, f"{model.__class__.__name__}_best.pt")

    for epoch in range(num_epochs):
        print(f"\nStarting epoch {epoch + 1}/{num_epochs}", flush=True)
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        print_every = max(1, len(train_loader) // 5)

        train_progress = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        )
        for i, (images, questions, labels) in train_progress:
            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images, questions)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_train_loss += loss.item() * batch_size
            train_samples += batch_size
            train_progress.set_postfix(loss=f"{loss.item():.4f}")

            if i % print_every == 0 or i == 1 or i == len(train_loader):
                print(f"Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)

        epoch_train_loss = running_train_loss / max(train_samples, 1)

        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        correct = 0

        with torch.no_grad():
            for images, questions, labels in val_loader:
                images = images.to(device)
                questions = questions.to(device)
                labels = labels.to(device)

                logits = model(images, questions)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                running_val_loss += loss.item() * batch_size
                val_samples += batch_size

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()

        epoch_val_loss = running_val_loss / max(val_samples, 1)
        epoch_val_accuracy = correct / max(val_samples, 1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_accuracy)

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_accuracy:.4f}",
            flush=True,
        )

    return history
