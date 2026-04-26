from typing import Dict

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def evaluate_model(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, questions, labels in dataloader:
            images = images.to(device)
            questions = questions.to(device)
            labels = labels.to(device)

            outputs = model(images, questions)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Balanced Accuracy:", bal_acc)

    return {
        "accuracy": acc,
        "f1_score_macro": f1,
        "balanced_accuracy": bal_acc,
    }
