"""
Comprehensive metrics calculation for VQA task
Includes: Accuracy, F1, Balanced Accuracy, MRR, ECE, Confusion Matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import json
import os


class VQAMetrics:
    """Comprehensive metrics calculator for VQA models"""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                             y_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate all metrics at once"""
        metrics = {}

        # Basic accuracy metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Advanced metrics with probabilities
        if y_proba is not None:
            metrics["mean_reciprocal_rank"] = self.mean_reciprocal_rank(y_true, y_proba)
            metrics["expected_calibration_error"] = self.expected_calibration_error(y_true, y_proba)
            metrics["max_calibration_error"] = self.max_calibration_error(y_true, y_proba)

        # Per-class metrics
        metrics["per_class_f1"] = self._per_class_metrics(y_true, y_pred)
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        return metrics

    @staticmethod
    def mean_reciprocal_rank(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        for true_label, probs in zip(y_true, y_proba):
            ranking = np.argsort(probs)[::-1]
            position = np.where(ranking == true_label)[0]
            if len(position) > 0:
                reciprocal_ranks.append(1.0 / (position[0] + 1))
            else:
                reciprocal_ranks.append(0.0)
        return float(np.mean(reciprocal_ranks))

    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        y_pred = np.argmax(y_proba, axis=1)
        max_probs = np.max(y_proba, axis=1)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(max_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_confidence = np.mean(max_probs[mask])
                bin_accuracy = np.mean(y_pred[mask] == y_true[mask])
                ece += np.abs(bin_confidence - bin_accuracy) * np.sum(mask) / len(y_true)

        return float(ece)

    @staticmethod
    def max_calibration_error(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Maximum Calibration Error"""
        y_pred = np.argmax(y_proba, axis=1)
        max_probs = np.max(y_proba, axis=1)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(max_probs, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        mce = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_confidence = np.mean(max_probs[mask])
                bin_accuracy = np.mean(y_pred[mask] == y_true[mask])
                mce = max(mce, np.abs(bin_confidence - bin_accuracy))

        return float(mce)

    def _per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate per-class F1 scores"""
        f1_scores = {}
        for i in range(min(self.num_classes, 20)):  # Limit to top 20 classes
            mask = y_true == i
            if np.sum(mask) > 0:
                f1 = f1_score(y_true == i, y_pred == i, zero_division=0)
                f1_scores[str(i)] = float(f1)
        return f1_scores

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None, figsize=(12, 10)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        # Limit to top 30 classes for visualization
        if cm.shape[0] > 30:
            cm = cm[:30, :30]
            class_labels = [str(i) for i in range(30)]
        else:
            class_labels = [str(i) for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax)

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix (Showing top {len(class_labels)} classes)')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved confusion matrix to {save_path}")

        return fig

    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                              save_path: Optional[str] = None, n_bins: int = 10):
        """Plot calibration curve"""
        y_pred = np.argmax(y_proba, axis=1)
        max_probs = np.max(y_proba, axis=1)
        is_correct = (y_pred == y_true).astype(int)

        prob_true, prob_pred = calibration_curve(is_correct, max_probs, n_bins=n_bins)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=2)
        ax.plot(prob_pred, prob_true, 's-', label='Model', linewidth=2, markersize=8)
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Calibration Curve', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved calibration curve to {save_path}")

        return fig

    def save_metrics(self, metrics: Dict, save_path: str):
        """Save metrics to JSON"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        metrics_json = self._convert_to_serializable(metrics)
        with open(save_path, 'w') as f:
            json.dump(metrics_json, f, indent=4)
        print(f"✓ Saved metrics to {save_path}")

    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, dict):
            return {k: VQAMetrics._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [VQAMetrics._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj


def print_metrics_summary(metrics: Dict) -> None:
    """Pretty print metrics summary"""
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(f"Accuracy:              {metrics.get('accuracy', 0):.4f}")
    print(f"Balanced Accuracy:     {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"F1 (Weighted):         {metrics.get('f1_weighted', 0):.4f}")
    print(f"F1 (Macro):            {metrics.get('f1_macro', 0):.4f}")

    if 'mean_reciprocal_rank' in metrics:
        print(f"Mean Reciprocal Rank:  {metrics['mean_reciprocal_rank']:.4f}")
    if 'expected_calibration_error' in metrics:
        print(f"Expected Cal. Error:   {metrics['expected_calibration_error']:.4f}")
    if 'max_calibration_error' in metrics:
        print(f"Max Cal. Error:        {metrics['max_calibration_error']:.4f}")
    print("="*60 + "\n")
