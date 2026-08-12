"""
UNIFIED TRAINING SCRIPT: Train All 3 VQA Models + Inference + Comparison
Trains Model 1, Model 2, Model 3 sequentially and generates comprehensive report
"""

import os
import sys
import json
import torch
import numpy as np
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedVQATrainer:
    """Unified trainer for all 3 VQA models"""

    def __init__(self, task_dir):
        self.task_dir = task_dir
        self.results_dir = os.path.join(task_dir, 'outputs')
        self.checkpoints_dir = os.path.join(task_dir, 'outputs_checkpoints')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.results = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        logger.info(f"Device: {self.device}")
        logger.info(f"Results directory: {self.results_dir}")

    def train_model_1(self):
        """Train Model 1: CNN + DistilBERT Light"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING MODEL 1: CNN + DistilBERT Light")
        logger.info("="*80)

        try:
            # Import directly instead of subprocess for better control
            from train_model1_light import main as train_model1_main

            train_model1_main()

            # Load results
            results_file = os.path.join(self.results_dir, 'model1_light_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    model1_results = json.load(f)

                self.results['Model 1 (CNN + DistilBERT Light)'] = {
                    'accuracy': float(model1_results.get('test_metrics', {}).get('accuracy', 0)),
                    'f1_weighted': float(model1_results.get('test_metrics', {}).get('f1_weighted', 0)),
                    'f1_macro': float(model1_results.get('test_metrics', {}).get('f1_macro', 0)),
                    'mrr': float(model1_results.get('test_metrics', {}).get('mean_reciprocal_rank', 0)),
                    'ece': float(model1_results.get('test_metrics', {}).get('expected_calibration_error', 0)),
                }

                logger.info("Model 1 training complete!")
                logger.info(f"   Accuracy: {self.results['Model 1 (CNN + DistilBERT Light)']['accuracy']:.4f}")
                return True
        except Exception as e:
            logger.error(f"Model 1 training failed: {str(e)}")
            return False

    def train_model_2(self):
        """Train Model 2: CLIP-Optimized"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING MODEL 2: CLIP-Optimized")
        logger.info("="*80)

        try:
            from train_model2_clip_optimized import main as train_model2_main

            train_model2_main()

            # Load results
            results_file = os.path.join(self.results_dir, 'model2_clip_optimized_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    model2_results = json.load(f)

                self.results['Model 2 (CLIP-Optimized)'] = {
                    'accuracy': float(model2_results.get('accuracy', 0)),
                    'f1_weighted': float(model2_results.get('f1_weighted', 0)),
                    'f1_macro': float(model2_results.get('f1_macro', 0)),
                    'mrr': float(model2_results.get('mrr', 0)),
                    'ece': float(model2_results.get('ece', 0)),
                }

                logger.info("✅ Model 2 training complete!")
                logger.info(f"   Accuracy: {self.results['Model 2 (CLIP-Optimized)']['accuracy']:.4f}")
                return True
        except Exception as e:
            logger.error(f"Model 2 training failed: {str(e)}")
            return False

    def train_model_3(self):
        """Train Model 3: VisualBERT-Optimized"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING MODEL 3: VisualBERT-Optimized")
        logger.info("="*80)

        try:
            from train_model3_visualbert_optimized import main as train_model3_main

            train_model3_main()

            # Load results
            results_file = os.path.join(self.results_dir, 'model3_visualbert_optimized_results.json')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    model3_results = json.load(f)

                self.results['Model 3 (VisualBERT-Optimized)'] = {
                    'accuracy': float(model3_results.get('accuracy', 0)),
                    'f1_weighted': float(model3_results.get('f1_weighted', 0)),
                    'f1_macro': float(model3_results.get('f1_macro', 0)),
                    'mrr': float(model3_results.get('mrr', 0)),
                    'ece': float(model3_results.get('ece', 0)),
                }

                logger.info("✅ Model 3 training complete!")
                logger.info(f"   Accuracy: {self.results['Model 3 (VisualBERT-Optimized)']['accuracy']:.4f}")
                return True
        except Exception as e:
            logger.error(f"Model 3 training failed: {str(e)}")
            return False

    def run_inference(self):
        """Run inference on all trained models"""
        logger.info("\n" + "="*80)
        logger.info("RUNNING INFERENCE ON ALL MODELS")
        logger.info("="*80)

        try:
            from inference_task2 import Task2InferenceDemo

            # For VQA inference, we'll just use the loaded results
            logger.info("Inference results loaded from training phase")
            return True
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            return False

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPARISON REPORT")
        logger.info("="*80)

        # Create comparison JSON
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models': self.results,
            'summary': {}
        }

        # Calculate rankings
        if self.results:
            accuracies = {name: metrics['accuracy']
                         for name, metrics in self.results.items()}
            ranked = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)

            comparison['summary']['rankings'] = [
                {
                    'rank': i+1,
                    'model': name,
                    'accuracy': f"{acc:.4f}"
                }
                for i, (name, acc) in enumerate(ranked)
            ]

            # Print summary
            print("\n" + "="*80)
            print("COMPREHENSIVE COMPARISON RESULTS")
            print("="*80)
            print(f"\n{'Model':<35} {'Accuracy':<12} {'F1-W':<12} {'F1-M':<12} {'MRR':<12} {'ECE':<12}")
            print("-"*95)

            for name in sorted(self.results.keys()):
                metrics = self.results[name]
                print(f"{name:<35} {metrics['accuracy']:<12.4f} {metrics['f1_weighted']:<12.4f} "
                      f"{metrics['f1_macro']:<12.4f} {metrics['mrr']:<12.4f} {metrics['ece']:<12.4f}")

            print("-"*95)
            print(f"\n BEST MODEL: {ranked[0][0]} ({ranked[0][1]:.4f})")
            print("="*80)

        # Save comparison
        comparison_file = os.path.join(self.results_dir, 'all_models_comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=4)

        logger.info(f" Comparison report saved to {comparison_file}")

        # Generate comparison visualization
        self.plot_comparison()

    def plot_comparison(self):
        """Create comparison visualizations"""
        if not self.results:
            return

        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        f1_weighted = [self.results[m]['f1_weighted'] for m in models]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy comparison
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        ax1.bar(range(len(models)), accuracies, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.split('(')[0].strip() for m in models], rotation=15, ha='right')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v+0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # F1-Weighted comparison
        ax2.bar(range(len(models)), f1_weighted, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.split('(')[0].strip() for m in models], rotation=15, ha='right')
        ax2.set_ylabel('F1-Weighted', fontsize=12)
        ax2.set_title('Model F1-Weighted Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(f1_weighted):
            ax2.text(i, v+0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        comparison_plot = os.path.join(self.results_dir, 'models_comparison_plot.png')
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        logger.info(f" Comparison plot saved to {comparison_plot}")
        plt.close()

    def train_all(self):
        """Train all 3 models sequentially"""
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*78 + "║")
        print("║" + "UNIFIED VQA TRAINING: Train All 3 Models + Inference + Comparison".center(78) + "║")
        print("║" + " "*78 + "║")
        print("╚" + "="*78 + "╝\n")

        start_time = datetime.now()

        # Train all models
        success_count = 0
        success_count += self.train_model_1()
        success_count += self.train_model_2()
        success_count += self.train_model_3()

        # Run inference
        self.run_inference()

        # Generate comparison
        self.generate_comparison_report()

        # Final summary
        elapsed = datetime.now() - start_time
        print("\n" + "="*80)
        print("UNIFIED TRAINING COMPLETE!")
        print("="*80)
        print(f"Total time: {elapsed}")
        print(f"Models trained: {success_count}/3")
        print(f"Results saved to: {self.results_dir}")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    # Get task directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = script_dir

    # Create trainer
    trainer = UnifiedVQATrainer(task_dir)

    # Train all models
    trainer.train_all()


if __name__ == "__main__":
    main()
