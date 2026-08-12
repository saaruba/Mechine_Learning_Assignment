"""
Training Script for Model 3: VisualBERT-Optimized
Enhanced training configuration for improved medical VQA performance
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config_models_2_3_fast import TRAINING_CONFIG_VISUALBERT, DATA_CONFIG, OUTPUT_CONFIG
from model_visualbert_fast import create_visualbert_fast_model
from metrics_improved import VQAMetrics, print_metrics_summary

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SLAKEDatasetVisualBERT(Dataset):
    """SLAKE dataset for VisualBERT model with augmentation"""

    def __init__(self, data, dataset_path, answer_vocab, split='train'):
        self.data = data
        self.dataset_path = dataset_path
        self.answer_vocab = answer_vocab
        self.split = split

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.dataset_path, 'imgs', sample['img_name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        question = sample['question']
        tokens = self.tokenizer(
            question,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        answer_label = self.answer_vocab[sample['answer']]

        return {
            'image': image,
            'question': question,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'answer_label': torch.tensor(answer_label, dtype=torch.long),
        }


class VisualBERTTrainer:
    """Trainer for VisualBERT model with enhanced optimization"""

    def __init__(self, model, train_loader, val_loader, test_loader, num_classes, class_names, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

        # Enhanced optimizer with warmup
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=2e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0
        self.best_checkpoint_path = None
        self.patience = 5
        self.patience_counter = 0
        self.metrics_calculator = VQAMetrics(num_classes)

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Model 3 Epoch {epoch} - Train")
        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['answer_label'].to(self.device)

            self.optimizer.zero_grad()

            logits, _ = self.model(images, input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': accuracy
        }

    def validate(self):
        """Validate on validation set"""
        from sklearn.metrics import f1_score
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc=f"Model 3 Epoch - Val")
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['answer_label'].to(self.device)

                logits, _ = self.model(images, input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = np.mean(all_preds == all_labels)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'f1': f1_weighted
        }

    def fit(self, num_epochs=15):
        """Train model for specified epochs"""
        history = {'train': [], 'val': []}

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Save checkpoint if validation accuracy improved
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
                checkpoint_path = os.path.join(
                    OUTPUT_CONFIG['checkpoint_dir'],
                    f"model3_visualbert_optimized_epoch_{epoch}_acc_{val_metrics['accuracy']:.4f}.pt"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                self.best_checkpoint_path = checkpoint_path
                logger.info(f"[SAVE] Checkpoint saved: {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            self.scheduler.step()
            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

        return history

    def test(self):
        """Evaluate on test set"""
        from sklearn.metrics import f1_score
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.test_loader, desc="Model 3 Testing")
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['answer_label'].to(self.device)

                logits, _ = self.model(images, input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)

                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = np.mean(all_preds == all_labels)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        mrr = self.metrics_calculator.mean_reciprocal_rank(all_labels, all_probs)
        ece = self.metrics_calculator.expected_calibration_error(all_labels, all_probs)

        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'mrr': mrr,
            'ece': ece,
            'preds': all_preds,
            'labels': all_labels,
            'probs': all_probs
        }


def main():
    """Main training pipeline"""

    print("""
============================================================
MODEL 3: VISUALBERT-OPTIMIZED TRAINING
Enhanced configuration for improved medical VQA performance
============================================================
    """)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info("")

    # Load SLAKE dataset
    logger.info("Loading SLAKE dataset...")
    dataset_path = DATA_CONFIG['dataset_path']
    qa_pairs = json.load(open(os.path.join(dataset_path, 'train.json'), encoding='utf-8'))

    answer_set = set([item['answer'] for item in qa_pairs])
    answer_vocab = {ans: idx for idx, ans in enumerate(sorted(answer_set))}
    num_classes = len(answer_vocab)
    logger.info(f"[LOAD] Loaded {len(qa_pairs)} samples from SLAKE")
    logger.info(f"[LOAD] Found {num_classes} unique answers")

    # Split dataset
    train_data, temp_data = train_test_split(qa_pairs, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    logger.info(f"[SPLIT] train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Create datasets and loaders
    train_dataset = SLAKEDatasetVisualBERT(train_data, dataset_path, answer_vocab, split='train')
    val_dataset = SLAKEDatasetVisualBERT(val_data, dataset_path, answer_vocab, split='val')
    test_dataset = SLAKEDatasetVisualBERT(test_data, dataset_path, answer_vocab, split='test')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Create model
    logger.info("")
    logger.info("Creating VisualBERT-Optimized model...")
    model = create_visualbert_fast_model(num_classes=num_classes, device=device)

    # Print model info
    model_info = model.get_model_info()
    print("\n" + "="*60)
    print("MODEL 3: VISUALBERT-OPTIMIZED ARCHITECTURE")
    print("="*60)
    for key, value in model_info.items():
        print(f"{key:.<40} {value}")
    print("="*60)

    # Train
    logger.info("Starting training...")
    trainer = VisualBERTTrainer(model, train_loader, val_loader, test_loader, num_classes, list(answer_vocab.keys()), device=device)
    history = trainer.fit(num_epochs=15)

    # Test
    logger.info("")
    logger.info("Loading best model and testing...")
    logger.info(f"[LOAD] Loaded best checkpoint: {os.path.basename(trainer.best_checkpoint_path)}")
    model.load_state_dict(torch.load(trainer.best_checkpoint_path))
    test_metrics = trainer.test()

    # Print results
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(f"Accuracy:              {test_metrics['accuracy']:.4f}")
    print(f"F1 (Weighted):         {test_metrics['f1_weighted']:.4f}")
    print(f"F1 (Macro):            {test_metrics['f1_macro']:.4f}")
    print(f"Mean Reciprocal Rank:  {test_metrics['mrr']:.4f}")
    print(f"Expected Cal. Error:   {test_metrics['ece']:.4f}")
    print("="*60)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'VisualBERT-Optimized',
        'accuracy': float(test_metrics['accuracy']),
        'f1_weighted': float(test_metrics['f1_weighted']),
        'f1_macro': float(test_metrics['f1_macro']),
        'mrr': float(test_metrics['mrr']),
        'ece': float(test_metrics['ece']),
        'num_classes': num_classes,
        'test_size': len(test_data)
    }

    results_path = os.path.join(output_dir, 'model3_visualbert_optimized_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"[SAVE] Results saved to {results_path}")

    # Generate visualizations
    logger.info("")
    logger.info("Generating visualizations...")
    cm_path = os.path.join(output_dir, 'model3_visualbert_optimized_confusion_matrix.png')
    trainer.metrics_calculator.plot_confusion_matrix(test_metrics['labels'], test_metrics['preds'], cm_path)
    logger.info(f"[SAVE] Confusion matrix saved to {cm_path}")

    calibration_path = os.path.join(output_dir, 'model3_visualbert_optimized_calibration.png')
    trainer.metrics_calculator.plot_calibration_curve(test_metrics['labels'], test_metrics['probs'], calibration_path)
    logger.info(f"[SAVE] Calibration curve saved to {calibration_path}")

    logger.info("")
    logger.info("Model 3 (VisualBERT-Optimized) training complete!")


if __name__ == "__main__":
    main()
