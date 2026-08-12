"""
Training Script for Model 1: CNN + BioBERT
Works with existing Task-1 structure and SLAKE dataset
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

# Import from improved modules
from config_improved import TRAINING_CONFIG, DATA_CONFIG, OUTPUT_CONFIG
from metrics_improved import VQAMetrics, print_metrics_summary
from model_cnn_biobert_improved import create_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Model1Trainer:
    """Trainer for CNN + BioBERT model"""

    def __init__(self, model, train_loader, val_loader, test_loader,
                 num_classes, class_names, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.metrics_calculator = VQAMetrics(num_classes, class_names)

        # Checkpoints
        os.makedirs(OUTPUT_CONFIG['checkpoint_dir'], exist_ok=True)
        os.makedirs(OUTPUT_CONFIG['result_dir'], exist_ok=True)

        self.best_val_accuracy = 0
        self.patience = TRAINING_CONFIG['patience']
        self.patience_counter = 0

        logger.info(f"✓ Trainer initialized")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} - Train")

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            input_ids = batch['question_tokens']['input_ids'].squeeze(1).to(self.device)
            attention_mask = batch['question_tokens']['attention_mask'].squeeze(1).to(self.device)
            labels = batch['answer_label'].to(self.device)

            self.optimizer.zero_grad()
            logits, _ = self.model(images, input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'loss': avg_loss,
            'accuracy': (all_preds == all_labels).mean(),
        }

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_proba = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} - Val")

        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['question_tokens']['input_ids'].squeeze(1).to(self.device)
            attention_mask = batch['question_tokens']['attention_mask'].squeeze(1).to(self.device)
            labels = batch['answer_label'].to(self.device)

            logits, _ = self.model(images, input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            proba = torch.softmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_proba.extend(proba)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)

        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics_calculator.calculate_all_metrics(all_labels, all_preds, all_proba)
        metrics['loss'] = avg_loss

        return metrics

    @torch.no_grad()
    def test(self):
        """Test the model"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_proba = []

        pbar = tqdm(self.test_loader, desc="Testing")

        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['question_tokens']['input_ids'].squeeze(1).to(self.device)
            attention_mask = batch['question_tokens']['attention_mask'].squeeze(1).to(self.device)
            labels = batch['answer_label'].to(self.device)

            logits, _ = self.model(images, input_ids, attention_mask)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            proba = torch.softmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_proba.extend(proba)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_proba = np.array(all_proba)

        metrics = self.metrics_calculator.calculate_all_metrics(all_labels, all_preds, all_proba)

        return metrics, all_labels, all_preds, all_proba

    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            OUTPUT_CONFIG['checkpoint_dir'],
            f'model1_epoch_{epoch}_acc_{metrics["accuracy"]:.4f}.pt'
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics,
        }, checkpoint_path)

        logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

    def load_best_checkpoint(self):
        """Load best checkpoint"""
        checkpoints = sorted([f for f in os.listdir(OUTPUT_CONFIG['checkpoint_dir']) if f.endswith('.pt')])
        if checkpoints:
            best_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(OUTPUT_CONFIG['checkpoint_dir'], best_checkpoint)
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict['model_state_dict'])
            logger.info(f"✓ Loaded best checkpoint: {best_checkpoint}")

    def fit(self, num_epochs):
        """Train for multiple epochs"""
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])

            val_metrics = self.validate(epoch)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1_weighted'])

            logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_weighted']:.4f}"
            )

            if val_metrics['accuracy'] > self.best_val_accuracy:
                self.best_val_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        return history


def load_data():
    """Load SLAKE dataset using existing Task-1 structure"""
    import json
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from sklearn.model_selection import train_test_split

    dataset_path = DATA_CONFIG['dataset_path']

    # Load JSON
    json_path = os.path.join(dataset_path, 'train.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    logger.info(f"✓ Loaded {len(all_data)} samples from SLAKE")

    # Build vocabulary
    answers = sorted(set(d['answer'] for d in all_data))
    answer_vocab = {ans: idx for idx, ans in enumerate(answers)}

    logger.info(f"✓ Found {len(answers)} unique answers")

    # Split data
    train_data, temp_data = train_test_split(
        all_data,
        test_size=DATA_CONFIG['test_split'] + DATA_CONFIG['val_split'],
        random_state=DATA_CONFIG['seed']
    )
    val_data, test_data = train_test_split(
        temp_data,
        test_size=DATA_CONFIG['test_split'] / (DATA_CONFIG['test_split'] + DATA_CONFIG['val_split']),
        random_state=DATA_CONFIG['seed']
    )

    logger.info(f"✓ Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Create dataset class
    class SLAKEDataset(Dataset):
        def __init__(self, data, dataset_path, answer_vocab, split='train'):
            self.data = data
            self.dataset_path = dataset_path
            self.answer_vocab = answer_vocab
            self.split = split

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.3) if split == 'train' else transforms.Lambda(lambda x: x),
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

            from model_cnn_biobert_improved import BioBERTTextEncoder
            tokenizer = BioBERTTextEncoder().tokenizer
            question = sample['question']
            tokens = tokenizer(question, max_length=64, padding='max_length', truncation=True, return_tensors='pt')

            answer_label = self.answer_vocab[sample['answer']]

            return {
                'image': image,
                'question': question,
                'question_tokens': {
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask']
                },
                'answer_label': torch.tensor(answer_label, dtype=torch.long),
            }

    # Create datasets and loaders
    train_dataset = SLAKEDataset(train_data, dataset_path, answer_vocab, 'train')
    val_dataset = SLAKEDataset(val_data, dataset_path, answer_vocab, 'val')
    test_dataset = SLAKEDataset(test_data, dataset_path, answer_vocab, 'test')

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader, len(answer_vocab), list(answer_vocab.keys())


def main():
    logger.info("="*60)
    logger.info("MODEL 1: CNN + BioBERT Training")
    logger.info("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("\nLoading SLAKE dataset...")
    train_loader, val_loader, test_loader, num_classes, class_names = load_data()

    # Create model
    logger.info("\nCreating CNN + BioBERT model...")
    model = create_model(num_classes=num_classes, device=device)
    model_info = model.get_model_info()

    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    for key, value in model_info.items():
        print(f"{key:.<35} {value}")
    print("="*60 + "\n")

    # Train
    logger.info("Starting training...\n")
    trainer = Model1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        class_names=class_names,
        device=device
    )

    history = trainer.fit(num_epochs=TRAINING_CONFIG['num_epochs'])

    # Test
    logger.info("\nLoading best model and testing...")
    trainer.load_best_checkpoint()
    test_metrics, test_labels, test_preds, test_proba = trainer.test()

    # Print results
    print_metrics_summary(test_metrics)

    # Save results
    results = {
        'model_name': 'CNN + BioBERT',
        'timestamp': datetime.now().isoformat(),
        'training_history': history,
        'test_metrics': test_metrics,
    }

    results_path = os.path.join(OUTPUT_CONFIG['result_dir'], 'model1_results.json')
    trainer.metrics_calculator.save_metrics(results, results_path)

    # Visualizations
    logger.info("\nGenerating visualizations...")
    confusion_path = os.path.join(OUTPUT_CONFIG['result_dir'], 'model1_confusion_matrix.png')
    trainer.metrics_calculator.plot_confusion_matrix(test_labels, test_preds, confusion_path)

    calibration_path = os.path.join(OUTPUT_CONFIG['result_dir'], 'model1_calibration.png')
    trainer.metrics_calculator.plot_calibration_curve(test_labels, test_proba, calibration_path)

    logger.info("\n✓ Training complete!")


if __name__ == "__main__":
    import torch
    main()
