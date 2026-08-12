"""
Improved Configuration for Task 1: Medical Visual Question Answering
Works with existing Task-1 structure
"""

import os

# Get the Task-1 directory
TASK1_DIR = os.path.dirname(os.path.abspath(__file__))

# Data configuration
DATA_CONFIG = {
    "dataset_path": os.path.join(TASK1_DIR, "data/Slake/Slake1.0"),
    "image_size": 224,
    "test_split": 0.2,
    "val_split": 0.1,
    "seed": 42,
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "num_epochs": 25,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 5,  # Early stopping patience
    "device": "cuda",  # or "cpu"
}

# Model configurations for 3 different approaches
MODEL_CONFIGS = {
    "model1_cnn_biobert": {
        "name": "CNN + BioBERT (Model 1)",
        "description": "ResNet50 image encoder + BioBERT text encoder with attention fusion",
        "image_encoder": "resnet50",
        "text_encoder": "dmis-lab/biobert-v1.1",
        "hidden_dim": 512,
        "fusion_type": "attention",
    },
    "model2_clip": {
        "name": "CLIP Fine-tuned (Model 2)",
        "description": "Vision-Language model fine-tuned on SLAKE",
        "model_id": "openai/clip-vit-base-patch32",
        "hidden_dim": 512,
    },
    "model3_visualbert": {
        "name": "VisualBERT (Model 3)",
        "description": "End-to-end transformer for joint vision-language understanding",
        "model_id": "uclanlp/visualbert-vqa",
        "hidden_dim": 768,
    },
}

# Evaluation metrics to compute
EVALUATION_METRICS = [
    "accuracy",
    "f1_weighted",
    "balanced_accuracy",
    "mean_reciprocal_rank",  # MRR
    "expected_calibration_error",  # ECE
    "confusion_matrix",
]

# Output configuration
OUTPUT_CONFIG = {
    "checkpoint_dir": os.path.join(TASK1_DIR, "outputs_checkpoints"),
    "log_dir": os.path.join(TASK1_DIR, "logs"),
    "result_dir": os.path.join(TASK1_DIR, "outputs"),
    "save_best_only": True,
    "save_frequency": 5,
}

# Text preprocessing
TEXT_CONFIG = {
    "max_length": 64,
    "tokenizer": "biobert",
    "lowercase": True,
}

# Image preprocessing
IMAGE_CONFIG = {
    "resize_size": 224,
    "center_crop": True,
    "normalize": True,
    "augmentation": True,
    "augmentation_strength": 0.3,
}

# Ablation study configuration
ABLATION_CONFIG = {
    "test_image_only": True,
    "test_text_only": True,
    "test_fusion_only": True,
}

# Verify dataset exists
if not os.path.exists(DATA_CONFIG["dataset_path"]):
    raise FileNotFoundError(f"Dataset not found at {DATA_CONFIG['dataset_path']}")

print(f"✓ Dataset found at: {DATA_CONFIG['dataset_path']}")
print(f"✓ Output directory: {OUTPUT_CONFIG['checkpoint_dir']}")
