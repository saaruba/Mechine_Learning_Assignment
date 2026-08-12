"""
FAST Configuration for Models 2 & 3 (CLIP + VisualBERT)
Optimized for quick training: ~8-10 hours total for both models
"""

import os

TASK1_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# FAST TRAINING CONFIG (Optimized for Speed)
# ============================================================

DATA_CONFIG = {
    "dataset_path": os.path.join(TASK1_DIR, "data/Slake/Slake1.0"),
    "image_size": 224,
    "test_split": 0.2,
    "val_split": 0.1,
    "seed": 42,
}

# OPTIMIZED FOR SPEED
TRAINING_CONFIG_CLIP = {
    "batch_size": 32,  # Larger batch = faster training
    "num_epochs": 5,   # REDUCED from 25 (4x faster!)
    "learning_rate": 2e-5,
    "weight_decay": 1e-5,
    "patience": 3,     # Early stopping
    "device": "cuda",
    "warmup_steps": 100,
    "gradient_accumulation_steps": 1,
}

TRAINING_CONFIG_VISUALBERT = {
    "batch_size": 32,  # Larger batch
    "num_epochs": 8,   # REDUCED from 25
    "learning_rate": 5e-5,
    "weight_decay": 1e-5,
    "patience": 3,
    "device": "cuda",
    "warmup_steps": 200,
    "gradient_accumulation_steps": 1,
}

# Model configurations - LIGHTWEIGHT VERSIONS
MODEL_CONFIGS_FAST = {
    "model2_clip_fast": {
        "name": "CLIP-Fast (Model 2)",
        "description": "Lightweight CLIP fine-tuned on SLAKE (5 epochs)",
        "model_id": "openai/clip-vit-base-patch32",  # Smaller than ViT-L
        "freeze_vision_encoder": False,  # Fine-tune vision
        "freeze_text_encoder": False,    # Fine-tune text
        "hidden_dim": 512,
        "num_classes": None,  # Will be computed from data
    },

    "model3_visualbert_fast": {
        "name": "VisualBERT-Fast (Model 3)",
        "description": "Lightweight VisualBERT fine-tuned on SLAKE (8 epochs)",
        "model_id": "uclanlp/visualbert-vqa-coco-pre",  # Pre-trained, faster to fine-tune
        "freeze_vision_encoder": True,   # Keep vision frozen (faster)
        "freeze_text_encoder": False,    # Fine-tune text encoder
        "hidden_dim": 768,
        "num_classes": None,
    },
}

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy",
    "f1_weighted",
    "balanced_accuracy",
    "mean_reciprocal_rank",
    "expected_calibration_error",
]

# Output configuration
OUTPUT_CONFIG = {
    "checkpoint_dir": os.path.join(TASK1_DIR, "outputs_checkpoints"),
    "log_dir": os.path.join(TASK1_DIR, "logs"),
    "result_dir": os.path.join(TASK1_DIR, "outputs"),
    "save_best_only": True,
    "save_frequency": 1,  # Save every epoch
}

# MINIMAL augmentation (faster training)
IMAGE_CONFIG = {
    "resize_size": 224,
    "center_crop": False,  # No center crop = faster
    "normalize": True,
    "augmentation": False,  # NO augmentation = faster
}

TEXT_CONFIG = {
    "max_length": 64,
    "tokenizer": "bert-base-uncased",
    "lowercase": True,
}

# Verify dataset exists
if not os.path.exists(DATA_CONFIG["dataset_path"]):
    raise FileNotFoundError(f"Dataset not found at {DATA_CONFIG['dataset_path']}")

print("""
╔════════════════════════════════════════════════════════════╗
║  MODELS 2 & 3: FAST TRAINING CONFIGURATION                ║
║  Optimized for 8-10 hours total completion                ║
╚════════════════════════════════════════════════════════════╝

Model 2 (CLIP-Fast): 5 epochs, batch_size=32, ~4-5 hours
Model 3 (VisualBERT-Fast): 8 epochs, batch_size=32, ~4-5 hours

Total estimated time: 8-10 hours on GPU
""")
