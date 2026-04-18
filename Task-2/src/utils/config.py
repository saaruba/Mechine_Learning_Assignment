"""
Shared configuration values for Task 2 (Robot Learning with MuJoCo).

Keep this file simple: only global constants that multiple scripts can reuse.
"""

from __future__ import annotations

# Environment settings
DEFAULT_ENV_NAME = "Reacher-v4"
FALLBACK_ENV_NAME = "Reacher-v5"

# Image observation settings
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
USE_GRAYSCALE = False
FRAME_STACK = 4

# Reproducibility / debug defaults
DEFAULT_RANDOM_SEED = 42
DEBUG_TOTAL_TIMESTEPS = 10_000

# Output directory names (inside Task-2/results/)
RESULTS_DIR_NAME = "results"
MODELS_DIR_NAME = "models"
LOGS_DIR_NAME = "logs"
EVAL_DIR_NAME = "evaluations"
PLOTS_DIR_NAME = "plots"
