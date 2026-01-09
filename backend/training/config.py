# -*- coding: utf-8 -*-
"""
Configuration pour l'entrainement du modele de detection d'emotions.
Fine-tuning EfficientNet-B0 sur FER2013 puis L3.
"""

import os
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Dataset paths
FER2013_DIR = DATA_DIR / "fer2013"
L3_DIR = DATA_DIR / "l3"

# Model paths (dynamique selon le backbone choisi)
def get_model_paths(backbone: str = None):
    """Retourne les chemins des modeles selon le backbone."""
    if backbone is None:
        backbone = BACKBONE
    return {
        "fer": MODELS_DIR / f"{backbone}_fer.pth",
        "l3": MODELS_DIR / f"{backbone}_l3.pth"
    }

FER_MODEL_PATH = MODELS_DIR / "efficientnet_b0_fer.pth"  # Default
L3_MODEL_PATH = MODELS_DIR / "efficientnet_b0_l3.pth"  # Default

# Classes FER2013 (7 emotions)
FER_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FER_NUM_CLASSES = len(FER_CLASSES)

# Classes L3 (2 classes: confort thermique)
L3_CLASSES = ["confortable", "inconfortable"]
L3_NUM_CLASSES = len(L3_CLASSES)

# Mapping emotions -> confort (pour creer dataset L3 depuis FER2013)
EMOTION_TO_COMFORT = {
    "happy": "confortable",
    "surprise": "confortable",
    "neutral": "confortable",
    "sad": "inconfortable",
    "angry": "inconfortable",
    "fear": "inconfortable",
    "disgust": "inconfortable",
}

# =============================================================================
# HYPERPARAMETRES ENTRAINEMENT
# =============================================================================

# Image
IMAGE_SIZE = 224  # EfficientNet-B0 et MobileNetV3 attendent 224x224
GRAYSCALE_TO_RGB = True  # FER2013 est en grayscale, convertir en RGB

# Modele backbone
# Options: "efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large"
BACKBONE = "efficientnet_b0"

# Training FER2013
FER_BATCH_SIZE = 32
FER_LEARNING_RATE = 1e-4
FER_EPOCHS = 50
FER_WEIGHT_DECAY = 1e-5

# Training L3 (fine-tuning)
L3_BATCH_SIZE = 16
L3_LEARNING_RATE = 1e-5  # Plus petit car fine-tuning
L3_EPOCHS = 20
L3_WEIGHT_DECAY = 1e-5

# Optimisation
OPTIMIZER = "adamw"  # "adam" ou "adamw"
SCHEDULER = "cosine"  # "cosine", "step", ou "none"
WARMUP_EPOCHS = 5

# Data split
VAL_SPLIT = 0.1  # 10% pour validation
TEST_SPLIT = 0.1  # 10% pour test

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Device
DEVICE = "mps"  # "cuda", "mps" (Apple Silicon), ou "cpu"

# Seed pour reproductibilite
SEED = 42

# Logging
LOG_INTERVAL = 10  # Log toutes les N batches
SAVE_BEST_ONLY = True

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

AUGMENTATION = {
    "horizontal_flip": True,
    "rotation": 15,  # degrees
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.1,
    "random_crop": True,
    "random_erasing": 0.1,  # probability
}

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_device():
    """Retourne le device optimal disponible."""
    import torch
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def ensure_dirs():
    """Cree les dossiers necessaires."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FER2013_DIR.mkdir(parents=True, exist_ok=True)
    L3_DIR.mkdir(parents=True, exist_ok=True)
    (L3_DIR / "confortable").mkdir(parents=True, exist_ok=True)
    (L3_DIR / "inconfortable").mkdir(parents=True, exist_ok=True)
