# -*- coding: utf-8 -*-
"""
Configuration du benchmark de classification d'émotions.
"""

from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets" / "data"
RESULTS_DIR = BASE_DIR / "results"

# Paramètres par défaut
DEFAULT_DATASET = "fer2013"
DEFAULT_LIMIT = None  # None = toutes les images
DEFAULT_WARMUP = 5

# Classifieurs à tester par défaut
DEFAULT_CLASSIFIERS = [
    "deepface",
    "hsemotion",
    "fer_pytorch",
    "rmn",
]

# Labels d'émotions (FER2013)
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Couleurs pour les visualisations
EMOTION_COLORS = {
    "angry": "#FF6B6B",
    "disgust": "#9B59B6",
    "fear": "#3498DB",
    "happy": "#F1C40F",
    "sad": "#1ABC9C",
    "surprise": "#E67E22",
    "neutral": "#95A5A6",
}
