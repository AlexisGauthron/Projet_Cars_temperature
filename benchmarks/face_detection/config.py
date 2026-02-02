# -*- coding: utf-8 -*-
"""
Configuration globale du benchmark de détection de visage.
"""

from pathlib import Path

# Chemins de base
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
DATASETS_DIR = ROOT_DIR / "datasets" / "data"
ANNOTATIONS_DIR = ROOT_DIR / "annotations"
RESULTS_DIR = ROOT_DIR / "results"

# Datasets supportés avec leurs configurations
DATASETS_CONFIG = {
    "wider_face": {
        "name": "WIDER FACE",
        "annotation_file": ANNOTATIONS_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt",
        "images_dir": DATASETS_DIR / "wider_face",
        "description": "Dataset de référence pour la détection de visage (3226 images)",
    },
    # Futurs datasets à ajouter ici
    # "fddb": {
    #     "name": "FDDB",
    #     "annotation_file": ANNOTATIONS_DIR / "fddb" / "annotations.txt",
    #     "images_dir": DATASETS_DIR / "fddb",
    #     "description": "Face Detection Data Set and Benchmark",
    # },
}

# Détecteurs lents (exclus en mode --fast)
SLOW_DETECTORS = [
    "MTCNN", "RetinaFace", "DLib-CNN", "SCRFD_34G", "SCRFD_10G",
    "DSFD", "TinaFace"  # Haute précision mais plus lents
]

# Seuil IoU par défaut
DEFAULT_IOU_THRESHOLD = 0.5

# Configuration du mode benchmark strict
STRICT_WARMUP_IMAGES = 10      # Nombre d'images de warmup (ignorées)
STRICT_NUM_PASSES = 3          # Nombre de passes par image
STRICT_CLEAR_GPU_MEMORY = True # Libérer la mémoire GPU entre modèles
