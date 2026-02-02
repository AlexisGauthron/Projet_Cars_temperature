# -*- coding: utf-8 -*-
"""
Loader pour le dataset ExpW (Expression in-the-Wild).

ExpW contient 91,793 images de visages collectées via Google Image Search
avec des expressions spontanées en conditions réelles.

Structure attendue:
    expw/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── label.lst  # ou annotations.txt

Format label.lst:
    image_name face_id_in_image x y w h expression_label

Téléchargement:
- Kaggle: https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset
- OpenDataLab: https://opendatalab.com/OpenDataLab/Expression_in-the-Wild
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .base import BaseEmotionDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionSample


# Répertoire racine du dataset
EXPW_DIR = Path(__file__).parent / "data" / "expw"


class ExpWDataset(BaseEmotionDataset):
    """
    Loader pour le dataset ExpW (Expression in-the-Wild).

    Dataset large-scale avec des expressions spontanées en conditions réelles.
    """

    name = "ExpW"
    description = "Expression in-the-Wild - 91K spontaneous facial expressions"
    num_classes = 7

    # Mapping ExpW labels -> EmotionLabel
    # 0: angry, 1: disgust, 2: fear, 3: happy, 4: sad, 5: surprise, 6: neutral
    INDEX_TO_LABEL = {
        0: EmotionLabel.ANGRY,
        1: EmotionLabel.DISGUST,
        2: EmotionLabel.FEAR,
        3: EmotionLabel.HAPPY,
        4: EmotionLabel.SAD,
        5: EmotionLabel.SURPRISE,
        6: EmotionLabel.NEUTRAL,
    }

    # Mapping nom de dossier -> EmotionLabel (si structure par dossiers)
    FOLDER_TO_LABEL = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
    }

    def __init__(self):
        """Initialise le dataset ExpW."""
        super().__init__()
        self._expw_dir = EXPW_DIR

    @property
    def data_dir(self) -> Path:
        return self._expw_dir

    @property
    def images_dir(self) -> Path:
        for subdir in ["origin", "images", "image", ""]:
            candidate = self._expw_dir / subdir if subdir else self._expw_dir
            if candidate.exists() and any(candidate.glob("*.jpg")):
                return candidate
        return self._expw_dir / "images"

    @property
    def annotation_file(self) -> Optional[Path]:
        for name in ["label.lst", "label/label.lst", "annotations.txt", "labels.txt"]:
            candidate = self._expw_dir / name
            if candidate.exists():
                return candidate
        return None

    def is_available(self) -> bool:
        """Vérifie si ExpW est disponible."""
        # Format annotations
        if self.annotation_file and self.annotation_file.exists():
            return True

        # Format dossiers
        for label in self.FOLDER_TO_LABEL.keys():
            if (self._expw_dir / label).exists():
                return True

        return False

    def load_samples(self) -> List[EmotionSample]:
        """Charge les échantillons ExpW."""
        # Essayer format dossiers d'abord
        samples = self._load_from_folders()
        if samples:
            return samples

        # Sinon format fichier d'annotations
        return self._load_from_annotation_file()

    def _load_from_folders(self) -> List[EmotionSample]:
        """Charge depuis une structure de dossiers par émotion."""
        samples = []

        for folder_name, label in self.FOLDER_TO_LABEL.items():
            folder_path = self._expw_dir / folder_name

            if not folder_path.exists():
                continue

            for img_path in folder_path.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    samples.append(EmotionSample(
                        image_path=str(img_path),
                        ground_truth=label
                    ))

        return samples

    def _load_from_annotation_file(self) -> List[EmotionSample]:
        """Charge depuis le fichier d'annotations."""
        samples = []

        if not self.annotation_file or not self.annotation_file.exists():
            return samples

        # Parser le fichier d'annotations
        # Format: image_name face_id x y w h confidence expression_label
        # Note: 8 columns, expression_label is the last one (index 7)
        try:
            with open(self.annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 8:
                        continue

                    img_name = parts[0]
                    # face_id = int(parts[1])
                    x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                    # confidence = float(parts[6])
                    label_idx = int(parts[7])  # Expression label is at index 7

                    label = self.INDEX_TO_LABEL.get(label_idx)
                    if label is None:
                        continue

                    # Chercher l'image
                    img_path = self.images_dir / img_name
                    if not img_path.exists():
                        img_path = self._expw_dir / img_name

                    if img_path.exists():
                        samples.append(EmotionSample(
                            image_path=str(img_path),
                            ground_truth=label,
                            face_bbox=(x, y, w, h)
                        ))

        except Exception:
            pass

        return samples

    def get_stats(self) -> dict:
        """Retourne les statistiques avec info sur le type de chargement."""
        stats = super().get_stats()
        stats["annotation_file"] = str(self.annotation_file) if self.annotation_file else None
        return stats
