# -*- coding: utf-8 -*-
"""
Loader pour le dataset FER2013 (Facial Expression Recognition 2013).

FER2013 contient 35,887 images de visages en niveaux de gris 48x48.
- Train: 28,709 images
- Public Test: 3,589 images
- Private Test: 3,589 images

7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

Sources:
- Kaggle: https://www.kaggle.com/datasets/msambare/fer2013
- Format CSV original ou structure de dossiers
"""

import csv
from pathlib import Path
from typing import List, Optional
import numpy as np

from .base import BaseEmotionDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionSample


# Répertoire racine du dataset
FER2013_DIR = Path(__file__).parent / "data" / "fer2013"


class FER2013Dataset(BaseEmotionDataset):
    """
    Loader pour le dataset FER2013.

    Supporte deux formats:
    1. Structure de dossiers: fer2013/train/angry/*.png, fer2013/test/happy/*.png
    2. Fichier CSV original: fer2013.csv avec colonnes emotion, pixels, Usage
    """

    name = "FER2013"
    description = "Facial Expression Recognition 2013 - 48x48 grayscale faces"
    num_classes = 7

    # Mapping index -> EmotionLabel (ordre FER2013)
    INDEX_TO_LABEL = {
        0: EmotionLabel.ANGRY,
        1: EmotionLabel.DISGUST,
        2: EmotionLabel.FEAR,
        3: EmotionLabel.HAPPY,
        4: EmotionLabel.SAD,
        5: EmotionLabel.SURPRISE,
        6: EmotionLabel.NEUTRAL,
    }

    # Mapping nom de dossier -> EmotionLabel
    FOLDER_TO_LABEL = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
    }

    def __init__(self, split: str = "test"):
        """
        Initialise le dataset FER2013.

        Args:
            split: "train", "test", ou "all"
        """
        super().__init__()
        self.split = split
        self._fer2013_dir = FER2013_DIR

    @property
    def data_dir(self) -> Path:
        return self._fer2013_dir

    def is_available(self) -> bool:
        """Vérifie si le dataset est disponible."""
        # Vérifier format dossiers
        if (self._fer2013_dir / "train").exists() or (self._fer2013_dir / "test").exists():
            return True

        # Vérifier format CSV
        if (self._fer2013_dir / "fer2013.csv").exists():
            return True

        return False

    def load_samples(self) -> List[EmotionSample]:
        """Charge les échantillons selon le format détecté."""
        # Essayer format dossiers d'abord
        if (self._fer2013_dir / "train").exists() or (self._fer2013_dir / "test").exists():
            return self._load_from_folders()

        # Sinon format CSV
        if (self._fer2013_dir / "fer2013.csv").exists():
            return self._load_from_csv()

        return []

    def _load_from_folders(self) -> List[EmotionSample]:
        """Charge depuis la structure de dossiers."""
        samples = []

        # Déterminer les dossiers à charger
        if self.split == "train":
            splits = ["train"]
        elif self.split == "test":
            splits = ["test"]
        else:  # "all"
            splits = ["train", "test"]

        for split_name in splits:
            split_dir = self._fer2013_dir / split_name

            if not split_dir.exists():
                continue

            # Parcourir les dossiers d'émotions
            for emotion_folder in split_dir.iterdir():
                if not emotion_folder.is_dir():
                    continue

                folder_name = emotion_folder.name.lower()
                if folder_name not in self.FOLDER_TO_LABEL:
                    continue

                label = self.FOLDER_TO_LABEL[folder_name]

                # Charger toutes les images du dossier
                for img_path in emotion_folder.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        samples.append(EmotionSample(
                            image_path=str(img_path),
                            ground_truth=label
                        ))

        return samples

    def _load_from_csv(self) -> List[EmotionSample]:
        """Charge depuis le fichier CSV original."""
        samples = []
        csv_path = self._fer2013_dir / "fer2013.csv"

        # Créer un dossier pour les images extraites
        images_dir = self._fer2013_dir / "images"
        images_dir.mkdir(exist_ok=True)

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                # Filtrer par split
                usage = row.get("Usage", "").lower()
                if self.split == "train" and usage != "training":
                    continue
                if self.split == "test" and usage not in ["publictest", "privatetest"]:
                    continue

                # Parser l'émotion
                emotion_idx = int(row.get("emotion", 0))
                label = self.INDEX_TO_LABEL.get(emotion_idx, EmotionLabel.NEUTRAL)

                # Parser les pixels et sauvegarder l'image
                pixels = row.get("pixels", "")
                if pixels:
                    img_path = images_dir / f"{idx}_{label.name.lower()}.png"

                    if not img_path.exists():
                        # Convertir les pixels en image
                        pixel_values = [int(p) for p in pixels.split()]
                        img_array = np.array(pixel_values, dtype=np.uint8).reshape(48, 48)

                        import cv2
                        cv2.imwrite(str(img_path), img_array)

                    samples.append(EmotionSample(
                        image_path=str(img_path),
                        ground_truth=label
                    ))

        return samples


class FER2013PlusDataset(FER2013Dataset):
    """
    FER2013+ avec labels améliorés (annotations multiples par image).
    """

    name = "FER2013+"
    description = "FER2013 with improved labels from crowd-sourcing"

    def __init__(self, split: str = "test"):
        super().__init__(split)
        # FER2013+ utilise des annotations majoritaires
        # TODO: Implémenter le chargement des labels FER2013+
