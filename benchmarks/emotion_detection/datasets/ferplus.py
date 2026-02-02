# -*- coding: utf-8 -*-
"""
Loader pour le dataset FER+ (FERPlus - Improved FER2013 Labels).

FER+ améliore FER2013 avec 10 annotateurs par image au lieu d'un seul.
Cela permet d'avoir une distribution de probabilités pour chaque image.

Structure attendue:
    ferplus/
    ├── fer2013.csv           # Images FER2013 originales
    └── fer2013new.csv        # Labels FER+ (10 votes par image)

Ou structure de dossiers:
    ferplus/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   └── ...
    └── test/
        ├── angry/
        └── ...

Téléchargement:
- Labels FER+: https://github.com/microsoft/FERPlus
- Images FER2013: https://www.kaggle.com/datasets/msambare/fer2013
"""

import csv
from pathlib import Path
from typing import List, Optional
import numpy as np

from .base import BaseEmotionDataset
from .fer2013 import FER2013Dataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionSample


# Répertoire racine du dataset
FERPLUS_DIR = Path(__file__).parent / "data" / "ferplus"


class FERPlusDataset(BaseEmotionDataset):
    """
    Loader pour le dataset FER+ (FERPlus).

    FER+ offre des labels améliorés avec vote majoritaire de 10 annotateurs.
    Supporte aussi les soft labels (distribution de probabilités).
    """

    name = "FERPlus"
    description = "FER2013 with improved labels (10 annotators per image)"
    num_classes = 8  # Inclut Contempt

    # Colonnes FER+ dans l'ordre
    # neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF
    FERPLUS_COLS = [
        EmotionLabel.NEUTRAL,
        EmotionLabel.HAPPY,
        EmotionLabel.SURPRISE,
        EmotionLabel.SAD,
        EmotionLabel.ANGRY,
        EmotionLabel.DISGUST,
        EmotionLabel.FEAR,
        EmotionLabel.CONTEMPT,
        None,  # unknown
        None,  # NF (Not a Face)
    ]

    # Mapping nom de dossier -> EmotionLabel
    FOLDER_TO_LABEL = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
        "contempt": EmotionLabel.CONTEMPT,
    }

    def __init__(self, split: str = "test", voting: str = "majority"):
        """
        Initialise le dataset FER+.

        Args:
            split: "train", "test", ou "all"
            voting: "majority" (label dominant) ou "probability" (soft labels)
        """
        super().__init__()
        self.split = split
        self.voting = voting
        self._ferplus_dir = FERPLUS_DIR

    @property
    def data_dir(self) -> Path:
        return self._ferplus_dir

    def is_available(self) -> bool:
        """Vérifie si FER+ est disponible."""
        # Vérifier format dossiers
        if (self._ferplus_dir / "train").exists() or (self._ferplus_dir / "test").exists():
            return True

        # Vérifier format CSV
        if (self._ferplus_dir / "fer2013new.csv").exists():
            return True

        return False

    def load_samples(self) -> List[EmotionSample]:
        """Charge les échantillons FER+."""
        # Essayer format dossiers d'abord
        if (self._ferplus_dir / "train").exists() or (self._ferplus_dir / "test").exists():
            return self._load_from_folders()

        # Sinon format CSV
        if (self._ferplus_dir / "fer2013new.csv").exists():
            return self._load_from_csv()

        return []

    def _load_from_folders(self) -> List[EmotionSample]:
        """Charge depuis la structure de dossiers."""
        samples = []

        # Déterminer les dossiers à charger
        if self.split == "train":
            splits = ["train", "Training"]
        elif self.split == "test":
            splits = ["test", "PublicTest", "PrivateTest"]
        else:  # "all"
            splits = ["train", "test", "Training", "PublicTest", "PrivateTest"]

        for split_name in splits:
            split_dir = self._ferplus_dir / split_name

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
        """Charge depuis les fichiers CSV (FER2013 + FER+ labels)."""
        samples = []

        fer_csv = self._ferplus_dir / "fer2013.csv"
        ferplus_csv = self._ferplus_dir / "fer2013new.csv"

        if not fer_csv.exists() or not ferplus_csv.exists():
            return samples

        # Créer un dossier pour les images
        images_dir = self._ferplus_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Charger les labels FER+
        ferplus_labels = []
        with open(ferplus_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 10:
                    # Votes: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF
                    votes = [int(v) if v.strip().isdigit() else 0 for v in row[:10]]
                    ferplus_labels.append(votes)

        # Charger les images FER2013
        with open(fer_csv, 'r') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                if idx >= len(ferplus_labels):
                    break

                # Filtrer par split
                usage = row.get("Usage", "").lower()
                if self.split == "train" and usage != "training":
                    continue
                if self.split == "test" and usage not in ["publictest", "privatetest"]:
                    continue

                # Obtenir le label FER+
                votes = ferplus_labels[idx]
                label = self._get_label_from_votes(votes)

                if label is None:
                    continue

                # Parser les pixels et sauvegarder l'image
                pixels = row.get("pixels", "")
                if pixels:
                    img_path = images_dir / f"{idx}_{label.name.lower()}.png"

                    if not img_path.exists():
                        import cv2
                        pixel_values = [int(p) for p in pixels.split()]
                        img_array = np.array(pixel_values, dtype=np.uint8).reshape(48, 48)
                        cv2.imwrite(str(img_path), img_array)

                    samples.append(EmotionSample(
                        image_path=str(img_path),
                        ground_truth=label
                    ))

        return samples

    def _get_label_from_votes(self, votes: List[int]) -> Optional[EmotionLabel]:
        """Détermine le label à partir des votes."""
        if len(votes) < 8:
            return None

        # Ignorer unknown (8) et NF (9)
        emotion_votes = votes[:8]

        # Vote majoritaire
        max_votes = max(emotion_votes)
        if max_votes == 0:
            return None

        max_idx = emotion_votes.index(max_votes)
        return self.FERPLUS_COLS[max_idx]


class FERPlusSoftLabelDataset(FERPlusDataset):
    """FER+ avec soft labels (distribution de probabilités)."""

    name = "FERPlus-Soft"
    description = "FERPlus with probability distribution labels"

    def __init__(self, split: str = "test"):
        super().__init__(split=split, voting="probability")
