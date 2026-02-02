# -*- coding: utf-8 -*-
"""
Loader pour le dataset AffectNet.

AffectNet est le plus grand dataset d'expressions faciales:
- ~450,000 images annotées manuellement
- ~550,000 images annotées automatiquement
- 8 classes: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt
- Images de tailles variées (faces croppées)

Source: http://mohammadmahoor.com/affectnet/
Note: Nécessite une demande d'accès académique
"""

import csv
from pathlib import Path
from typing import List, Optional
import json

from .base import BaseEmotionDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionSample


# Répertoire racine du dataset
AFFECTNET_DIR = Path(__file__).parent / "data" / "affectnet"


class AffectNetDataset(BaseEmotionDataset):
    """
    Loader pour le dataset AffectNet.

    Structure attendue:
    affectnet/
    ├── train/
    │   ├── 0/  (Neutral)
    │   ├── 1/  (Happy)
    │   ├── 2/  (Sad)
    │   ├── 3/  (Surprise)
    │   ├── 4/  (Fear)
    │   ├── 5/  (Disgust)
    │   ├── 6/  (Anger)
    │   └── 7/  (Contempt)
    └── val/
        └── ...
    """

    name = "AffectNet"
    description = "Large-scale Facial Expression Database - 8 classes"
    num_classes = 8

    # Mapping index AffectNet -> EmotionLabel
    INDEX_TO_LABEL = {
        0: EmotionLabel.NEUTRAL,
        1: EmotionLabel.HAPPY,
        2: EmotionLabel.SAD,
        3: EmotionLabel.SURPRISE,
        4: EmotionLabel.FEAR,
        5: EmotionLabel.DISGUST,
        6: EmotionLabel.ANGRY,
        7: EmotionLabel.CONTEMPT,
    }

    def __init__(self, split: str = "val", include_contempt: bool = False):
        """
        Initialise le dataset AffectNet.

        Args:
            split: "train", "val", ou "all"
            include_contempt: Inclure la classe Contempt (8 classes au lieu de 7)
        """
        super().__init__()
        self.split = split
        self.include_contempt = include_contempt
        self._affectnet_dir = AFFECTNET_DIR

        if not include_contempt:
            self.num_classes = 7

    @property
    def data_dir(self) -> Path:
        return self._affectnet_dir

    def is_available(self) -> bool:
        """Vérifie si le dataset est disponible."""
        return (self._affectnet_dir / "train").exists() or (self._affectnet_dir / "val").exists()

    def load_samples(self) -> List[EmotionSample]:
        """Charge les échantillons."""
        samples = []

        # Déterminer les splits à charger
        if self.split == "train":
            splits = ["train"]
        elif self.split == "val":
            splits = ["val"]
        else:  # "all"
            splits = ["train", "val"]

        for split_name in splits:
            split_dir = self._affectnet_dir / split_name

            if not split_dir.exists():
                continue

            # Parcourir les dossiers de classes (0-7)
            for class_idx in range(8):
                # Skip contempt si non inclus
                if class_idx == 7 and not self.include_contempt:
                    continue

                class_dir = split_dir / str(class_idx)
                if not class_dir.exists():
                    continue

                label = self.INDEX_TO_LABEL[class_idx]

                # Charger toutes les images
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        samples.append(EmotionSample(
                            image_path=str(img_path),
                            ground_truth=label
                        ))

        return samples


class AffectNet7Dataset(AffectNetDataset):
    """AffectNet avec 7 classes (sans Contempt) pour compatibilité FER2013."""

    name = "AffectNet-7"
    description = "AffectNet with 7 classes (no Contempt)"

    def __init__(self, split: str = "val"):
        super().__init__(split=split, include_contempt=False)
