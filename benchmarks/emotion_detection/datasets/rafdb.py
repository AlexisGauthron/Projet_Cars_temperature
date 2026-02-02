# -*- coding: utf-8 -*-
"""
Loader pour le dataset RAF-DB (Real-world Affective Faces Database).

RAF-DB contient ~30,000 images de visages avec expressions naturelles.
- Images collectées depuis Internet
- Annotations par crowd-sourcing (40 annotateurs par image)
- 7 classes de base + 12 expressions composées

Source: http://www.whdeng.cn/raf/model1.html
"""

from pathlib import Path
from typing import List, Dict
import re

from .base import BaseEmotionDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionSample


# Répertoire racine du dataset
RAFDB_DIR = Path(__file__).parent / "data" / "rafdb"


class RAFDBDataset(BaseEmotionDataset):
    """
    Loader pour le dataset RAF-DB.

    Structure attendue:
    rafdb/
    ├── train/
    │   ├── 1/  (Surprise)
    │   ├── 2/  (Fear)
    │   ├── 3/  (Disgust)
    │   ├── 4/  (Happiness)
    │   ├── 5/  (Sadness)
    │   ├── 6/  (Anger)
    │   └── 7/  (Neutral)
    ├── test/
    │   └── ...
    └── list_patition_label.txt  (optionnel)
    """

    name = "RAF-DB"
    description = "Real-world Affective Faces Database"
    num_classes = 7

    # Mapping RAF-DB -> EmotionLabel (attention: ordre différent de FER2013!)
    INDEX_TO_LABEL = {
        1: EmotionLabel.SURPRISE,
        2: EmotionLabel.FEAR,
        3: EmotionLabel.DISGUST,
        4: EmotionLabel.HAPPY,
        5: EmotionLabel.SAD,
        6: EmotionLabel.ANGRY,
        7: EmotionLabel.NEUTRAL,
    }

    def __init__(self, split: str = "test"):
        """
        Initialise le dataset RAF-DB.

        Args:
            split: "train", "test", ou "all"
        """
        super().__init__()
        self.split = split
        self._rafdb_dir = RAFDB_DIR

    @property
    def data_dir(self) -> Path:
        return self._rafdb_dir

    def is_available(self) -> bool:
        """Vérifie si le dataset est disponible."""
        return (self._rafdb_dir / "train").exists() or (self._rafdb_dir / "test").exists()

    def load_samples(self) -> List[EmotionSample]:
        """Charge les échantillons."""
        samples = []

        # Essayer de charger depuis le fichier de labels
        label_file = self._rafdb_dir / "list_patition_label.txt"
        if label_file.exists():
            samples = self._load_from_label_file(label_file)
            if samples:
                return samples

        # Sinon charger depuis la structure de dossiers
        return self._load_from_folders()

    def _load_from_label_file(self, label_file: Path) -> List[EmotionSample]:
        """Charge depuis le fichier de labels."""
        samples = []

        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: "train_00001_aligned.jpg 1" ou "test_0001.jpg 3"
                parts = line.split()
                if len(parts) != 2:
                    continue

                filename, label_idx = parts
                label_idx = int(label_idx)

                # Déterminer le split depuis le nom du fichier
                if filename.startswith("train"):
                    file_split = "train"
                else:
                    file_split = "test"

                # Filtrer par split
                if self.split != "all" and self.split != file_split:
                    continue

                # Chercher l'image
                img_path = self._rafdb_dir / file_split / filename
                if not img_path.exists():
                    # Essayer avec le dossier de classe
                    img_path = self._rafdb_dir / file_split / str(label_idx) / filename

                if img_path.exists():
                    label = self.INDEX_TO_LABEL.get(label_idx, EmotionLabel.NEUTRAL)
                    samples.append(EmotionSample(
                        image_path=str(img_path),
                        ground_truth=label
                    ))

        return samples

    def _load_from_folders(self) -> List[EmotionSample]:
        """Charge depuis la structure de dossiers."""
        samples = []

        # Déterminer les splits à charger
        if self.split == "train":
            splits = ["train"]
        elif self.split == "test":
            splits = ["test"]
        else:
            splits = ["train", "test"]

        for split_name in splits:
            split_dir = self._rafdb_dir / split_name

            if not split_dir.exists():
                continue

            # Parcourir les dossiers de classes (1-7)
            for class_idx in range(1, 8):
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
