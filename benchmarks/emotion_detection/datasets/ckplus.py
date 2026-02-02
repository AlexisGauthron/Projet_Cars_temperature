# -*- coding: utf-8 -*-
"""
Loader pour le dataset CK+ (Extended Cohn-Kanade).

CK+ contient 593 séquences vidéo de 123 sujets avec 327 séquences labellisées.
Les images montrent la progression d'une expression neutre vers l'apex.

Supporte deux formats:
1. Format CSV (Kaggle): ckextended.csv avec colonnes emotion, pixels, Usage
2. Format dossiers: images/ et labels/ avec structure hiérarchique

Téléchargement:
- Kaggle: https://www.kaggle.com/datasets/davilsena/ckdataset
- Zenodo: https://zenodo.org/records/11221351
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
CKPLUS_DIR = Path(__file__).parent / "data" / "ckplus"


class CKPlusDataset(BaseEmotionDataset):
    """
    Loader pour le dataset CK+ (Extended Cohn-Kanade).

    Supporte:
    - Format CSV (ckextended.csv) - Kaggle download
    - Format dossiers (images/ + labels/) - Original format
    """

    name = "CK+"
    description = "Extended Cohn-Kanade - Lab-controlled facial expressions"
    num_classes = 7

    # Mapping CK+ labels (fichier txt contient un entier)
    # 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise
    INDEX_TO_LABEL = {
        0: EmotionLabel.NEUTRAL,
        1: EmotionLabel.ANGRY,
        2: None,  # Contempt - pas dans FER2013 standard
        3: EmotionLabel.DISGUST,
        4: EmotionLabel.FEAR,
        5: EmotionLabel.HAPPY,
        6: EmotionLabel.SAD,
        7: EmotionLabel.SURPRISE,
    }

    def __init__(self, apex_only: bool = True):
        """
        Initialise le dataset CK+.

        Args:
            apex_only: Si True, uniquement les images apex (dernière frame)
        """
        super().__init__()
        self.apex_only = apex_only
        self._ckplus_dir = CKPLUS_DIR

    @property
    def data_dir(self) -> Path:
        return self._ckplus_dir

    @property
    def images_dir(self) -> Path:
        # Supporte plusieurs structures de dossier
        for subdir in ["cohn-kanade-images", "images", "Emotion", ""]:
            candidate = self._ckplus_dir / subdir if subdir else self._ckplus_dir
            if candidate.exists() and any(candidate.glob("S*")):
                return candidate
        return self._ckplus_dir / "images"

    @property
    def labels_dir(self) -> Path:
        for subdir in ["Emotion", "labels", "emotion"]:
            candidate = self._ckplus_dir / subdir
            if candidate.exists():
                return candidate
        return self._ckplus_dir / "labels"

    @property
    def csv_file(self) -> Path:
        """Chemin vers le fichier CSV (format Kaggle)."""
        return self._ckplus_dir / "ckextended.csv"

    def is_available(self) -> bool:
        """Vérifie si CK+ est disponible (CSV ou dossiers)."""
        # Format CSV (Kaggle)
        if self.csv_file.exists():
            return True
        # Format dossiers (original)
        return self.images_dir.exists() and any(self.images_dir.glob("S*"))

    def load_samples(self) -> List[EmotionSample]:
        """Charge les échantillons CK+."""
        # Essayer format CSV d'abord
        if self.csv_file.exists():
            return self._load_from_csv()

        # Sinon format dossiers
        return self._load_from_folders()

    def _load_from_csv(self) -> List[EmotionSample]:
        """Charge depuis le fichier CSV (format Kaggle)."""
        samples = []

        # Créer un dossier pour les images extraites
        images_dir = self._ckplus_dir / "images_extracted"
        images_dir.mkdir(exist_ok=True)

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                # Parser l'émotion (0-7)
                emotion_idx = int(row.get("emotion", 0))
                label = self.INDEX_TO_LABEL.get(emotion_idx)

                if label is None:
                    continue

                # Parser les pixels et sauvegarder l'image
                pixels = row.get("pixels", "")
                if pixels:
                    img_path = images_dir / f"ck_{idx}_{label.name.lower()}.png"

                    if not img_path.exists():
                        import cv2
                        pixel_values = [int(p) for p in pixels.split()]
                        # CK+ est 48x48 dans ce format
                        img_array = np.array(pixel_values, dtype=np.uint8).reshape(48, 48)
                        cv2.imwrite(str(img_path), img_array)

                    samples.append(EmotionSample(
                        image_path=str(img_path),
                        ground_truth=label
                    ))

        return samples

    def _load_from_folders(self) -> List[EmotionSample]:
        """Charge depuis la structure de dossiers (format original)."""
        samples = []

        if not self.images_dir.exists():
            return samples

        # Parcourir les sujets (S005, S010, etc.)
        for subject_dir in sorted(self.images_dir.glob("S*")):
            if not subject_dir.is_dir():
                continue

            # Parcourir les séquences (001, 002, etc.)
            for seq_dir in sorted(subject_dir.glob("*")):
                if not seq_dir.is_dir():
                    continue

                # Trouver le label pour cette séquence
                label = self._get_sequence_label(subject_dir.name, seq_dir.name)
                if label is None:
                    continue

                # Charger les images
                images = sorted(seq_dir.glob("*.png"))
                if not images:
                    images = sorted(seq_dir.glob("*.jpg"))

                if not images:
                    continue

                if self.apex_only:
                    # Seulement la dernière image (apex)
                    apex_img = images[-1]
                    samples.append(EmotionSample(
                        image_path=str(apex_img),
                        ground_truth=label
                    ))
                else:
                    # Toutes les images de la séquence
                    for img_path in images:
                        samples.append(EmotionSample(
                            image_path=str(img_path),
                            ground_truth=label
                        ))

        return samples

    def _get_sequence_label(self, subject: str, sequence: str) -> Optional[EmotionLabel]:
        """Récupère le label d'une séquence."""
        # Chercher le fichier de label
        label_dir = self.labels_dir / subject / sequence
        if not label_dir.exists():
            return None

        label_files = list(label_dir.glob("*_emotion.txt"))
        if not label_files:
            label_files = list(label_dir.glob("*.txt"))

        if not label_files:
            return None

        try:
            with open(label_files[0], 'r') as f:
                content = f.read().strip()
                label_idx = int(float(content))
                return self.INDEX_TO_LABEL.get(label_idx)
        except (ValueError, IOError):
            return None


class CKPlusAllFramesDataset(CKPlusDataset):
    """CK+ avec toutes les frames (pas seulement apex)."""

    name = "CK+-AllFrames"
    description = "CK+ with all sequence frames"

    def __init__(self):
        super().__init__(apex_only=False)
