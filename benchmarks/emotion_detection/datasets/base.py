# -*- coding: utf-8 -*-
"""
Classe de base pour les dataset loaders d'émotions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionSample


class BaseEmotionDataset(ABC):
    """Interface commune pour tous les datasets d'émotions."""

    name: str = "BaseEmotionDataset"
    description: str = "Dataset de base"
    num_classes: int = 7  # FER2013 standard

    def __init__(self):
        """Initialise le dataset."""
        self._samples: Optional[List[EmotionSample]] = None

    @property
    @abstractmethod
    def data_dir(self) -> Path:
        """Chemin vers le dossier des données."""
        pass

    @abstractmethod
    def load_samples(self) -> List[EmotionSample]:
        """
        Charge et retourne les échantillons du dataset.

        Returns:
            List[EmotionSample]: Liste des échantillons
        """
        pass

    def is_available(self) -> bool:
        """Vérifie si le dataset est disponible (fichiers présents)."""
        return self.data_dir.exists()

    def get_samples(self) -> List[EmotionSample]:
        """Retourne les échantillons (charge si nécessaire)."""
        if self._samples is None:
            self._samples = self.load_samples()
        return self._samples

    def __len__(self) -> int:
        """Retourne le nombre d'échantillons."""
        return len(self.get_samples())

    def __iter__(self) -> Iterator[EmotionSample]:
        """Itère sur les échantillons."""
        return iter(self.get_samples())

    def __getitem__(self, idx: int) -> EmotionSample:
        """Retourne un échantillon par index."""
        return self.get_samples()[idx]

    def get_image(self, sample: EmotionSample) -> Optional[np.ndarray]:
        """
        Charge et retourne l'image d'un échantillon.

        Args:
            sample: EmotionSample contenant le chemin de l'image

        Returns:
            Image numpy BGR ou None si erreur
        """
        image_path = Path(sample.image_path)

        # Si chemin relatif, le résoudre par rapport à data_dir
        if not image_path.is_absolute():
            image_path = self.data_dir / image_path

        if not image_path.exists():
            return None

        image = cv2.imread(str(image_path))
        return image

    def iterate_with_images(self, limit: Optional[int] = None) -> Iterator[Tuple[EmotionSample, np.ndarray]]:
        """
        Itère sur les échantillons avec leurs images.

        Args:
            limit: Nombre maximum d'échantillons

        Yields:
            Tuple (EmotionSample, image numpy)
        """
        samples = self.get_samples()
        if limit:
            samples = samples[:limit]

        for sample in samples:
            image = self.get_image(sample)
            if image is not None:
                yield sample, image

    def get_class_distribution(self) -> Dict[EmotionLabel, int]:
        """Retourne la distribution des classes."""
        distribution = {}
        for sample in self.get_samples():
            label = sample.ground_truth
            distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def get_stats(self) -> dict:
        """Retourne les statistiques du dataset."""
        samples = self.get_samples()
        distribution = self.get_class_distribution()

        return {
            "name": self.name,
            "description": self.description,
            "total_samples": len(samples),
            "num_classes": self.num_classes,
            "class_distribution": {
                label.name: count for label, count in distribution.items()
            },
        }

    def get_subset(self, labels: List[EmotionLabel]) -> List[EmotionSample]:
        """Retourne un sous-ensemble filtré par labels."""
        return [s for s in self.get_samples() if s.ground_truth in labels]

    def get_balanced_subset(self, samples_per_class: int) -> List[EmotionSample]:
        """Retourne un sous-ensemble équilibré."""
        import random
        subset = []
        for label in EmotionLabel.fer2013_labels():
            class_samples = [s for s in self.get_samples() if s.ground_truth == label]
            if len(class_samples) >= samples_per_class:
                subset.extend(random.sample(class_samples, samples_per_class))
            else:
                subset.extend(class_samples)
        return subset

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "not available"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
