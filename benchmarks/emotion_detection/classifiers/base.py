# -*- coding: utf-8 -*-
"""
Classe de base pour les classifieurs d'émotions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionPrediction


# Répertoire des modèles
MODELS_DIR = Path(__file__).parent.parent / "models"


class BaseEmotionClassifier(ABC):
    """Interface commune pour tous les classifieurs d'émotions."""

    name: str = "BaseClassifier"
    description: str = "Classifieur de base"

    # Labels supportés par ce classifieur
    supported_labels: List[EmotionLabel] = EmotionLabel.fer2013_labels()

    def __init__(self):
        """Initialise le classifieur."""
        self._model = None

    @abstractmethod
    def _load_model(self):
        """Charge le modèle. À implémenter par les sous-classes."""
        pass

    @abstractmethod
    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """
        Implémentation de la prédiction.

        Args:
            image: Image BGR numpy

        Returns:
            Tuple (label prédit, confidence, dict des probabilités par classe)
        """
        pass

    def is_available(self) -> bool:
        """Vérifie si le classifieur peut être utilisé."""
        try:
            if self._model is None:
                self._load_model()
            return self._model is not None
        except Exception:
            return False

    def predict(self, image: np.ndarray) -> EmotionPrediction:
        """
        Prédit l'émotion d'une image.

        Args:
            image: Image BGR numpy (visage croppé de préférence)

        Returns:
            EmotionPrediction avec label, confidence et probabilités
        """
        # Charger le modèle si nécessaire
        if self._model is None:
            self._load_model()

        # Prédire
        label, confidence, probabilities = self._predict_impl(image)

        return EmotionPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities
        )

    def predict_batch(self, images: List[np.ndarray]) -> List[EmotionPrediction]:
        """
        Prédit les émotions pour un batch d'images.

        Args:
            images: Liste d'images BGR numpy

        Returns:
            Liste de EmotionPrediction
        """
        # Implémentation par défaut: prédiction séquentielle
        # Les sous-classes peuvent override pour batch processing
        return [self.predict(img) for img in images]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement de l'image avant prédiction.
        Par défaut, retourne l'image inchangée.
        Les sous-classes peuvent override.

        Args:
            image: Image BGR numpy

        Returns:
            Image prétraitée
        """
        return image

    def get_model_path(self, filename: str) -> Path:
        """Retourne le chemin vers un fichier modèle."""
        return MODELS_DIR / self.name.lower().replace(" ", "_") / filename

    def warmup(self, num_iterations: int = 3):
        """
        Warmup du modèle avec des images factices.

        Args:
            num_iterations: Nombre d'itérations de warmup
        """
        # Créer une image factice 48x48 (taille FER2013)
        dummy_image = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)

        for _ in range(num_iterations):
            try:
                self.predict(dummy_image)
            except Exception:
                pass

    def __repr__(self) -> str:
        status = "available" if self.is_available() else "not available"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
