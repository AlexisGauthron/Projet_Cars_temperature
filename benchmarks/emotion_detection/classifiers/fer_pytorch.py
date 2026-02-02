# -*- coding: utf-8 -*-
"""
Classifieur d'émotions FER (Facial Expression Recognition) avec PyTorch.

Utilise le package 'fer' qui fournit un CNN pré-entraîné sur FER2013.

Installation: pip install fer
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import cv2

from .base import BaseEmotionClassifier
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class FERPytorchClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant la bibliothèque FER.

    FER utilise un CNN et peut optionnellement utiliser MTCNN
    pour la détection de visages.
    """

    name = "FER"
    description = "FER library - CNN trained on FER2013"

    # Mapping FER -> EmotionLabel
    FER_TO_LABEL = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
    }

    def __init__(self, mtcnn: bool = False):
        """
        Initialise le classifieur FER.

        Args:
            mtcnn: Utiliser MTCNN pour la détection de visages
        """
        super().__init__()
        self.use_mtcnn = mtcnn
        self._detector = None

    def _load_model(self):
        """Charge le modèle FER."""
        try:
            from fer import FER

            self._detector = FER(mtcnn=self.use_mtcnn)
            self._model = self._detector

        except ImportError:
            self._model = None
            raise ImportError("fer non installé. Installez avec: pip install fer")

    def is_available(self) -> bool:
        """Vérifie si FER est disponible."""
        try:
            from fer import FER
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec FER."""
        try:
            # FER attend une image BGR (OpenCV format)
            # Détecter les émotions
            results = self._detector.detect_emotions(image)

            if not results:
                # Aucun visage détecté, essayer sur l'image entière
                # en supposant qu'elle contient déjà un visage croppé
                results = self._detector.detect_emotions(image)

            if not results:
                return EmotionLabel.NEUTRAL, 0.0, {}

            # Prendre le premier visage détecté
            emotions = results[0].get('emotions', {})

            # Convertir en EmotionLabel
            probabilities = {}
            for emotion_name, score in emotions.items():
                label = self.FER_TO_LABEL.get(emotion_name.lower())
                if label:
                    probabilities[label] = float(score)

            # Trouver le label dominant
            if probabilities:
                predicted_label = max(probabilities, key=probabilities.get)
                confidence = probabilities[predicted_label]
            else:
                predicted_label = EmotionLabel.NEUTRAL
                confidence = 0.0

            return predicted_label, confidence, probabilities

        except Exception as e:
            return EmotionLabel.NEUTRAL, 0.0, {}


class FERMTCNNClassifier(FERPytorchClassifier):
    """FER avec détection de visages MTCNN."""

    name = "FER-MTCNN"
    description = "FER with MTCNN face detection"

    def __init__(self):
        super().__init__(mtcnn=True)
