# -*- coding: utf-8 -*-
"""
Classifieur d'émotions utilisant DeepFace.

DeepFace est une bibliothèque légère qui wrappe plusieurs modèles
de reconnaissance faciale et d'analyse d'émotions.

Installation: pip install deepface
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from .base import BaseEmotionClassifier, MODELS_DIR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionPrediction


class DeepFaceClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant DeepFace avec le modèle Emotion par défaut.

    Le modèle DeepFace Emotion est un CNN entraîné sur FER2013.
    """

    name = "DeepFace"
    description = "DeepFace Emotion model (CNN on FER2013)"

    # Mapping DeepFace -> EmotionLabel
    DEEPFACE_TO_LABEL = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
    }

    def __init__(self, enforce_detection: bool = False):
        """
        Initialise le classifieur DeepFace.

        Args:
            enforce_detection: Si True, lève une erreur si aucun visage détecté
        """
        super().__init__()
        self.enforce_detection = enforce_detection
        self._deepface = None

    def _load_model(self):
        """Charge DeepFace."""
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._model = True  # Flag pour indiquer que le modèle est chargé

            # Warmup pour télécharger les poids si nécessaire
            dummy = np.zeros((48, 48, 3), dtype=np.uint8)
            try:
                self._deepface.analyze(
                    dummy,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
            except Exception:
                pass

        except ImportError:
            self._model = None
            raise ImportError("DeepFace non installé. Installez avec: pip install deepface")

    def is_available(self) -> bool:
        """Vérifie si DeepFace est disponible."""
        try:
            from deepface import DeepFace
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec DeepFace."""
        try:
            # Analyser l'image
            results = self._deepface.analyze(
                image,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                silent=True
            )

            # DeepFace retourne une liste si plusieurs visages
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            # Extraire les probabilités
            emotion_scores = result.get('emotion', {})

            # Convertir en EmotionLabel
            probabilities = {}
            for emotion_name, score in emotion_scores.items():
                label = self.DEEPFACE_TO_LABEL.get(emotion_name.lower())
                if label:
                    probabilities[label] = score / 100.0  # DeepFace retourne des pourcentages

            # Trouver le label dominant
            dominant_emotion = result.get('dominant_emotion', 'neutral').lower()
            predicted_label = self.DEEPFACE_TO_LABEL.get(dominant_emotion, EmotionLabel.NEUTRAL)
            confidence = probabilities.get(predicted_label, 0.0)

            return predicted_label, confidence, probabilities

        except Exception as e:
            # En cas d'erreur, retourner Neutral avec faible confiance
            return EmotionLabel.NEUTRAL, 0.0, {}


class DeepFaceEmotionClassifier(DeepFaceClassifier):
    """Alias pour DeepFaceClassifier."""
    name = "DeepFace-Emotion"
