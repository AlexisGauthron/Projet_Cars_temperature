# -*- coding: utf-8 -*-
"""
Classifieur d'émotions HSEmotion ONNX.

Version ONNX de HSEmotion pour une meilleure compatibilité
et des performances optimisées.

Installation: pip install hsemotion-onnx
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from .base import BaseEmotionClassifier
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class HSEmotionONNXClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant HSEmotion ONNX.

    Version optimisée avec ONNX Runtime pour une meilleure
    compatibilité cross-platform et des performances améliorées.
    """

    name = "HSEmotion-ONNX"
    description = "HSEmotion with ONNX Runtime (fast & compatible)"

    # Mapping HSEmotion -> EmotionLabel (8 émotions)
    HSEMOTION_TO_LABEL = {
        0: EmotionLabel.ANGRY,
        1: EmotionLabel.DISGUST,
        2: EmotionLabel.FEAR,
        3: EmotionLabel.HAPPY,
        4: EmotionLabel.SAD,
        5: EmotionLabel.SURPRISE,
        6: EmotionLabel.NEUTRAL,
        # 7: Contempt (non supporté dans EmotionLabel.fer2013_labels)
    }

    LABEL_NAMES = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral', 'Contempt']

    def __init__(self, model_name: str = "enet_b0_8_best_afew"):
        """
        Initialise le classifieur HSEmotion ONNX.

        Args:
            model_name: Nom du modèle à utiliser
        """
        super().__init__()
        self.model_name = model_name
        self._fer = None

    def _load_model(self):
        """Charge le modèle HSEmotion ONNX."""
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

            self._fer = HSEmotionRecognizer(model_name=self.model_name)
            self._model = self._fer

        except ImportError:
            self._model = None
            raise ImportError(
                "hsemotion-onnx non installé. "
                "Installez avec: pip install hsemotion-onnx"
            )

    def is_available(self) -> bool:
        """Vérifie si HSEmotion ONNX est disponible."""
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec HSEmotion ONNX."""
        try:
            import cv2

            # Convertir en RGB si nécessaire
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Prédire avec HSEmotion ONNX
            emotion, scores = self._fer.predict_emotions(image_rgb, logits=True)

            # Construire les probabilités (softmax des logits)
            exp_scores = np.exp(scores - np.max(scores))
            probas = exp_scores / exp_scores.sum()

            probabilities = {}
            for i, prob in enumerate(probas):
                if i < 7:  # Seulement les 7 émotions de base
                    label = self.HSEMOTION_TO_LABEL.get(i)
                    if label:
                        probabilities[label] = float(prob)

            # Label prédit
            predicted_idx = int(emotion) if isinstance(emotion, (int, np.integer)) else 6
            if predicted_idx >= 7:
                predicted_idx = 6  # Default to neutral

            predicted_label = self.HSEMOTION_TO_LABEL.get(predicted_idx, EmotionLabel.NEUTRAL)
            confidence = probabilities.get(predicted_label, 0.0)

            return predicted_label, confidence, probabilities

        except Exception as e:
            return EmotionLabel.NEUTRAL, 0.0, {}
