# -*- coding: utf-8 -*-
"""
Classifieur d'émotions HSEmotion.

HSEmotion (High-Speed Emotion) est un modèle léger et rapide
basé sur EfficientNet pour la classification d'émotions.

GitHub: https://github.com/av-savchenko/face-emotion-recognition
Installation: pip install hsemotion

Modèles disponibles:
- enet_b0_8_best_afew: EfficientNet-B0, 8 classes (AffectNet)
- enet_b0_8_va_mtl: EfficientNet-B0, 8 classes + valence/arousal
- enet_b2_8: EfficientNet-B2, 8 classes
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

from .base import BaseEmotionClassifier, MODELS_DIR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class HSEmotionClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant HSEmotion (EfficientNet-based).
    """

    name = "HSEmotion"
    description = "EfficientNet-based emotion classifier (fast & accurate)"

    # Mapping HSEmotion (AffectNet order) -> EmotionLabel
    INDEX_TO_LABEL = {
        0: EmotionLabel.ANGRY,
        1: EmotionLabel.DISGUST,
        2: EmotionLabel.FEAR,
        3: EmotionLabel.HAPPY,
        4: EmotionLabel.NEUTRAL,
        5: EmotionLabel.SAD,
        6: EmotionLabel.SURPRISE,
        7: EmotionLabel.CONTEMPT,
    }

    def __init__(self, model_name: str = "enet_b0_8_best_afew"):
        """
        Initialise HSEmotion.

        Args:
            model_name: Nom du modèle HSEmotion à utiliser
        """
        super().__init__()
        self.model_name = model_name
        self._fer = None

    def _load_model(self):
        """Charge le modèle HSEmotion."""
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer

            self._fer = HSEmotionRecognizer(model_name=self.model_name)
            self._model = self._fer

        except ImportError:
            self._model = None
            raise ImportError("hsemotion non installé. Installez avec: pip install hsemotion")

    def is_available(self) -> bool:
        """Vérifie si HSEmotion est disponible."""
        try:
            from hsemotion.facial_emotions import HSEmotionRecognizer
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec HSEmotion."""
        try:
            # HSEmotion attend une image RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Prédire
            emotion, scores = self._fer.predict_emotions(image_rgb, logits=False)

            # Convertir les scores en probabilités par label
            probabilities = {}
            for idx, score in enumerate(scores):
                if idx < len(self.INDEX_TO_LABEL):
                    label = self.INDEX_TO_LABEL[idx]
                    probabilities[label] = float(score)

            # Trouver le label avec le score max
            max_idx = int(np.argmax(scores))
            predicted_label = self.INDEX_TO_LABEL.get(max_idx, EmotionLabel.NEUTRAL)
            confidence = float(scores[max_idx])

            return predicted_label, confidence, probabilities

        except Exception as e:
            return EmotionLabel.NEUTRAL, 0.0, {}


class HSEmotionB2Classifier(HSEmotionClassifier):
    """HSEmotion avec EfficientNet-B2 (plus précis mais plus lent)."""

    name = "HSEmotion-B2"
    description = "HSEmotion with EfficientNet-B2 backbone"

    def __init__(self):
        super().__init__(model_name="enet_b2_8")
