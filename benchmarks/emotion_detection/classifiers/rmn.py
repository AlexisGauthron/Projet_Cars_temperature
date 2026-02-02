# -*- coding: utf-8 -*-
"""
Classifieur d'émotions RMN (Residual Masking Network).

RMN est un modèle state-of-the-art pour la reconnaissance d'émotions
qui utilise des masques résiduels pour améliorer la précision.

GitHub: https://github.com/phamquiluan/ResidualMaskingNetwork
Installation: pip install rmn
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import cv2

from .base import BaseEmotionClassifier
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class RMNClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant Residual Masking Network (RMN).

    RMN a obtenu d'excellents résultats sur FER2013 et RAF-DB.
    """

    name = "RMN"
    description = "Residual Masking Network - State-of-the-art on FER2013"

    # Mapping RMN -> EmotionLabel
    RMN_TO_LABEL = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
    }

    def __init__(self):
        """Initialise le classifieur RMN."""
        super().__init__()
        self._rmn = None

    def _load_model(self):
        """Charge le modèle RMN."""
        try:
            from rmn import RMN

            self._rmn = RMN()
            self._model = self._rmn

        except ImportError:
            self._model = None
            raise ImportError("rmn non installé. Installez avec: pip install rmn")

    def is_available(self) -> bool:
        """Vérifie si RMN est disponible."""
        try:
            from rmn import RMN
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec RMN."""
        try:
            # RMN attend une image BGR
            results = self._rmn.detect_emotion_for_single_frame(image)

            if not results:
                return EmotionLabel.NEUTRAL, 0.0, {}

            # Prendre le premier visage
            face_result = results[0]

            # Extraire les probabilités
            proba_list = face_result.get('proba_list', [])
            emo_label = face_result.get('emo_label', 'neutral')

            # Labels RMN dans l'ordre
            rmn_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

            probabilities = {}
            for idx, prob in enumerate(proba_list):
                if idx < len(rmn_labels):
                    label = self.RMN_TO_LABEL.get(rmn_labels[idx])
                    if label:
                        probabilities[label] = float(prob)

            # Label prédit
            predicted_label = self.RMN_TO_LABEL.get(emo_label.lower(), EmotionLabel.NEUTRAL)
            confidence = probabilities.get(predicted_label, 0.0)

            return predicted_label, confidence, probabilities

        except Exception as e:
            return EmotionLabel.NEUTRAL, 0.0, {}

    def predict_with_face_detection(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict, list]:
        """
        Prédit l'émotion avec détection de visage intégrée.

        Returns:
            Tuple (label, confidence, probabilities, face_boxes)
        """
        try:
            results = self._rmn.detect_emotion_for_single_frame(image)

            if not results:
                return EmotionLabel.NEUTRAL, 0.0, {}, []

            face_result = results[0]

            # Extraire la bounding box
            xmin = face_result.get('xmin', 0)
            ymin = face_result.get('ymin', 0)
            xmax = face_result.get('xmax', 0)
            ymax = face_result.get('ymax', 0)
            face_boxes = [(xmin, ymin, xmax - xmin, ymax - ymin)]

            # Prédiction
            label, confidence, probabilities = self._predict_impl(image)

            return label, confidence, probabilities, face_boxes

        except Exception:
            return EmotionLabel.NEUTRAL, 0.0, {}, []
