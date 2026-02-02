# -*- coding: utf-8 -*-
"""
Classifieur d'émotions Py-Feat (Python Facial Expression Analysis Toolbox).

Py-Feat est une boîte à outils complète pour l'analyse des expressions faciales
qui inclut la détection de visages, landmarks, action units et émotions.

Installation: pip install py-feat
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from .base import BaseEmotionClassifier
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class PyFeatClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant Py-Feat (Facial Expression Analysis Toolbox).

    Py-Feat offre une analyse complète incluant:
    - Détection de visages
    - Landmarks faciaux
    - Action Units (AUs)
    - Émotions
    """

    name = "Py-Feat"
    description = "Python Facial Expression Analysis Toolbox"

    # Mapping Py-Feat -> EmotionLabel
    PYFEAT_TO_LABEL = {
        "anger": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happiness": EmotionLabel.HAPPY,
        "happy": EmotionLabel.HAPPY,
        "sadness": EmotionLabel.SAD,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
    }

    def __init__(self):
        """Initialise le classifieur Py-Feat."""
        super().__init__()
        self._detector = None

    def _load_model(self):
        """Charge le détecteur Py-Feat."""
        try:
            from feat import Detector

            # Charger le détecteur avec les modèles par défaut
            self._detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="xgb",
                emotion_model="resmasknet",
                facepose_model="img2pose"
            )
            self._model = self._detector

        except ImportError:
            self._model = None
            raise ImportError(
                "py-feat non installé. "
                "Installez avec: pip install py-feat"
            )

    def is_available(self) -> bool:
        """Vérifie si Py-Feat est disponible."""
        try:
            from feat import Detector
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec Py-Feat."""
        try:
            import cv2

            # Py-Feat attend une image BGR ou un path
            # Détection sur l'image
            results = self._detector.detect_image(image)

            if results is None or len(results) == 0:
                return EmotionLabel.NEUTRAL, 0.0, {}

            # Extraire les émotions du premier visage
            # Les colonnes d'émotion sont: anger, disgust, fear, happiness, sadness, surprise, neutral
            emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

            probabilities = {}
            max_score = 0.0
            predicted_label = EmotionLabel.NEUTRAL

            for col in emotion_cols:
                if col in results.columns:
                    score = float(results[col].iloc[0])
                    label = self.PYFEAT_TO_LABEL.get(col.lower())
                    if label:
                        probabilities[label] = score
                        if score > max_score:
                            max_score = score
                            predicted_label = label

            confidence = max_score

            return predicted_label, confidence, probabilities

        except Exception as e:
            return EmotionLabel.NEUTRAL, 0.0, {}


class PyFeatSVMClassifier(PyFeatClassifier):
    """Py-Feat avec modèle SVM pour les émotions."""

    name = "Py-Feat-SVM"
    description = "Py-Feat with SVM emotion model"

    def _load_model(self):
        """Charge Py-Feat avec SVM."""
        try:
            from feat import Detector

            self._detector = Detector(
                face_model="retinaface",
                landmark_model="mobilefacenet",
                au_model="xgb",
                emotion_model="svm",
                facepose_model="img2pose"
            )
            self._model = self._detector

        except ImportError:
            self._model = None
            raise ImportError("py-feat non installé.")
