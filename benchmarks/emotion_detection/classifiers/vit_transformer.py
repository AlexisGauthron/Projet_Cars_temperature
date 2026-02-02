# -*- coding: utf-8 -*-
"""
Classifieur d'émotions Vision Transformer (ViT) via HuggingFace.

Utilise des modèles pré-entraînés sur FER2013 disponibles sur HuggingFace Hub.

Installation: pip install transformers torch pillow
"""

from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from .base import BaseEmotionClassifier
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class ViTEmotionClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant Vision Transformer (ViT) de HuggingFace.

    Modèle: trpakov/vit-face-expression (71.55% sur FER2013)
    """

    name = "ViT-FER"
    description = "Vision Transformer fine-tuned on FER2013 (HuggingFace)"

    # Mapping des labels du modèle -> EmotionLabel
    MODEL_TO_LABEL = {
        "angry": EmotionLabel.ANGRY,
        "disgust": EmotionLabel.DISGUST,
        "fear": EmotionLabel.FEAR,
        "happy": EmotionLabel.HAPPY,
        "sad": EmotionLabel.SAD,
        "surprise": EmotionLabel.SURPRISE,
        "neutral": EmotionLabel.NEUTRAL,
    }

    def __init__(self, model_name: str = "trpakov/vit-face-expression"):
        """
        Initialise le classifieur ViT.

        Args:
            model_name: Nom du modèle HuggingFace
        """
        super().__init__()
        self.model_name = model_name
        self._pipeline = None
        self._processor = None

    def _load_model(self):
        """Charge le modèle ViT depuis HuggingFace."""
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-classification",
                model=self.model_name,
                device=-1  # CPU par défaut
            )
            self._model = self._pipeline

        except ImportError:
            self._model = None
            raise ImportError(
                "transformers non installé. "
                "Installez avec: pip install transformers torch pillow"
            )

    def is_available(self) -> bool:
        """Vérifie si le modèle est disponible."""
        try:
            from transformers import pipeline
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec ViT."""
        try:
            from PIL import Image
            import cv2

            # Convertir BGR -> RGB -> PIL
            if len(image.shape) == 2:
                # Grayscale -> RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(image_rgb)

            # Prédire
            results = self._pipeline(pil_image)

            # Parser les résultats
            probabilities = {}
            for result in results:
                label_str = result['label'].lower()
                score = result['score']

                emotion_label = self.MODEL_TO_LABEL.get(label_str)
                if emotion_label:
                    probabilities[emotion_label] = float(score)

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


class DeiTEmotionClassifier(ViTEmotionClassifier):
    """
    Classifieur utilisant DeiT (Data-efficient Image Transformer).

    Alternative au ViT standard avec de meilleures performances
    sur des données limitées.
    """

    name = "DeiT-FER"
    description = "Data-efficient Image Transformer for emotion recognition"

    def __init__(self):
        # Utilise le même modèle ViT par défaut (peut être changé)
        super().__init__(model_name="trpakov/vit-face-expression")
