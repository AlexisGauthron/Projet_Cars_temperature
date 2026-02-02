# -*- coding: utf-8 -*-
"""
Classifieur d'émotions EfficientNet via timm.

EfficientNet est une architecture CNN efficiente qui offre un bon compromis
entre vitesse et précision pour la classification d'émotions.

Installation: pip install timm torch pillow
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

from .base import BaseEmotionClassifier, MODELS_DIR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class EfficientNetClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant EfficientNet via timm.

    Utilise un modèle EfficientNet pré-entraîné qu'on peut
    fine-tuner sur FER2013 ou charger des poids existants.
    """

    name = "EfficientNet-B0"
    description = "EfficientNet-B0 for emotion classification (timm)"

    # Labels FER2013 dans l'ordre
    LABELS = [
        EmotionLabel.ANGRY,
        EmotionLabel.DISGUST,
        EmotionLabel.FEAR,
        EmotionLabel.HAPPY,
        EmotionLabel.SAD,
        EmotionLabel.SURPRISE,
        EmotionLabel.NEUTRAL,
    ]

    def __init__(self, model_name: str = "efficientnet_b0", weights_path: Optional[str] = None):
        """
        Initialise le classifieur EfficientNet.

        Args:
            model_name: Nom du modèle timm (efficientnet_b0, efficientnet_b2, etc.)
            weights_path: Chemin vers des poids pré-entraînés (optionnel)
        """
        super().__init__()
        self.model_name = model_name
        self.weights_path = weights_path
        self._transform = None

    def _load_model(self):
        """Charge le modèle EfficientNet."""
        try:
            import timm
            import torch

            # Créer le modèle avec 7 classes (émotions FER2013)
            self._model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=7
            )

            # Charger des poids personnalisés si fournis
            if self.weights_path and Path(self.weights_path).exists():
                state_dict = torch.load(self.weights_path, map_location='cpu')
                self._model.load_state_dict(state_dict)

            self._model.eval()

            # Transformation pour le preprocessing
            data_config = timm.data.resolve_model_data_config(self._model)
            self._transform = timm.data.create_transform(**data_config, is_training=False)

        except ImportError:
            self._model = None
            raise ImportError(
                "timm non installé. "
                "Installez avec: pip install timm torch pillow"
            )

    def is_available(self) -> bool:
        """Vérifie si timm est disponible."""
        try:
            import timm
            import torch
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec EfficientNet."""
        try:
            import torch
            from PIL import Image
            import cv2

            # Convertir BGR -> RGB -> PIL
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(image_rgb)

            # Appliquer les transformations
            input_tensor = self._transform(pil_image).unsqueeze(0)

            # Prédiction
            with torch.no_grad():
                output = self._model(input_tensor)
                probas = torch.softmax(output, dim=1)[0]

            # Construire le dictionnaire de probabilités
            probabilities = {}
            for i, label in enumerate(self.LABELS):
                probabilities[label] = float(probas[i])

            # Trouver le label dominant
            predicted_idx = torch.argmax(probas).item()
            predicted_label = self.LABELS[predicted_idx]
            confidence = probabilities[predicted_label]

            return predicted_label, confidence, probabilities

        except Exception as e:
            return EmotionLabel.NEUTRAL, 0.0, {}


class EfficientNetB2Classifier(EfficientNetClassifier):
    """EfficientNet-B2 (plus grand, plus précis)."""

    name = "EfficientNet-B2"
    description = "EfficientNet-B2 for emotion classification"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(model_name="efficientnet_b2", weights_path=weights_path)


class EfficientNetV2Classifier(EfficientNetClassifier):
    """EfficientNet-V2 (version améliorée)."""

    name = "EfficientNetV2-S"
    description = "EfficientNet-V2 Small for emotion classification"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(model_name="efficientnetv2_rw_s", weights_path=weights_path)
