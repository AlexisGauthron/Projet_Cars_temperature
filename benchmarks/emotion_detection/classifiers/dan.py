# -*- coding: utf-8 -*-
"""
Classifieur d'émotions DAN (Distract your Attention Network).

DAN utilise une architecture multi-head attention pour capturer
simultanément plusieurs régions faciales pertinentes.

Performances:
- RAF-DB: 89.70%
- AffectNet-7: 65.69%
- AffectNet-8: 62.09%

Installation:
    git clone https://github.com/yaoing/DAN
    pip install torch torchvision

Poids pré-entraînés:
    Télécharger depuis: https://github.com/yaoing/DAN
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

from .base import BaseEmotionClassifier, MODELS_DIR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class DANClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant DAN (Distract your Attention Network).

    Composants clés:
    - Feature Clustering Network (FCN)
    - Multi-head Attention Network (MAN)
    - Attention Fusion Network (AFN)
    """

    name = "DAN"
    description = "Distract your Attention Network - Multi-head attention FER"

    # Labels RAF-DB
    RAFDB_LABELS = [
        EmotionLabel.SURPRISE,
        EmotionLabel.FEAR,
        EmotionLabel.DISGUST,
        EmotionLabel.HAPPY,
        EmotionLabel.SAD,
        EmotionLabel.ANGRY,
        EmotionLabel.NEUTRAL,
    ]

    # Labels AffectNet-8
    AFFECTNET8_LABELS = [
        EmotionLabel.NEUTRAL,
        EmotionLabel.HAPPY,
        EmotionLabel.SAD,
        EmotionLabel.SURPRISE,
        EmotionLabel.FEAR,
        EmotionLabel.DISGUST,
        EmotionLabel.ANGRY,
        EmotionLabel.CONTEMPT,
    ]

    def __init__(self, weights_path: Optional[str] = None, dataset: str = "rafdb", num_head: int = 4):
        """
        Initialise le classifieur DAN.

        Args:
            weights_path: Chemin vers les poids pré-entraînés
            dataset: Dataset utilisé ("rafdb", "affectnet7", "affectnet8")
            num_head: Nombre de têtes d'attention
        """
        super().__init__()
        self.dataset = dataset
        self.num_head = num_head
        self.weights_path = weights_path or str(MODELS_DIR / "dan" / f"dan_{dataset}.pth")
        self._transform = None

        # Choisir les labels selon le dataset
        if "affectnet8" in dataset:
            self.LABELS = self.AFFECTNET8_LABELS
        else:
            self.LABELS = self.RAFDB_LABELS[:7]

    def _load_model(self):
        """Charge le modèle DAN."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms

            # Vérifier les poids
            weights_file = Path(self.weights_path)
            if not weights_file.exists():
                raise FileNotFoundError(
                    f"Poids DAN non trouvés: {self.weights_path}\n"
                    f"Téléchargez depuis: https://github.com/yaoing/DAN"
                )

            # Essayer d'importer DAN
            try:
                from networks.dan import DAN
            except ImportError:
                dan_path = MODELS_DIR / "dan" / "DAN"
                if dan_path.exists():
                    sys.path.insert(0, str(dan_path))
                    from networks.dan import DAN
                else:
                    raise ImportError(
                        "DAN non installé.\n"
                        "Installation:\n"
                        "  cd benchmarks/emotion_detection/models\n"
                        "  git clone https://github.com/yaoing/DAN dan/DAN"
                    )

            # Créer le modèle
            num_classes = len(self.LABELS)
            self._model = DAN(num_head=self.num_head, num_class=num_classes)

            # Charger les poids
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['state_dict'])
            else:
                self._model.load_state_dict(checkpoint)

            self._model.eval()

            # Transformation
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        except ImportError as e:
            self._model = None
            raise ImportError(str(e))

    def is_available(self) -> bool:
        """Vérifie si DAN est disponible."""
        weights_file = Path(self.weights_path)
        if not weights_file.exists():
            return False

        try:
            import torch
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec DAN."""
        try:
            import torch
            import cv2

            # Convertir BGR -> RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Appliquer les transformations
            input_tensor = self._transform(image_rgb).unsqueeze(0)

            # Prédiction
            with torch.no_grad():
                output, _, _ = self._model(input_tensor)
                probas = torch.softmax(output, dim=1)[0]

            # Construire les probabilités
            probabilities = {}
            for i, label in enumerate(self.LABELS):
                if i < len(probas):
                    probabilities[label] = float(probas[i])

            # Label prédit
            predicted_idx = torch.argmax(probas).item()
            if predicted_idx < len(self.LABELS):
                predicted_label = self.LABELS[predicted_idx]
            else:
                predicted_label = EmotionLabel.NEUTRAL

            confidence = probabilities.get(predicted_label, 0.0)

            return predicted_label, confidence, probabilities

        except Exception as e:
            return EmotionLabel.NEUTRAL, 0.0, {}


class DANAffectNetClassifier(DANClassifier):
    """DAN pré-entraîné sur AffectNet-8."""

    name = "DAN-AffectNet8"
    description = "DAN trained on AffectNet-8 (62.09%)"

    def __init__(self):
        super().__init__(dataset="affectnet8")
