# -*- coding: utf-8 -*-
"""
Classifieur d'émotions POSTER V2 (POSTER++).

POSTER++ est un réseau state-of-the-art pour la reconnaissance d'émotions
utilisant une architecture Transformer avec cross-fusion pyramidale.

Performances:
- RAF-DB: 92.21%
- AffectNet-7: 67.49%
- FER2013: ~80%

Installation:
    git clone https://github.com/Talented-Q/POSTER_V2
    cd POSTER_V2
    pip install -r requirements.txt

Poids pré-entraînés:
    Télécharger depuis le repo GitHub et placer dans models/poster/
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

from .base import BaseEmotionClassifier, MODELS_DIR
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel


class POSTERClassifier(BaseEmotionClassifier):
    """
    Classifieur utilisant POSTER V2 (POSTER++).

    State-of-the-art sur RAF-DB (92.21%) et AffectNet.
    Utilise une architecture Transformer avec:
    - Window-based cross-attention
    - Two-stream feature extraction
    - Multi-scale processing
    """

    name = "POSTER++"
    description = "POSTER V2 - SOTA on RAF-DB (92.21%)"

    # Labels dans l'ordre RAF-DB/AffectNet
    LABELS = [
        EmotionLabel.SURPRISE,
        EmotionLabel.FEAR,
        EmotionLabel.DISGUST,
        EmotionLabel.HAPPY,
        EmotionLabel.SAD,
        EmotionLabel.ANGRY,
        EmotionLabel.NEUTRAL,
    ]

    def __init__(self, weights_path: Optional[str] = None, dataset: str = "rafdb"):
        """
        Initialise le classifieur POSTER++.

        Args:
            weights_path: Chemin vers les poids pré-entraînés
            dataset: Dataset utilisé pour les poids ("rafdb", "affectnet7", "affectnet8")
        """
        super().__init__()
        self.dataset = dataset
        self.weights_path = weights_path or str(MODELS_DIR / "poster" / f"poster_{dataset}.pth")
        self._transform = None

    def _load_model(self):
        """Charge le modèle POSTER++."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms

            # Vérifier si les poids existent
            weights_file = Path(self.weights_path)
            if not weights_file.exists():
                raise FileNotFoundError(
                    f"Poids POSTER++ non trouvés: {self.weights_path}\n"
                    f"Téléchargez depuis: https://github.com/Talented-Q/POSTER_V2"
                )

            # Essayer d'importer POSTER V2
            try:
                # Si POSTER_V2 est installé comme package
                from models.PosterV2_7cls import pyramid_trans_expr2
            except ImportError:
                # Sinon, essayer de charger depuis un chemin local
                poster_path = MODELS_DIR / "poster" / "POSTER_V2"
                if poster_path.exists():
                    sys.path.insert(0, str(poster_path))
                    from models.PosterV2_7cls import pyramid_trans_expr2
                else:
                    raise ImportError(
                        "POSTER_V2 non installé.\n"
                        "Installation:\n"
                        "  cd benchmarks/emotion_detection/models\n"
                        "  git clone https://github.com/Talented-Q/POSTER_V2 poster/POSTER_V2\n"
                        "  pip install timm==0.9.7"
                    )

            # Créer le modèle
            num_classes = 8 if "affectnet8" in self.dataset else 7
            self._model = pyramid_trans_expr2(
                img_size=224,
                num_classes=num_classes
            )

            # Charger les poids
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            if 'state_dict' in checkpoint:
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
        """Vérifie si POSTER++ est disponible."""
        weights_file = Path(self.weights_path)
        if not weights_file.exists():
            return False

        try:
            import torch
            import timm
            return True
        except ImportError:
            return False

    def _predict_impl(self, image: np.ndarray) -> Tuple[EmotionLabel, float, Dict[EmotionLabel, float]]:
        """Prédit l'émotion avec POSTER++."""
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
                output = self._model(input_tensor)
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


class POSTERAffectNetClassifier(POSTERClassifier):
    """POSTER++ pré-entraîné sur AffectNet-7."""

    name = "POSTER++-AffectNet"
    description = "POSTER V2 trained on AffectNet-7 (67.49%)"

    def __init__(self):
        super().__init__(dataset="affectnet7")
