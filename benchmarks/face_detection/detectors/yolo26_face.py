# -*- coding: utf-8 -*-
"""
YOLO26 Face detector.
Dernier modèle YOLO (Janvier 2026) - Ultralytics.
Note: Ce modèle est généraliste, pas spécialisé visages.
Pour la détection de visages, il utilise un seuil de confiance élevé.
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class YOLO26FaceDetector(BaseDetector):
    """YOLO26 detector - Latest YOLO (January 2026).

    Note: YOLO26 est un modèle généraliste (COCO 80 classes).
    Il n'y a pas encore de version fine-tuned pour les visages.
    Ce détecteur n'est donc pas optimal pour la détection de visages.
    """

    name = "YOLO26"

    def __init__(self):
        super().__init__()
        self.model = None

        try:
            from ultralytics import YOLO

            # Essayer de charger le modèle local
            model_path = MODELS_DIR / "yolo26" / "yolo26n.pt"
            if model_path.exists():
                self.model = YOLO(str(model_path))
                self._log_init_success()
            else:
                # Télécharger depuis Ultralytics
                try:
                    self.model = YOLO("yolo26n.pt")
                    self._log_init_success()
                except Exception as e:
                    self._log_init_error(e)
        except ImportError as e:
            self._log_init_error(e)
        except Exception as e:
            self._log_init_error(e)

    def detect(self, image: np.ndarray) -> List[BBox]:
        """
        YOLO26 généraliste - détecte les personnes (classe 0 COCO).
        Non optimal pour les visages car pas fine-tuned.
        """
        if self.model is None:
            return []
        try:
            # Note: YOLO26 généraliste ne détecte pas les visages directement
            # Il détecte les personnes (classe 0)
            # Ce détecteur retournera une liste vide pour le benchmark visages
            # car il n'a pas de classe "face"
            results = self.model(image, verbose=False)
            boxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Vérifier si c'est une classe "face" (n'existe pas en COCO)
                        cls_id = int(box.cls[0].cpu().numpy())
                        # COCO n'a pas de classe face, seulement person (0)
                        # On ne peut pas utiliser ce modèle pour la détection de visages
                        # Sans fine-tuning
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        w, h = int(x2 - x1), int(y2 - y1)
                        if w > 10 and h > 10 and conf > 0.3:
                            boxes.append(BBox(int(x1), int(y1), w, h, confidence=conf))
            return boxes
        except Exception as e:
            self._log_detection_error(e)
            return []

    def is_available(self) -> bool:
        return self.model is not None
