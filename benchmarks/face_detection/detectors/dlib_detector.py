# -*- coding: utf-8 -*-
"""
DLib face detectors (HOG et CNN).
Classique et robuste, avec support des landmarks 68 points.
"""

from typing import List
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class DLibHOGDetector(BaseDetector):
    """DLib HOG face detector - Classique et rapide."""

    name = "DLib-HOG"

    def __init__(self):
        self.detector = None

        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
        except ImportError:
            pass
        except Exception:
            pass

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None or cv2 is None:
            return []
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            return [
                BBox(
                    max(0, f.left()),
                    max(0, f.top()),
                    f.width(),
                    f.height(),
                    confidence=1.0
                )
                for f in faces
            ]
        except Exception:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class DLibCNNDetector(BaseDetector):
    """DLib CNN face detector - Plus précis mais plus lent."""

    name = "DLib-CNN"

    def __init__(self):
        self.detector = None

        try:
            import dlib
            # Le modèle CNN nécessite un fichier .dat
            model_path = MODELS_DIR / "dlib" / "mmod_human_face_detector.dat"
            if model_path.exists():
                self.detector = dlib.cnn_face_detection_model_v1(str(model_path))
        except ImportError:
            pass
        except Exception:
            pass

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None:
            return []
        try:
            # DLib CNN attend une image RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cv2 else image
            faces = self.detector(rgb, 1)
            return [
                BBox(
                    max(0, f.rect.left()),
                    max(0, f.rect.top()),
                    f.rect.width(),
                    f.rect.height(),
                    confidence=float(f.confidence)
                )
                for f in faces
            ]
        except Exception:
            return []

    def is_available(self) -> bool:
        return self.detector is not None
