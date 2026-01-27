# -*- coding: utf-8 -*-
"""
YuNet face detector (OpenCV).
Robuste aux occlusions partielles, léger et rapide.
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


class YuNetDetector(BaseDetector):
    """YuNet face detector - OpenCV implementation."""

    name = "YuNet"

    def __init__(self):
        super().__init__()
        self.detector = None
        model_path = MODELS_DIR / "yunet" / "face_detection_yunet_2023mar.onnx"

        if cv2 is None:
            self._log_init_error(ImportError("OpenCV not installed"))
            return

        if not model_path.exists():
            self._log_init_error(FileNotFoundError(f"Model not found: {model_path}"))
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(model_path), "", (320, 320), 0.7, 0.3
            )
            self._log_init_success()
        except Exception as e:
            self._log_init_error(e)

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None:
            return []

        try:
            h, w = image.shape[:2]
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image)

            if faces is None:
                return []

            return [
                BBox(
                    int(f[0]), int(f[1]), int(f[2]), int(f[3]),
                    confidence=float(f[14]) if len(f) > 14 else 1.0
                )
                for f in faces if f[2] > 10 and f[3] > 10
            ]
        except Exception as e:
            self._log_detection_error(e)
            return []

    def is_available(self) -> bool:
        return self.detector is not None
