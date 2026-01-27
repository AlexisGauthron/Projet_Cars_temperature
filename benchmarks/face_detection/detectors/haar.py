# -*- coding: utf-8 -*-
"""
Haar Cascade face detector (OpenCV).
Ultra rapide mais précision limitée.
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


class HaarDetector(BaseDetector):
    """Haar Cascade face detector."""

    name = "Haar"

    def __init__(self):
        super().__init__()
        self.detector = None
        cascade_path = MODELS_DIR / "haar" / "haarcascade_frontalface_default.xml"

        if cv2 is None:
            self._log_init_error(ImportError("OpenCV not installed"))
            return

        if not cascade_path.exists():
            self._log_init_error(FileNotFoundError(f"Cascade not found: {cascade_path}"))
            return

        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            self._log_init_error(ValueError("Failed to load cascade classifier"))
            self.detector = None
        else:
            self._log_init_success()

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None:
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            return [BBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        except Exception as e:
            self._log_detection_error(e)
            return []

    def is_available(self) -> bool:
        return self.detector is not None
