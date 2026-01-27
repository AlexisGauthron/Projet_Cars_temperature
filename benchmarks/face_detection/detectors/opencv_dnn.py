# -*- coding: utf-8 -*-
"""
OpenCV DNN face detector (SSD ResNet).
Meilleur compromis précision/vitesse.
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


class OpenCVDNNDetector(BaseDetector):
    """OpenCV DNN face detector - SSD ResNet implementation."""

    name = "OpenCV-DNN"

    def __init__(self):
        super().__init__()
        self.net = None
        prototxt = MODELS_DIR / "opencv_dnn" / "deploy.prototxt"
        model = MODELS_DIR / "opencv_dnn" / "res10_300x300_ssd_iter_140000.caffemodel"

        if cv2 is None:
            self._log_init_error(ImportError("OpenCV not installed"))
            return

        if not prototxt.exists():
            self._log_init_error(FileNotFoundError(f"Prototxt not found: {prototxt}"))
            return

        if not model.exists():
            self._log_init_error(FileNotFoundError(f"Model not found: {model}"))
            return

        try:
            self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
            self._log_init_success()
        except Exception as e:
            self._log_init_error(e)

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.net is None:
            return []

        try:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            self.net.setInput(blob)
            detections = self.net.forward()
            boxes = []
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    boxes.append(BBox(
                        max(0, x1), max(0, y1), x2 - x1, y2 - y1,
                        confidence=confidence
                    ))
            return boxes
        except Exception as e:
            self._log_detection_error(e)
            return []

    def is_available(self) -> bool:
        return self.net is not None
