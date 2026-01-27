# -*- coding: utf-8 -*-
"""
MTCNN face detector.
Standard du domaine, bonne précision mais lent.
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
from .base import BaseDetector


class MTCNNDetector(BaseDetector):
    """MTCNN face detector."""

    name = "MTCNN"

    def __init__(self):
        super().__init__()
        self.detector = None
        self._is_pytorch = False

        # Essayer mtcnn (TensorFlow)
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            self._log_init_success()
            return
        except ImportError as e:
            self._logger.debug(f"mtcnn (TensorFlow) not available: {e}")

        # Sinon essayer facenet-pytorch
        try:
            from facenet_pytorch import MTCNN as MTCNN_PT
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.detector = MTCNN_PT(keep_all=True, device=device)
            self._is_pytorch = True
            self._log_init_success()
        except ImportError as e:
            self._log_init_error(ImportError("Neither mtcnn nor facenet-pytorch installed"))

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None:
            return []

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self._is_pytorch:
                boxes, probs = self.detector.detect(rgb)
                if boxes is None:
                    return []
                return [
                    BBox(
                        int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]),
                        confidence=float(probs[i]) if probs is not None else 1.0
                    )
                    for i, b in enumerate(boxes)
                ]
            else:
                faces = self.detector.detect_faces(rgb)
                return [BBox(*f['box'], confidence=f['confidence']) for f in faces]
        except Exception as e:
            self._log_detection_error(e)
            return []

    def is_available(self) -> bool:
        return self.detector is not None
