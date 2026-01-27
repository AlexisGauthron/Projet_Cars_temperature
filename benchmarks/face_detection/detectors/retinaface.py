# -*- coding: utf-8 -*-
"""
RetinaFace face detector.
Haute précision, plus lent, nécessite images HD.

Utilise insightface comme backend (évite les conflits TensorFlow).
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from .base import BaseDetector


class RetinaFaceDetector(BaseDetector):
    """RetinaFace face detector via InsightFace."""

    name = "RetinaFace"

    def __init__(self):
        super().__init__()
        self.detector = None
        self._use_retinaface_pkg = False

        # Try insightface with retinaface model
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection'],
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.detector = self.app
            self._log_init_success()
            return
        except ImportError as e:
            self._logger.debug(f"insightface not available: {e}")
        except Exception as e:
            self._logger.debug(f"insightface init failed: {e}")

        # Fallback to retinaface package
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
            self._use_retinaface_pkg = True
            self._log_init_success()
        except ImportError as e:
            self._log_init_error(ImportError("Neither insightface nor retinaface installed"))

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None:
            return []

        try:
            if self._use_retinaface_pkg:
                faces = self.detector.detect_faces(image)
                return [
                    BBox(
                        f['facial_area'][0], f['facial_area'][1],
                        f['facial_area'][2] - f['facial_area'][0],
                        f['facial_area'][3] - f['facial_area'][1],
                        confidence=f.get('score', 1.0)
                    )
                    for f in faces.values()
                ]
            else:
                faces = self.detector.get(image)
                boxes = []
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        conf = float(face.det_score) if hasattr(face, 'det_score') else 1.0
                        boxes.append(BBox(x1, y1, w, h, confidence=conf))
                return boxes
        except Exception as e:
            self._log_detection_error(e)
            return []

    def is_available(self) -> bool:
        return self.detector is not None
