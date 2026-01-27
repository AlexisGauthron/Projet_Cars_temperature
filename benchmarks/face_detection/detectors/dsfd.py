# -*- coding: utf-8 -*-
"""
DSFD - Dual Shot Face Detector.
State-of-the-art face detector (CVPR 2019).
Source: github.com/hukkelas/DSFD-Pytorch-Inference

Performance WIDER FACE:
- Easy: 96.6%
- Medium: 95.7%
- Hard: 90.4%

L'un des meilleurs détecteurs, mais plus lourd.
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class DSFDDetector(BaseDetector):
    """DSFD - Dual Shot Face Detector (CVPR 2019)."""

    name = "DSFD"

    def __init__(self):
        super().__init__()
        self.detector = None
        self.onnx_session = None
        self._try_load_model()
        if self.is_available():
            self._log_init_success()
        else:
            self._log_init_error(ImportError("Neither face_detection nor ONNX model available"))

    def _try_load_model(self):
        """Try to load DSFD model."""
        # Method 1: Via face_detection package (hukkelas)
        try:
            import face_detection
            self.detector = face_detection.build_detector(
                "DSFDDetector",
                confidence_threshold=0.5,
                nms_iou_threshold=0.3
            )
            return
        except ImportError as e:
            self._logger.debug(f"face_detection package not available: {e}")
        except Exception as e:
            self._logger.debug(f"face_detection init failed: {e}")

        # Method 2: Via ONNX model
        try:
            import onnxruntime as ort
            model_path = MODELS_DIR / "dsfd" / "dsfd.onnx"
            if model_path.exists():
                self.onnx_session = ort.InferenceSession(
                    str(model_path),
                    providers=['CPUExecutionProvider']
                )
                self.detector = "onnx"
                return
        except ImportError as e:
            self._logger.debug(f"onnxruntime not available: {e}")
        except Exception as e:
            self._logger.debug(f"ONNX load failed: {e}")

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None:
            return []

        try:
            # face_detection package interface
            if self.detector != "onnx":
                # RGB conversion (face_detection expects RGB)
                import cv2
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detections = self.detector.detect(rgb_image)

                boxes = []
                for det in detections:
                    x1, y1, x2, y2, conf = det
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 10 and h > 10:
                        boxes.append(BBox(int(x1), int(y1), w, h, confidence=float(conf)))
                return boxes
            else:
                # ONNX inference (fallback)
                return self._detect_onnx(image)
        except Exception as e:
            self._log_detection_error(e)
            return []

    def _detect_onnx(self, image: np.ndarray) -> List[BBox]:
        """ONNX inference fallback."""
        # Not implemented - DSFD ONNX model requires specific preprocessing
        return []

    def is_available(self) -> bool:
        return self.detector is not None
