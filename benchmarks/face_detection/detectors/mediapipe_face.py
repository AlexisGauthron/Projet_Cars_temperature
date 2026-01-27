# -*- coding: utf-8 -*-
"""
MediaPipe Face detector (BlazeFace).
Google - Ultra-rapide, optimisé pour mobile.

Supporte les deux APIs:
- Legacy (mp.solutions.face_detection) pour MediaPipe < 0.10.x
- Tasks API (mp.tasks.python.vision) pour MediaPipe >= 0.10.x
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


class MediaPipeFaceDetector(BaseDetector):
    """MediaPipe Face detector - Google BlazeFace."""

    name = "MediaPipe"

    def __init__(self):
        super().__init__()
        self.detector = None
        self.use_tasks_api = False

        # Try new Tasks API first (MediaPipe >= 0.10.x)
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            # Download or use local model
            model_path = MODELS_DIR / "mediapipe" / "blaze_face_short_range.tflite"
            if not model_path.exists():
                # Use bundled model path
                base_options = python.BaseOptions(
                    model_asset_path=None  # Will use default
                )
            else:
                base_options = python.BaseOptions(
                    model_asset_path=str(model_path)
                )

            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5
            )
            self.detector = vision.FaceDetector.create_from_options(options)
            self.use_tasks_api = True
            self.mp = mp
            self._log_init_success()
            return
        except Exception as e:
            self._logger.debug(f"MediaPipe Tasks API not available: {e}")

        # Fallback to legacy API (MediaPipe < 0.10.x)
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                self.mp_face = mp.solutions.face_detection
                self.detector = self.mp_face.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5
                )
                self.use_tasks_api = False
                self._log_init_success()
        except ImportError as e:
            self._log_init_error(e)
        except Exception as e:
            self._log_init_error(e)

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None or cv2 is None:
            return []

        try:
            h, w = image.shape[:2]

            if self.use_tasks_api:
                # New Tasks API
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
                result = self.detector.detect(mp_image)

                boxes = []
                for detection in result.detections:
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    bw = bbox.width
                    bh = bbox.height
                    conf = detection.categories[0].score if detection.categories else 1.0

                    x = max(0, x)
                    y = max(0, y)
                    if bw > 10 and bh > 10:
                        boxes.append(BBox(int(x), int(y), int(bw), int(bh), confidence=float(conf)))
                return boxes
            else:
                # Legacy API
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.detector.process(rgb)

                if not results.detections:
                    return []

                boxes = []
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    conf = detection.score[0] if detection.score else 1.0

                    x = max(0, x)
                    y = max(0, y)
                    if bw > 10 and bh > 10:
                        boxes.append(BBox(x, y, bw, bh, confidence=float(conf)))
                return boxes
        except Exception as e:
            self._log_detection_error(e)
            return []

    def is_available(self) -> bool:
        return self.detector is not None

    def __del__(self):
        """Libère les ressources MediaPipe."""
        if self.detector is not None:
            try:
                if self.use_tasks_api:
                    self.detector.close()
                else:
                    self.detector.close()
            except Exception:
                pass  # Ignorer les erreurs de cleanup
