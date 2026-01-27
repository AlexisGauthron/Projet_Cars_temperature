# -*- coding: utf-8 -*-
"""
YOLO5Face detector.
"Why Reinventing a Face Detector" - arXiv:2105.12931
State-of-the-art sur WIDER FACE: 96.67% Easy, 95.08% Medium, 86.55% Hard.
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class YOLO5FaceDetector(BaseDetector):
    """YOLO5Face detector - State-of-the-art face detection."""

    name = "YOLO5Face"

    def __init__(self):
        super().__init__()
        self.model = None

        try:
            from ultralytics import YOLO

            # Essayer de charger un modèle local
            model_path = MODELS_DIR / "yolo5face" / "yolov5n-face.pt"
            if model_path.exists():
                self.model = YOLO(str(model_path))
                self._log_init_success()
            else:
                # Alternative: utiliser le modèle YOLOv8-face comme fallback
                # car YOLO5Face original nécessite une installation spécifique
                try:
                    from huggingface_hub import hf_hub_download
                    # Essayer d'abord yolo5face si disponible
                    try:
                        model_file = hf_hub_download(
                            repo_id="akanametov/yolo5-face",
                            filename="yolov5n-face.pt"
                        )
                        self.model = YOLO(model_file)
                        self._log_init_success()
                    except Exception as e:
                        self._logger.debug(f"yolo5-face download failed: {e}")
                        # Fallback sur YOLOv8-face
                        model_file = hf_hub_download(
                            repo_id="arnabdhar/YOLOv8-Face-Detection",
                            filename="model.pt"
                        )
                        self.model = YOLO(model_file)
                        self._log_init_success()
                except Exception as e:
                    self._log_init_error(e)
        except ImportError as e:
            self._log_init_error(e)
        except Exception as e:
            self._log_init_error(e)

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.model is None:
            return []
        try:
            results = self.model(image, verbose=False)
            boxes = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
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
