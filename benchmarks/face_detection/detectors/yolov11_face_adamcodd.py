# -*- coding: utf-8 -*-
"""
YOLOv11 Face detector - AdamCodd variant.
Entraîné sur WIDERFACE (225 epochs).
Easy AP: 94.2%, Medium: 92.1%, Hard: 81.0%.
Source: https://huggingface.co/AdamCodd/YOLOv11n-face-detection
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class YOLOv11FaceAdamCoddDetector(BaseDetector):
    """YOLOv11 Face detector - AdamCodd variant (WIDERFACE trained)."""

    name = "YOLOv11-face-AdamCodd"

    def __init__(self):
        super().__init__()
        self.model = None

        try:
            from ultralytics import YOLO

            # Essayer de charger un modèle local
            model_path = MODELS_DIR / "yolov11" / "yolov11n-face-adamcodd.pt"
            if model_path.exists():
                self.model = YOLO(str(model_path))
                self._log_init_success()
            else:
                # Télécharger depuis HuggingFace (AdamCodd)
                try:
                    from huggingface_hub import hf_hub_download
                    model_file = hf_hub_download(
                        repo_id="AdamCodd/YOLOv11n-face-detection",
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
