# -*- coding: utf-8 -*-
"""
FaceBoxes detector.
CPU-friendly face detector optimisé pour les appareils embarqués.
Source: github.com/zisianw/FaceBoxes.PyTorch

Performance WIDER FACE:
- Easy: 96.0%
- Medium: 95.1%
- Hard: 89.7%

Léger et rapide, idéal pour CPU et edge devices.
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class FaceBoxesDetector(BaseDetector):
    """FaceBoxes - CPU-friendly face detector."""

    name = "FaceBoxes"

    def __init__(self):
        self.detector = None
        self.onnx_session = None
        self._try_load_model()

    def _try_load_model(self):
        """Try to load FaceBoxes model."""
        # Method 1: Via face_detection package (hukkelas)
        try:
            import face_detection
            self.detector = face_detection.build_detector(
                "FaceBoxesDetector",
                confidence_threshold=0.5,
                nms_iou_threshold=0.3
            )
            return
        except (ImportError, ValueError):
            pass
        except Exception:
            pass

        # Method 2: Via ONNX model
        try:
            import onnxruntime as ort
            model_path = MODELS_DIR / "faceboxes" / "faceboxes.onnx"
            if model_path.exists():
                self.onnx_session = ort.InferenceSession(
                    str(model_path),
                    providers=['CPUExecutionProvider']
                )
                return
            # Try downloading from HuggingFace
            try:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(
                    repo_id="public-data/faceboxes",
                    filename="faceboxes.onnx"
                )
                self.onnx_session = ort.InferenceSession(
                    model_file,
                    providers=['CPUExecutionProvider']
                )
            except Exception:
                pass
        except ImportError:
            pass
        except Exception:
            pass

    def _preprocess(self, image: np.ndarray, target_size: int = 1024):
        """Preprocess image for FaceBoxes ONNX."""
        import cv2
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Convert to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32)

        # Normalize (ImageNet mean)
        mean = np.array([104, 117, 123], dtype=np.float32)
        rgb -= mean

        # Transpose to NCHW
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]

        return tensor, scale

    def detect(self, image: np.ndarray) -> List[BBox]:
        # Method 1: face_detection package
        if self.detector is not None:
            try:
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
            except Exception:
                return []

        # Method 2: ONNX inference
        if self.onnx_session is not None:
            try:
                input_tensor, scale = self._preprocess(image)
                input_name = self.onnx_session.get_inputs()[0].name
                outputs = self.onnx_session.run(None, {input_name: input_tensor})

                # Decode output (format depends on ONNX export)
                detections = outputs[0]  # Typically [batch, num_detections, 5]
                boxes = []
                for det in detections[0]:
                    x1, y1, x2, y2, conf = det[:5]
                    if conf < 0.5:
                        continue
                    # Scale back to original image
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 10 and h > 10:
                        boxes.append(BBox(int(x1), int(y1), w, h, confidence=float(conf)))
                return boxes
            except Exception:
                return []

        return []

    def is_available(self) -> bool:
        return self.detector is not None or self.onnx_session is not None
