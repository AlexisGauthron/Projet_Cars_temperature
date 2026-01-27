# -*- coding: utf-8 -*-
"""
TinaFace detector.
State-of-the-art face detector sur WIDER FACE Hard (CVPR 2021).
Source: github.com/Media-Smart/vedadet (mmdetection)

Performance WIDER FACE:
- Easy: 96.3%
- Medium: 95.6%
- Hard: 92.1%  ← Record sur Hard !

Le meilleur détecteur sur les petits visages et cas difficiles.
Note: Requiert mmdetection ou un modèle ONNX pré-converti.
"""

from typing import List, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class TinaFaceDetector(BaseDetector):
    """TinaFace - SOTA on WIDER FACE Hard (92.1%)."""

    name = "TinaFace"

    def __init__(self):
        super().__init__()
        self.detector = None
        self.onnx_session = None
        self.cfg = None
        self._try_load_model()
        if self.is_available():
            self._log_init_success()
        else:
            self._log_init_error(ImportError("Neither ONNX nor mmdetection model available"))

    def _try_load_model(self):
        """Try to load TinaFace model."""
        # Method 1: Via ONNX model (preferred, simpler)
        try:
            import onnxruntime as ort
            model_path = MODELS_DIR / "tinaface" / "tinaface.onnx"
            if model_path.exists():
                self.onnx_session = ort.InferenceSession(
                    str(model_path),
                    providers=['CPUExecutionProvider']
                )
                return
            # Try HuggingFace
            try:
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(
                    repo_id="public-data/tinaface",
                    filename="tinaface.onnx"
                )
                self.onnx_session = ort.InferenceSession(
                    model_file,
                    providers=['CPUExecutionProvider']
                )
                return
            except Exception as e:
                self._logger.debug(f"HuggingFace download failed: {e}")
        except ImportError as e:
            self._logger.debug(f"onnxruntime not available: {e}")
        except Exception as e:
            self._logger.debug(f"ONNX load failed: {e}")

        # Method 2: Via mmdetection/vedadet (complex)
        try:
            from mmdet.apis import init_detector, inference_detector

            config_path = MODELS_DIR / "tinaface" / "tinaface_r50_fpn.py"
            checkpoint_path = MODELS_DIR / "tinaface" / "tinaface_r50_fpn.pth"

            if config_path.exists() and checkpoint_path.exists():
                self.detector = init_detector(
                    str(config_path),
                    str(checkpoint_path),
                    device='cpu'
                )
                return
        except ImportError as e:
            self._logger.debug(f"mmdetection not available: {e}")
        except Exception as e:
            self._logger.debug(f"mmdetection init failed: {e}")

    def _preprocess(self, image: np.ndarray, target_size: int = 1100):
        """Preprocess image for ONNX inference."""
        import cv2
        h, w = image.shape[:2]

        # Multi-scale inference for best accuracy
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize (ImageNet)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        rgb = (rgb.astype(np.float32) - mean) / std

        # Transpose to NCHW
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]

        return tensor.astype(np.float32), scale

    def _decode_output(self, outputs, scale: float, conf_thresh: float = 0.5) -> List[BBox]:
        """Decode ONNX output to bounding boxes."""
        boxes = []

        # Output format depends on ONNX export configuration
        if len(outputs) >= 2:
            # Format: [boxes, scores] or [boxes_scores]
            detections = outputs[0]
            for det in detections:
                if len(det) >= 5:
                    x1, y1, x2, y2, conf = det[:5]
                else:
                    continue
                if conf < conf_thresh:
                    continue
                # Scale back to original image
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                w, h = int(x2 - x1), int(y2 - y1)
                if w > 10 and h > 10:
                    boxes.append(BBox(int(x1), int(y1), w, h, confidence=float(conf)))
        else:
            # Single output with format [N, 5] or [N, 6]
            detections = outputs[0]
            if len(detections.shape) == 3:
                detections = detections[0]  # Remove batch dim
            for det in detections:
                if len(det) >= 5:
                    x1, y1, x2, y2, conf = det[:5]
                    if conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 10 and h > 10:
                        boxes.append(BBox(int(x1), int(y1), w, h, confidence=float(conf)))

        return boxes

    def detect(self, image: np.ndarray) -> List[BBox]:
        # Method 1: ONNX inference
        if self.onnx_session is not None:
            try:
                input_tensor, scale = self._preprocess(image)
                input_name = self.onnx_session.get_inputs()[0].name
                outputs = self.onnx_session.run(None, {input_name: input_tensor})
                return self._decode_output(outputs, scale)
            except Exception as e:
                self._log_detection_error(e)
                return []

        # Method 2: mmdetection
        if self.detector is not None:
            try:
                from mmdet.apis import inference_detector
                result = inference_detector(self.detector, image)

                boxes = []
                # mmdetection v2 format
                if isinstance(result, tuple):
                    result = result[0]
                # result is a list of arrays, one per class
                face_dets = result[0] if len(result) > 0 else []
                for det in face_dets:
                    x1, y1, x2, y2, conf = det[:5]
                    if conf < 0.5:
                        continue
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 10 and h > 10:
                        boxes.append(BBox(int(x1), int(y1), w, h, confidence=float(conf)))
                return boxes
            except Exception as e:
                self._log_detection_error(e)
                return []

        return []

    def is_available(self) -> bool:
        return self.detector is not None or self.onnx_session is not None
