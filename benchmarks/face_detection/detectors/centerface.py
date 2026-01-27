# -*- coding: utf-8 -*-
"""
CenterFace detector.
Anchor-free face detector basé sur CenterNet.
Source: github.com/Star-Clouds/CenterFace / github.com/chenjun2hao/CenterFace.pytorch

Performance WIDER FACE:
- Easy: 93.2%
- Medium: 92.1%
- Hard: 87.3%

Ultra-rapide et léger, idéal pour le temps réel.
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class CenterFaceDetector(BaseDetector):
    """CenterFace - Anchor-free face detector."""

    name = "CenterFace"

    def __init__(self):
        super().__init__()
        self.detector = None
        self.onnx_session = None
        self._try_load_model()
        if self.is_available():
            self._log_init_success()
        else:
            self._log_init_error(ImportError("Neither centerface nor ONNX model available"))

    def _try_load_model(self):
        """Try to load CenterFace model."""
        # Method 1: Via centerface pip package
        try:
            from centerface import CenterFace
            self.detector = CenterFace()
            return
        except ImportError as e:
            self._logger.debug(f"centerface package not available: {e}")
        except Exception as e:
            self._logger.debug(f"centerface init failed: {e}")

        # Method 2: Via ONNX model
        try:
            import onnxruntime as ort
            model_path = MODELS_DIR / "centerface" / "centerface.onnx"
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
                    repo_id="public-data/centerface",
                    filename="centerface.onnx"
                )
                self.onnx_session = ort.InferenceSession(
                    model_file,
                    providers=['CPUExecutionProvider']
                )
            except Exception as e:
                self._logger.debug(f"HuggingFace download failed: {e}")
        except ImportError as e:
            self._logger.debug(f"onnxruntime not available: {e}")
        except Exception as e:
            self._logger.debug(f"ONNX load failed: {e}")

    def _preprocess(self, image: np.ndarray, target_size: int = 640):
        """Preprocess image for ONNX inference."""
        h, w = image.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        import cv2
        resized = cv2.resize(image, (new_w, new_h))

        # Pad to target_size
        padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = resized

        # Normalize and transpose to NCHW
        padded = padded.astype(np.float32) / 255.0
        padded = padded.transpose(2, 0, 1)[np.newaxis, ...]

        return padded, scale, (0, 0)

    def _decode_output(self, heatmap, scale, wh, offset, thresh=0.5):
        """Decode CenterFace output to bounding boxes."""
        h, w = heatmap.shape[2:4]
        heatmap = heatmap[0, 0]

        # Find peaks
        boxes = []
        indices = np.where(heatmap > thresh)

        for i in range(len(indices[0])):
            y_idx, x_idx = indices[0][i], indices[1][i]
            score = heatmap[y_idx, x_idx]

            # Get offset and wh
            dx = offset[0, 0, y_idx, x_idx] if offset is not None else 0
            dy = offset[0, 1, y_idx, x_idx] if offset is not None else 0
            dw = wh[0, 0, y_idx, x_idx]
            dh = wh[0, 1, y_idx, x_idx]

            cx = (x_idx + dx) * 4 / scale
            cy = (y_idx + dy) * 4 / scale
            bw = dw * 4 / scale
            bh = dh * 4 / scale

            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)

            boxes.append(BBox(x1, y1, int(bw), int(bh), confidence=float(score)))

        return boxes

    def detect(self, image: np.ndarray) -> List[BBox]:
        # Method 1: centerface package
        if self.detector is not None:
            try:
                dets, _ = self.detector(image, threshold=0.5)
                boxes = []
                for det in dets:
                    x1, y1, x2, y2, score = det[:5]
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 10 and h > 10:
                        boxes.append(BBox(int(x1), int(y1), w, h, confidence=float(score)))
                return boxes
            except Exception as e:
                self._log_detection_error(e)
                return []

        # Method 2: ONNX inference
        if self.onnx_session is not None:
            try:
                input_tensor, scale, pad = self._preprocess(image)
                input_name = self.onnx_session.get_inputs()[0].name
                outputs = self.onnx_session.run(None, {input_name: input_tensor})

                # Output interpretation depends on model version
                if len(outputs) >= 3:
                    heatmap, scale_out, offset = outputs[0], outputs[1], outputs[2]
                    return self._decode_output(heatmap, scale, scale_out, offset)
                else:
                    # Simplified output
                    heatmap = outputs[0]
                    return self._decode_output(heatmap, scale, outputs[1] if len(outputs) > 1 else None, None)
            except Exception as e:
                self._log_detection_error(e)
                return []

        return []

    def is_available(self) -> bool:
        return self.detector is not None or self.onnx_session is not None
