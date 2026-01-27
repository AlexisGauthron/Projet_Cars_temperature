# -*- coding: utf-8 -*-
"""
SCRFD face detectors (InsightFace).
State-of-the-art, meilleur compromis précision/vitesse (ICLR 2022).

Variantes disponibles:
- SCRFD_500M: Ultra-léger (0.5 GFLOPs) - Mobile/Embarqué
- SCRFD_2.5G: Équilibré (2.5 GFLOPs) - Desktop CPU
- SCRFD_10G: Haute précision (10 GFLOPs) - Desktop GPU
- SCRFD_34G: State-of-the-art (34 GFLOPs) - Server/Cloud
"""

from typing import List, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import MODELS_DIR
from .base import BaseDetector


class BaseSCRFDDetector(BaseDetector):
    """Base class for SCRFD detectors."""

    name = "SCRFD"
    model_file: Optional[str] = None  # ONNX model filename

    def __init__(self):
        self.detector = None
        self._try_load_model()

    def _try_load_model(self):
        """Try to load the SCRFD model."""
        # Method 1: Direct ONNX loading via insightface.model_zoo
        if self.model_file:
            try:
                from insightface.model_zoo import get_model
                model_path = MODELS_DIR / "scrfd" / self.model_file
                if model_path.exists():
                    self.detector = get_model(str(model_path))
                    self.detector.prepare(ctx_id=-1, input_size=(640, 640))
                    return
            except Exception:
                pass

            # Try downloading from insightface model zoo
            try:
                from insightface.model_zoo import get_model
                # Model names in insightface format
                model_name = self.model_file.replace('.onnx', '')
                self.detector = get_model(model_name)
                self.detector.prepare(ctx_id=-1, input_size=(640, 640))
                return
            except Exception:
                pass

        # Method 2: FaceAnalysis with specific model pack
        try:
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(
                name='buffalo_sc',
                allowed_modules=['detection'],
                providers=['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self.detector = self.app
            return
        except Exception:
            pass

    def detect(self, image: np.ndarray) -> List[BBox]:
        if self.detector is None:
            return []
        try:
            # Check if using FaceAnalysis or direct model
            if hasattr(self.detector, 'get'):
                # FaceAnalysis interface
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
            else:
                # Direct model interface
                bboxes, kpss = self.detector.detect(image)
                boxes = []
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2, conf = bbox
                    w, h = int(x2 - x1), int(y2 - y1)
                    if w > 0 and h > 0:
                        boxes.append(BBox(int(x1), int(y1), w, h, confidence=float(conf)))
                return boxes
        except Exception:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class SCRFDDetector(BaseSCRFDDetector):
    """SCRFD default detector (buffalo_sc / 500M)."""
    name = "SCRFD"
    model_file = None  # Uses FaceAnalysis default


class SCRFD500MDetector(BaseSCRFDDetector):
    """SCRFD 500M - Ultra-léger pour mobile/embarqué.

    Performance WIDER FACE:
    - Easy: 90.57%
    - Medium: 88.12%
    - Hard: 68.51%
    """
    name = "SCRFD_500M"
    model_file = "scrfd_500m_bnkps.onnx"


class SCRFD2500MDetector(BaseSCRFDDetector):
    """SCRFD 2.5G - Équilibré pour desktop CPU.

    Performance WIDER FACE:
    - Easy: 93.78%
    - Medium: 92.16%
    - Hard: 77.87%
    """
    name = "SCRFD_2.5G"
    model_file = "scrfd_2.5g_bnkps.onnx"


class SCRFD10GDetector(BaseSCRFDDetector):
    """SCRFD 10G - Haute précision pour desktop GPU.

    Performance WIDER FACE:
    - Easy: 95.16%
    - Medium: 93.87%
    - Hard: 83.05%
    """
    name = "SCRFD_10G"
    model_file = "scrfd_10g_bnkps.onnx"


class SCRFD34GDetector(BaseSCRFDDetector):
    """SCRFD 34G - State-of-the-art pour serveur/cloud.

    Performance WIDER FACE:
    - Easy: 96.06%
    - Medium: 94.92%
    - Hard: 85.29%
    """
    name = "SCRFD_34G"
    model_file = "scrfd_34g.onnx"
