# -*- coding: utf-8 -*-
"""
Détecteurs de visage pour le benchmark.

Modèles supportés:
- YuNet: Léger, robuste aux occlusions (OpenCV)
- OpenCV-DNN: Meilleur compromis (SSD ResNet-10)
- Haar: Ultra-rapide, précision limitée
- MTCNN: Standard académique
- RetinaFace: Haute précision
- SCRFD: State-of-the-art InsightFace (ICLR 2022)
- YOLO5Face: SOTA WIDER FACE (86.6% Hard)
- YOLOv8-face: YOLO moderne
- YOLOv9-face: 2024 (lindevs)
- YOLOv10-face: NeurIPS 2024
- YOLOv11-face: 2024
- YOLOv12-face: Latest (2026)
- MediaPipe: Google BlazeFace (mobile)
- DLib-HOG: Classique rapide
- DLib-CNN: Classique précis
- CenterFace: Anchor-free, rapide
- DSFD: Dual Shot Face Detector (CVPR 2019)
- FaceBoxes: CPU-friendly, léger
- TinaFace: SOTA WIDER Hard (92.1%)
"""

from .base import BaseDetector
from .yunet import YuNetDetector
from .opencv_dnn import OpenCVDNNDetector
from .haar import HaarDetector
from .mtcnn import MTCNNDetector
from .retinaface import RetinaFaceDetector
from .scrfd import (
    SCRFDDetector,
    SCRFD500MDetector,
    SCRFD2500MDetector,
    SCRFD10GDetector,
    SCRFD34GDetector,
)
from .yolov8_face import YOLOv8FaceDetector
from .yolov9_face import YOLOv9FaceDetector
from .yolov10_face import YOLOv10FaceDetector
from .yolov11_face import YOLOv11FaceDetector
from .yolov11_face_adamcodd import YOLOv11FaceAdamCoddDetector
from .yolov12_face import YOLOv12FaceDetector
from .yolo26_face import YOLO26FaceDetector
from .yolo5face import YOLO5FaceDetector
from .mediapipe_face import MediaPipeFaceDetector
from .dlib_detector import DLibHOGDetector, DLibCNNDetector
from .centerface import CenterFaceDetector
from .dsfd import DSFDDetector
from .faceboxes import FaceBoxesDetector
from .tinaface import TinaFaceDetector

# Registry des détecteurs (nom -> classe)
# Organisé par catégorie pour plus de clarté
DETECTOR_REGISTRY = {
    # === Modèles légers / Embarqué ===
    "YuNet": YuNetDetector,
    "Haar": HaarDetector,
    "MediaPipe": MediaPipeFaceDetector,
    "FaceBoxes": FaceBoxesDetector,  # CPU-friendly
    "CenterFace": CenterFaceDetector,  # Anchor-free, rapide

    # === Modèles équilibrés ===
    "OpenCV-DNN": OpenCVDNNDetector,
    "DLib-HOG": DLibHOGDetector,

    # === Modèles YOLO (rapides et précis) ===
    "YOLO5Face": YOLO5FaceDetector,
    "YOLOv8-face": YOLOv8FaceDetector,
    "YOLOv9-face": YOLOv9FaceDetector,  # 2024 (lindevs)
    "YOLOv10-face": YOLOv10FaceDetector,  # NeurIPS 2024
    "YOLOv11-face": YOLOv11FaceDetector,  # deepghs variant
    "YOLOv11-face-AdamCodd": YOLOv11FaceAdamCoddDetector,  # WIDERFACE trained
    "YOLOv12-face": YOLOv12FaceDetector,  # Latest 2026
    "YOLO26": YOLO26FaceDetector,  # Généraliste (Janvier 2026)

    # === Modèles haute précision ===
    "SCRFD": SCRFDDetector,
    "SCRFD_500M": SCRFD500MDetector,
    "SCRFD_2.5G": SCRFD2500MDetector,
    "SCRFD_10G": SCRFD10GDetector,
    "SCRFD_34G": SCRFD34GDetector,
    "RetinaFace": RetinaFaceDetector,
    "MTCNN": MTCNNDetector,
    "DLib-CNN": DLibCNNDetector,
    "DSFD": DSFDDetector,  # CVPR 2019 - 90.4% Hard
    "TinaFace": TinaFaceDetector,  # SOTA 92.1% Hard
}


def get_detector(name: str) -> BaseDetector:
    """
    Retourne une instance du détecteur demandé.

    Args:
        name: Nom du détecteur

    Returns:
        Instance du détecteur

    Raises:
        ValueError: Si le détecteur n'existe pas
    """
    if name not in DETECTOR_REGISTRY:
        available = list(DETECTOR_REGISTRY.keys())
        raise ValueError(f"Détecteur '{name}' inconnu. Disponibles: {available}")

    return DETECTOR_REGISTRY[name]()


def get_all_detectors() -> list:
    """Retourne tous les détecteurs disponibles (initialisés et fonctionnels)."""
    detectors = []
    for cls in DETECTOR_REGISTRY.values():
        try:
            detector = cls()
            if detector.is_available():
                detectors.append(detector)
        except Exception:
            pass
    return detectors


def list_detectors() -> dict:
    """Retourne la liste des détecteurs avec leurs infos."""
    detectors_info = {}
    for name, cls in DETECTOR_REGISTRY.items():
        try:
            instance = cls()
            detectors_info[name] = {
                "name": instance.name,
                "is_available": instance.is_available(),
            }
        except Exception:
            detectors_info[name] = {
                "name": name,
                "is_available": False,
            }
    return detectors_info


__all__ = [
    # Base
    "BaseDetector",
    # Détecteurs
    "YuNetDetector",
    "OpenCVDNNDetector",
    "HaarDetector",
    "MTCNNDetector",
    "RetinaFaceDetector",
    "SCRFDDetector",
    "SCRFD500MDetector",
    "SCRFD2500MDetector",
    "SCRFD10GDetector",
    "SCRFD34GDetector",
    "YOLO5FaceDetector",
    "YOLOv8FaceDetector",
    "YOLOv9FaceDetector",
    "YOLOv10FaceDetector",
    "YOLOv11FaceDetector",
    "YOLOv11FaceAdamCoddDetector",
    "YOLOv12FaceDetector",
    "YOLO26FaceDetector",
    "MediaPipeFaceDetector",
    "DLibHOGDetector",
    "DLibCNNDetector",
    "CenterFaceDetector",
    "DSFDDetector",
    "FaceBoxesDetector",
    "TinaFaceDetector",
    # Fonctions
    "get_detector",
    "get_all_detectors",
    "list_detectors",
    "DETECTOR_REGISTRY",
]
