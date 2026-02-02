# -*- coding: utf-8 -*-
"""
Classifieurs d'émotions pour le benchmark.
"""

from .base import BaseEmotionClassifier
from .deepface_classifier import DeepFaceClassifier
from .hsemotion import HSEmotionClassifier
from .fer_pytorch import FERPytorchClassifier
from .rmn import RMNClassifier
from .vit_transformer import ViTEmotionClassifier, DeiTEmotionClassifier
from .pyfeat import PyFeatClassifier, PyFeatSVMClassifier
from .efficientnet import EfficientNetClassifier, EfficientNetB2Classifier, EfficientNetV2Classifier
from .hsemotion_onnx import HSEmotionONNXClassifier
from .poster import POSTERClassifier, POSTERAffectNetClassifier
from .dan import DANClassifier, DANAffectNetClassifier

# Registry des classifieurs
CLASSIFIER_REGISTRY = {
    # Production-ready (pip install)
    "deepface": DeepFaceClassifier,
    "hsemotion": HSEmotionClassifier,
    "hsemotion_onnx": HSEmotionONNXClassifier,
    "fer_pytorch": FERPytorchClassifier,
    "rmn": RMNClassifier,
    "vit": ViTEmotionClassifier,
    "deit": DeiTEmotionClassifier,
    "pyfeat": PyFeatClassifier,
    "pyfeat_svm": PyFeatSVMClassifier,
    "efficientnet": EfficientNetClassifier,
    "efficientnet_b2": EfficientNetB2Classifier,
    "efficientnet_v2": EfficientNetV2Classifier,
    # State-of-the-art (require manual setup)
    "poster": POSTERClassifier,
    "poster_affectnet": POSTERAffectNetClassifier,
    "dan": DANClassifier,
    "dan_affectnet": DANAffectNetClassifier,
}


def get_classifier(name: str) -> BaseEmotionClassifier:
    """
    Retourne une instance du classifieur demandé.

    Args:
        name: Nom du classifieur

    Returns:
        Instance du classifieur

    Raises:
        ValueError: Si le classifieur n'existe pas
    """
    if name not in CLASSIFIER_REGISTRY:
        available = list(CLASSIFIER_REGISTRY.keys())
        raise ValueError(f"Classifieur '{name}' inconnu. Disponibles: {available}")

    return CLASSIFIER_REGISTRY[name]()


def list_classifiers() -> dict:
    """Retourne la liste des classifieurs disponibles."""
    classifiers_info = {}
    for name, cls in CLASSIFIER_REGISTRY.items():
        try:
            instance = cls()
            classifiers_info[name] = {
                "name": instance.name,
                "description": instance.description,
                "is_available": instance.is_available(),
            }
        except Exception as e:
            classifiers_info[name] = {
                "name": name,
                "description": "Error loading",
                "is_available": False,
                "error": str(e),
            }
    return classifiers_info


def get_available_classifiers() -> list:
    """Retourne la liste des classifieurs disponibles et fonctionnels."""
    available = []
    for name, cls in CLASSIFIER_REGISTRY.items():
        try:
            instance = cls()
            if instance.is_available():
                available.append(instance)
        except Exception:
            pass
    return available


__all__ = [
    "BaseEmotionClassifier",
    "DeepFaceClassifier",
    "HSEmotionClassifier",
    "HSEmotionONNXClassifier",
    "FERPytorchClassifier",
    "RMNClassifier",
    "ViTEmotionClassifier",
    "DeiTEmotionClassifier",
    "PyFeatClassifier",
    "PyFeatSVMClassifier",
    "EfficientNetClassifier",
    "EfficientNetB2Classifier",
    "EfficientNetV2Classifier",
    "get_classifier",
    "list_classifiers",
    "get_available_classifiers",
    "CLASSIFIER_REGISTRY",
]
