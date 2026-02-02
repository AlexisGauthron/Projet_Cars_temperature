# -*- coding: utf-8 -*-
"""
Dataset loaders pour le benchmark de classification d'émotions.
"""

from .base import BaseEmotionDataset
from .fer2013 import FER2013Dataset
from .affectnet import AffectNetDataset
from .rafdb import RAFDBDataset
from .ckplus import CKPlusDataset, CKPlusAllFramesDataset
from .ferplus import FERPlusDataset, FERPlusSoftLabelDataset
from .expw import ExpWDataset

# Registry des datasets
DATASET_REGISTRY = {
    "fer2013": FER2013Dataset,
    "affectnet": AffectNetDataset,
    "rafdb": RAFDBDataset,
    "ckplus": CKPlusDataset,
    "ckplus_all": CKPlusAllFramesDataset,
    "ferplus": FERPlusDataset,
    "ferplus_soft": FERPlusSoftLabelDataset,
    "expw": ExpWDataset,
}


def get_dataset(name: str) -> BaseEmotionDataset:
    """
    Retourne une instance du dataset demandé.

    Args:
        name: Nom du dataset (fer2013, affectnet, rafdb)

    Returns:
        Instance du dataset loader

    Raises:
        ValueError: Si le dataset n'existe pas
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(f"Dataset '{name}' inconnu. Disponibles: {available}")

    return DATASET_REGISTRY[name]()


def list_datasets() -> dict:
    """Retourne la liste des datasets disponibles avec leurs infos."""
    datasets_info = {}
    for name, cls in DATASET_REGISTRY.items():
        instance = cls()
        datasets_info[name] = {
            "name": instance.name,
            "description": instance.description,
            "is_available": instance.is_available(),
        }
    return datasets_info


__all__ = [
    "BaseEmotionDataset",
    "FER2013Dataset",
    "AffectNetDataset",
    "RAFDBDataset",
    "CKPlusDataset",
    "CKPlusAllFramesDataset",
    "FERPlusDataset",
    "FERPlusSoftLabelDataset",
    "ExpWDataset",
    "get_dataset",
    "list_datasets",
    "DATASET_REGISTRY",
]
