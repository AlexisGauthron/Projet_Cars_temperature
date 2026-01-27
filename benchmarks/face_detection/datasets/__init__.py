# -*- coding: utf-8 -*-
"""
Dataset loaders pour le benchmark de détection de visage.
"""

from .base import BaseDataset
from .wider_face import WiderFaceDataset
from .mafa import MAFADataset
from .sviro import SVIRODataset, SVIROBMWi3Dataset, SVIROTeslaDataset
from .sviro_subset import SVIROSubsetDataset

# Registry des datasets
DATASET_REGISTRY = {
    "wider_face": WiderFaceDataset,
    "mafa": MAFADataset,
    "sviro": SVIRODataset,
    "sviro_bmw_i3": SVIROBMWi3Dataset,
    "sviro_tesla": SVIROTeslaDataset,
    "sviro_subset": SVIROSubsetDataset,
}


def get_dataset(name: str) -> BaseDataset:
    """
    Retourne une instance du dataset demandé.

    Args:
        name: Nom du dataset (wider_face, fddb, etc.)

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
    "BaseDataset",
    "WiderFaceDataset",
    "MAFADataset",
    "SVIRODataset",
    "SVIROBMWi3Dataset",
    "SVIROTeslaDataset",
    "SVIROSubsetDataset",
    "get_dataset",
    "list_datasets",
    "DATASET_REGISTRY",
]
