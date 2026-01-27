# -*- coding: utf-8 -*-
"""
Classe de base pour les dataset loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox


class BaseDataset(ABC):
    """Interface commune pour tous les datasets."""

    name: str = "BaseDataset"
    description: str = "Dataset de base"

    def __init__(self):
        """Initialise le dataset."""
        self._annotations: Optional[Dict[str, List[BBox]]] = None

    @property
    @abstractmethod
    def annotation_file(self) -> Path:
        """Chemin vers le fichier d'annotations."""
        pass

    @property
    @abstractmethod
    def images_dir(self) -> Path:
        """Chemin vers le dossier des images."""
        pass

    @abstractmethod
    def load_annotations(self) -> Dict[str, List[BBox]]:
        """
        Charge et retourne les annotations du dataset.

        Returns:
            Dict[image_path, List[BBox]]: Mapping image -> bounding boxes
        """
        pass

    def is_available(self) -> bool:
        """Vérifie si le dataset est disponible (fichiers présents)."""
        return self.annotation_file.exists() and self.images_dir.exists()

    def get_annotations(self) -> Dict[str, List[BBox]]:
        """Retourne les annotations (charge si nécessaire)."""
        if self._annotations is None:
            self._annotations = self.load_annotations()
        return self._annotations

    def get_image_path(self, rel_path: str) -> Optional[Path]:
        """
        Résout le chemin complet d'une image.

        Args:
            rel_path: Chemin relatif de l'image dans les annotations

        Returns:
            Path complet si trouvé, None sinon
        """
        # Essayer le chemin direct
        image_path = self.images_dir / rel_path
        if image_path.exists():
            return image_path

        # Certains datasets ont une structure imbriquée
        # Ex: WIDER_val/images/<event>/<image>
        for subdir in ["WIDER_val/images", "WIDER_train/images", "images"]:
            alt_path = self.images_dir / subdir / rel_path
            if alt_path.exists():
                return alt_path

        return None

    def get_stats(self) -> dict:
        """Retourne les statistiques du dataset."""
        annotations = self.get_annotations()
        total_faces = sum(len(boxes) for boxes in annotations.values())
        return {
            "name": self.name,
            "total_images": len(annotations),
            "total_faces": total_faces,
            "avg_faces_per_image": total_faces / len(annotations) if annotations else 0,
        }

    def __repr__(self) -> str:
        status = "✓" if self.is_available() else "✗"
        return f"{self.__class__.__name__}(name='{self.name}', available={status})"
