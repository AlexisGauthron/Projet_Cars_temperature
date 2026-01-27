# -*- coding: utf-8 -*-
"""
Dataset loader pour SVIRO Subset (1000 images JPG).

Version allégée du dataset SVIRO pour inclusion dans git.
- 100 images par véhicule (10 véhicules)
- Format JPG qualité 95%
- Annotations bounding boxes pour occupants
"""

from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox


# Répertoire racine du dataset
SVIRO_SUBSET_DIR = Path(__file__).parent / "sviro_subset"


class SVIROSubsetDataset(BaseDataset):
    """
    Loader pour le dataset SVIRO Subset (1000 images).

    Contient 100 images par véhicule pour 10 véhicules différents.
    Format: JPG qualité 95%, annotations txt.
    """

    name = "SVIRO-Subset"
    description = "SVIRO subset - 1000 images JPG (100 par véhicule)"

    def __init__(self, vehicle: Optional[str] = None):
        """
        Initialise le dataset SVIRO Subset.

        Args:
            vehicle: Filtrer par véhicule spécifique (ex: "bmw_i3", "tesla_model3")
        """
        super().__init__()
        self.vehicle_filter = vehicle
        self._sviro_dir = SVIRO_SUBSET_DIR

    @property
    def annotation_file(self) -> Path:
        return self._sviro_dir / "index.txt"

    @property
    def images_dir(self) -> Path:
        return self._sviro_dir / "images"

    @property
    def annotations_dir(self) -> Path:
        return self._sviro_dir / "annotations"

    def is_available(self) -> bool:
        return self.images_dir.exists() and any(self.images_dir.glob("*.jpg"))

    def load_annotations(self) -> Dict[str, List[BBox]]:
        """Charge les annotations du dataset."""
        annotations = {}

        if not self.images_dir.exists():
            return annotations

        for img_path in self.images_dir.glob("*.jpg"):
            # Filtrer par véhicule si spécifié
            if self.vehicle_filter:
                if not img_path.name.startswith(self.vehicle_filter):
                    continue

            rel_path = img_path.name
            bboxes = self._load_image_annotations(img_path)
            annotations[rel_path] = bboxes

        return annotations

    def _load_image_annotations(self, image_path: Path) -> List[BBox]:
        """Charge les annotations pour une image."""
        bboxes = []

        # Construire le nom du fichier annotation
        # Image: bmw_i3_i3_test_imageID_152_GT_5_0_0.jpg
        # Annotation: i3_test_imageID_152_GT_5_0_0.txt
        img_name = image_path.stem

        # Retirer le préfixe véhicule (bmw_i3_, tesla_model3_, etc.)
        prefixes = [
            "bmw_i3_", "bmw_x5_", "ford_escape_", "hyundai_tucson_",
            "lexus_gsf_", "mercedes_a_", "renault_zoe_", "tesla_model3_",
            "toyota_hilux_", "vw_tiguan_"
        ]

        annotation_name = img_name
        for prefix in prefixes:
            if img_name.startswith(prefix):
                annotation_name = img_name[len(prefix):]
                break

        # Chercher le fichier annotation
        annotation_file = self.annotations_dir / f"{annotation_name}.txt"

        if annotation_file.exists():
            bboxes = self._parse_bbox_file(annotation_file, image_path)

        return bboxes

    def _parse_bbox_file(self, txt_file: Path, image_path: Path) -> List[BBox]:
        """Parse le fichier d'annotations (format: x,y,w,h ou x1,y1,x2,y2,score)."""
        bboxes = []

        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        # Format: x1,y1,x2,y2 ou x,y,w,h
                        values = [int(float(p)) for p in parts[:4]]

                        # Détecter le format
                        if values[2] > values[0] and values[3] > values[1]:
                            # Format x1,y1,x2,y2
                            x, y = values[0], values[1]
                            w, h = values[2] - values[0], values[3] - values[1]
                        else:
                            # Format x,y,w,h
                            x, y, w, h = values

                        if w > 0 and h > 0:
                            bboxes.append(BBox(x=x, y=y, w=w, h=h))

        except Exception:
            pass

        return bboxes

    def get_image_path(self, rel_path: str) -> Optional[Path]:
        image_path = self.images_dir / rel_path
        if image_path.exists():
            return image_path
        return None

    def get_vehicles(self) -> List[str]:
        """Retourne la liste des véhicules disponibles."""
        vehicles = set()
        if self.images_dir.exists():
            for img in self.images_dir.glob("*.jpg"):
                # Extraire le préfixe véhicule
                name = img.stem
                for v in ["bmw_i3", "bmw_x5", "ford_escape", "hyundai_tucson",
                         "lexus_gsf", "mercedes_a", "renault_zoe", "tesla_model3",
                         "toyota_hilux", "vw_tiguan"]:
                    if name.startswith(v):
                        vehicles.add(v)
                        break
        return sorted(vehicles)

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats["vehicles"] = self.get_vehicles()
        stats["format"] = "JPG 95%"
        return stats
