# -*- coding: utf-8 -*-
"""
Dataset loader pour SVIRO (Synthetic Vehicle Interior Rear Seat Occupancy).

⚠️ NOTE IMPORTANTE:
SVIRO est un dataset de détection d'OCCUPANTS de véhicules, pas de visages.
Les bounding boxes correspondent aux personnes assises, pas aux visages.
Ce loader est utile pour:
- Tester la détection de visages dans un contexte véhicule (infrarouge)
- Évaluer les performances sans ground truth face (mode "detection only")
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseDataset
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox


# Répertoire racine du dataset
SVIRO_DIR = Path(__file__).parent / "sviro"


class SVIRODataset(BaseDataset):
    """
    Loader pour le dataset SVIRO.

    SVIRO contient des images synthétiques d'intérieurs de véhicules
    avec annotations de bounding boxes pour les occupants.
    """

    name = "SVIRO"
    description = "Synthetic Vehicle Interior - 10 véhicules, images infrarouge"

    def __init__(self, vehicle: Optional[str] = None):
        """
        Initialise le dataset SVIRO.

        Args:
            vehicle: Filtrer par véhicule spécifique (ex: "bmw_i3", "tesla_model3")
                    Si None, charge toutes les images.
        """
        super().__init__()
        self.vehicle_filter = vehicle
        self._sviro_dir = SVIRO_DIR

    @property
    def annotation_file(self) -> Path:
        """Chemin vers le fichier d'index des images."""
        return self._sviro_dir / "index.txt"

    @property
    def images_dir(self) -> Path:
        """Chemin vers le dossier des images."""
        return self._sviro_dir / "images"

    def is_available(self) -> bool:
        """Vérifie si le dataset est disponible."""
        return self.images_dir.exists() and any(self.images_dir.glob("*.png"))

    def load_annotations(self) -> Dict[str, List[BBox]]:
        """
        Charge les annotations du dataset.

        Note: SVIRO a des annotations d'occupants, pas de visages.
        On retourne des listes vides de BBox pour permettre le benchmarking
        en mode "detection only" (sans calcul de precision/recall).
        """
        annotations = {}

        if not self.images_dir.exists():
            return annotations

        # Lister toutes les images
        image_extensions = ['.png', '.jpg', '.jpeg']
        for ext in image_extensions:
            for img_path in self.images_dir.glob(f"*{ext}"):
                # Filtrer par véhicule si spécifié
                if self.vehicle_filter:
                    if not img_path.name.startswith(self.vehicle_filter):
                        continue

                # Chemin relatif pour la clé
                rel_path = img_path.name

                # Charger les annotations si disponibles
                bboxes = self._load_image_annotations(img_path)
                annotations[rel_path] = bboxes

        return annotations

    def _load_image_annotations(self, image_path: Path) -> List[BBox]:
        """
        Charge les annotations pour une image spécifique.

        Cherche dans le dossier annotations/<vehicle>/ pour les fichiers
        correspondants.
        """
        bboxes = []

        # Extraire le nom du véhicule depuis le nom de l'image
        # Format: vehicle_originalname.png
        parts = image_path.stem.split('_', 1)
        if len(parts) < 2:
            return bboxes

        vehicle = parts[0]
        original_name = parts[1]

        # Chercher le fichier d'annotations
        annotations_dir = self._sviro_dir / "annotations" / vehicle

        if not annotations_dir.exists():
            return bboxes

        # Essayer différents formats d'annotations
        # Format YOLO: image_name.txt avec format "class x_center y_center width height"
        yolo_file = annotations_dir / f"{original_name}.txt"
        if yolo_file.exists():
            bboxes = self._parse_yolo_annotations(yolo_file, image_path)

        # Format CSV
        csv_file = annotations_dir / "annotations.csv"
        if csv_file.exists() and not bboxes:
            bboxes = self._parse_csv_annotations(csv_file, original_name, image_path)

        return bboxes

    def _parse_yolo_annotations(self, txt_file: Path, image_path: Path) -> List[BBox]:
        """Parse les annotations au format YOLO."""
        import cv2
        bboxes = []

        try:
            # Lire les dimensions de l'image
            img = cv2.imread(str(image_path))
            if img is None:
                return bboxes
            h, w = img.shape[:2]

            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_w = float(parts[3]) * w
                        box_h = float(parts[4]) * h

                        x = x_center - box_w / 2
                        y = y_center - box_h / 2

                        bboxes.append(BBox(
                            x=int(x),
                            y=int(y),
                            w=int(box_w),
                            h=int(box_h)
                        ))

        except Exception as e:
            pass

        return bboxes

    def _parse_csv_annotations(self, csv_file: Path, image_name: str, image_path: Path) -> List[BBox]:
        """Parse les annotations au format CSV."""
        bboxes = []

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Chercher la ligne correspondant à notre image
                    if row.get('image', '').find(image_name) != -1:
                        bboxes.append(BBox(
                            x=int(float(row.get('x', 0))),
                            y=int(float(row.get('y', 0))),
                            w=int(float(row.get('width', 0))),
                            h=int(float(row.get('height', 0)))
                        ))

        except Exception as e:
            pass

        return bboxes

    def get_image_path(self, rel_path: str) -> Optional[Path]:
        """Résout le chemin complet d'une image."""
        image_path = self.images_dir / rel_path
        if image_path.exists():
            return image_path
        return None

    def get_vehicles(self) -> List[str]:
        """Retourne la liste des véhicules disponibles."""
        vehicles = set()
        if self.images_dir.exists():
            for img in self.images_dir.glob("*.png"):
                parts = img.stem.split('_', 1)
                if parts:
                    vehicles.add(parts[0])
        return sorted(vehicles)

    def get_stats(self) -> dict:
        """Retourne les statistiques du dataset."""
        stats = super().get_stats()
        stats["vehicles"] = self.get_vehicles()
        stats["note"] = "Annotations: occupants (pas visages)"
        return stats


# Sous-classes pour filtrer par véhicule
class SVIROBMWi3Dataset(SVIRODataset):
    name = "SVIRO-BMW-i3"
    description = "SVIRO - BMW i3 uniquement"
    def __init__(self):
        super().__init__(vehicle="bmw_i3")


class SVIROTeslaDataset(SVIRODataset):
    name = "SVIRO-Tesla"
    description = "SVIRO - Tesla Model 3 uniquement"
    def __init__(self):
        super().__init__(vehicle="tesla_model3")
