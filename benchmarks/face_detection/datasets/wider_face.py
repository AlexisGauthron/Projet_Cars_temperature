# -*- coding: utf-8 -*-
"""
WIDER FACE dataset loader.
Dataset de référence pour le benchmark de détection de visage.
"""

from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import ANNOTATIONS_DIR, DATASETS_DIR
from .base import BaseDataset


class WiderFaceDataset(BaseDataset):
    """Loader pour le dataset WIDER FACE."""

    name = "WIDER FACE"
    description = "Dataset de référence pour la détection de visage (3226 images validation)"

    @property
    def annotation_file(self) -> Path:
        return ANNOTATIONS_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt"

    @property
    def images_dir(self) -> Path:
        return DATASETS_DIR / "wider_face"

    def load_annotations(self) -> Dict[str, List[BBox]]:
        """
        Charge les annotations WIDER FACE.

        Format du fichier:
            image_path
            num_faces
            x y w h blur expression illumination invalid occlusion pose
            ...

        Attributs WIDER FACE:
            - blur: 0=clear, 1=normal, 2=heavy
            - expression: 0=typical, 1=exaggerate
            - illumination: 0=normal, 1=extreme
            - invalid: 0=valid, 1=invalid
            - occlusion: 0=none, 1=partial, 2=heavy
            - pose: 0=typical, 1=atypical
        """
        annotations = {}

        with open(self.annotation_file, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            # Ligne 1: chemin de l'image
            image_path = lines[i].strip()
            i += 1

            if i >= len(lines):
                break

            # Ligne 2: nombre de visages
            num_faces = int(lines[i].strip())
            i += 1

            boxes = []
            for _ in range(num_faces):
                if i >= len(lines):
                    break

                parts = lines[i].strip().split()
                i += 1

                if len(parts) >= 4:
                    x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

                    # Attributs optionnels
                    blur = int(parts[4]) if len(parts) > 4 else 0
                    expression = int(parts[5]) if len(parts) > 5 else 0
                    illumination = int(parts[6]) if len(parts) > 6 else 0
                    invalid = int(parts[7]) if len(parts) > 7 else 0
                    occlusion = int(parts[8]) if len(parts) > 8 else 0
                    pose = int(parts[9]) if len(parts) > 9 else 0

                    # Ignorer les boxes invalides (w ou h <= 0)
                    if w > 0 and h > 0:
                        boxes.append(BBox(
                            x=x, y=y, w=w, h=h,
                            blur=blur, occlusion=occlusion, pose=pose, invalid=invalid
                        ))

            annotations[image_path] = boxes

        return annotations

    def get_image_path(self, rel_path: str) -> Path:
        """Résout le chemin complet d'une image WIDER FACE."""
        # Essayer le chemin direct
        image_path = self.images_dir / rel_path
        if image_path.exists():
            return image_path

        # WIDER FACE a souvent les images dans WIDER_val/images/
        alt_path = self.images_dir / "WIDER_val" / "images" / rel_path
        if alt_path.exists():
            return alt_path

        return None
