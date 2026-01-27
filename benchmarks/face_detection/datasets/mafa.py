# -*- coding: utf-8 -*-
"""
MAFA (Masked Faces) dataset loader.
Dataset pour la détection de visages masqués/occultés.
"""

from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from config import DATASETS_DIR
from .base import BaseDataset

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MAFADataset(BaseDataset):
    """
    Loader pour le dataset MAFA (Masked Faces).

    Annotations format (18 dimensions):
        (x, y, w, h, face_type, x1, y1, w1, h1, occ_type, occ_degree,
         gender, race, orientation, x2, y2, w2, h2)

    face_type: 1=masked, 2=unmasked, 3=invalid
    occ_type: 1=simple, 2=complex, 3=human body
    orientation: 1=left, 2=left frontal, 3=frontal, 4=right frontal, 5=right
    """

    name = "MAFA"
    description = "Dataset de visages masqués/occultés (30811 images)"

    def __init__(self, split: str = "test"):
        """
        Initialise le dataset MAFA.

        Args:
            split: 'test' ou 'train'
        """
        super().__init__()
        self.split = split
        self._mat_file = f"Label{'Test' if split == 'test' else 'Train'}All.mat"

    @property
    def annotation_file(self) -> Path:
        return DATASETS_DIR / "mafa" / "MAFA" / self._mat_file

    @property
    def images_dir(self) -> Path:
        return DATASETS_DIR / "mafa" / "MAFA" / "images"

    def is_available(self) -> bool:
        """Vérifie si le dataset est disponible."""
        if not HAS_SCIPY:
            return False
        return self.annotation_file.exists() and self.images_dir.exists()

    def load_annotations(self) -> Dict[str, List[BBox]]:
        """
        Charge les annotations MAFA depuis le fichier .mat.

        Returns:
            Dict[image_name, List[BBox]]
        """
        if not HAS_SCIPY:
            raise ImportError("scipy est requis pour charger les annotations MAFA. "
                            "Installez-le avec: pip install scipy")

        annotations = {}

        mat_data = sio.loadmat(str(self.annotation_file))

        # La structure dépend du split (test vs train)
        key = 'label_test' if self.split == 'test' else 'label_train'
        if key not in mat_data:
            # Essayer les clés alternatives
            for k in mat_data.keys():
                if not k.startswith('__'):
                    key = k
                    break

        label_data = mat_data[key]

        for item in label_data[0]:
            # item[0] contient le nom de l'image
            # item[1] contient les annotations des visages

            if len(item) < 2:
                continue

            image_name = str(item[0][0]) if item[0].size > 0 else None
            if image_name is None:
                continue

            faces_data = item[1]
            boxes = []

            if faces_data.size > 0:
                # Reshape si nécessaire
                if faces_data.ndim == 1:
                    faces_data = faces_data.reshape(1, -1)

                for face in faces_data:
                    if len(face) >= 4:
                        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])

                        # Attributs optionnels
                        face_type = int(face[4]) if len(face) > 4 else 1
                        occ_type = int(face[9]) if len(face) > 9 else 0
                        occ_degree = int(face[10]) if len(face) > 10 else 0
                        orientation = int(face[13]) if len(face) > 13 else 3

                        # Ignorer les visages invalides (face_type=3) et boxes invalides
                        if face_type != 3 and w > 0 and h > 0:
                            # Mapper face_type vers occlusion level
                            # 1=masked -> occlusion=2 (heavy)
                            # 2=unmasked -> occlusion=0 (none)
                            occlusion = 2 if face_type == 1 else 0

                            # Mapper orientation vers pose
                            # 3=frontal -> pose=0, autres -> pose=1
                            pose = 0 if orientation == 3 else 1

                            boxes.append(BBox(
                                x=x, y=y, w=w, h=h,
                                blur=0,
                                occlusion=occlusion,
                                pose=pose,
                                invalid=0
                            ))

            annotations[image_name] = boxes

        return annotations

    def get_image_path(self, rel_path: str) -> Optional[Path]:
        """Résout le chemin complet d'une image MAFA."""
        # Les images MAFA sont directement dans le dossier images/
        image_path = self.images_dir / rel_path
        if image_path.exists():
            return image_path

        # Essayer avec l'extension .jpg si pas présente
        if not rel_path.endswith('.jpg'):
            image_path = self.images_dir / f"{rel_path}.jpg"
            if image_path.exists():
                return image_path

        return None

    def get_stats(self) -> dict:
        """Retourne les statistiques du dataset MAFA."""
        stats = super().get_stats()

        # Ajouter des stats spécifiques MAFA
        annotations = self.get_annotations()
        masked_count = 0
        unmasked_count = 0

        for boxes in annotations.values():
            for box in boxes:
                if box.occlusion == 2:  # masked
                    masked_count += 1
                else:
                    unmasked_count += 1

        stats.update({
            "masked_faces": masked_count,
            "unmasked_faces": unmasked_count,
            "split": self.split,
        })

        return stats
