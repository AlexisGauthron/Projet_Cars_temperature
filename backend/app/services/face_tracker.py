# -*- coding: utf-8 -*-
"""
Service de tracking de visage pour maintenir le focus sur la même personne.

En mode "single", on veut toujours suivre le même visage même si d'autres
personnes apparaissent dans le champ de la caméra.

Algorithme:
1. Premier visage détecté → devient le "target"
2. Frames suivantes → trouver le visage le plus proche du target (IoU + distance)
3. Si le target disparaît trop longtemps → reset et prendre le plus grand visage
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import time


@dataclass
class TrackedFace:
    """Représente un visage suivi."""
    box: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[float, float]  # Centre du visage
    area: int  # Surface du visage
    last_seen: float  # Timestamp de dernière détection


class FaceTrackerConfig:
    """Configuration du tracker."""

    # Seuil IoU minimum pour considérer que c'est le même visage
    IOU_THRESHOLD = 0.3

    # Seuil de distance maximum (en pixels) pour le matching
    MAX_CENTER_DISTANCE = 150

    # Temps (en secondes) avant de considérer le target comme perdu
    TARGET_LOST_TIMEOUT = 2.0

    # Nombre de frames sans détection avant reset
    MAX_FRAMES_WITHOUT_DETECTION = 60


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calcule l'Intersection over Union (IoU) entre deux boxes.

    Args:
        box1, box2: (x, y, w, h)

    Returns:
        IoU entre 0 et 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convertir en (x1, y1, x2, y2)
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calcule le centre d'une box."""
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def compute_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Calcule la distance euclidienne entre deux points."""
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


class FaceTracker:
    """
    Tracker de visage pour le mode single.

    Maintient le focus sur un visage "target" et le suit entre les frames.
    """

    def __init__(self):
        self.target: Optional[TrackedFace] = None
        self.frames_without_target = 0
        self._initialized = False

    def update(self, faces: List[Dict]) -> Optional[Dict]:
        """
        Met à jour le tracker avec les nouveaux visages détectés.

        Args:
            faces: Liste des visages détectés [{'box': (x,y,w,h), 'emotions': {...}}, ...]

        Returns:
            Le visage cible à suivre, ou None si aucun
        """
        if not faces:
            self.frames_without_target += 1

            # Reset si trop longtemps sans détection
            if self.frames_without_target > FaceTrackerConfig.MAX_FRAMES_WITHOUT_DETECTION:
                self.reset()

            return None

        current_time = time.time()

        # Pas encore de target → initialiser avec le plus grand visage
        if self.target is None or not self._initialized:
            return self._initialize_target(faces, current_time)

        # Vérifier si le target est trop vieux
        if current_time - self.target.last_seen > FaceTrackerConfig.TARGET_LOST_TIMEOUT:
            print("[TRACKER] Target perdu (timeout), réinitialisation...")
            return self._initialize_target(faces, current_time)

        # Trouver le meilleur match pour le target actuel
        best_match = self._find_best_match(faces)

        if best_match is not None:
            # Mettre à jour le target
            box = tuple(best_match['box'])
            self.target = TrackedFace(
                box=box,
                center=compute_center(box),
                area=box[2] * box[3],
                last_seen=current_time
            )
            self.frames_without_target = 0
            return best_match
        else:
            # Pas de match trouvé
            self.frames_without_target += 1

            # Si trop de frames sans match, réinitialiser
            if self.frames_without_target > FaceTrackerConfig.MAX_FRAMES_WITHOUT_DETECTION // 2:
                print("[TRACKER] Trop de frames sans match, réinitialisation...")
                return self._initialize_target(faces, current_time)

            # Garder l'ancien target (interpolation)
            return None

    def _initialize_target(self, faces: List[Dict], current_time: float) -> Dict:
        """
        Initialise le target avec le plus grand visage (le plus proche de la caméra).
        """
        # Trouver le visage avec la plus grande surface
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        box = tuple(largest_face['box'])

        self.target = TrackedFace(
            box=box,
            center=compute_center(box),
            area=box[2] * box[3],
            last_seen=current_time
        )
        self._initialized = True
        self.frames_without_target = 0

        print(f"[TRACKER] Target initialisé: box={box}, area={self.target.area}")

        return largest_face

    def _find_best_match(self, faces: List[Dict]) -> Optional[Dict]:
        """
        Trouve le visage qui correspond le mieux au target actuel.

        Utilise une combinaison de:
        - IoU (Intersection over Union)
        - Distance entre les centres
        - Similarité de taille
        """
        if self.target is None:
            return None

        best_score = -1
        best_face = None

        for face in faces:
            box = tuple(face['box'])
            center = compute_center(box)
            area = box[2] * box[3]

            # Calculer l'IoU
            iou = compute_iou(self.target.box, box)

            # Calculer la distance normalisée
            distance = compute_distance(self.target.center, center)

            # Ignorer si trop loin
            if distance > FaceTrackerConfig.MAX_CENTER_DISTANCE and iou < FaceTrackerConfig.IOU_THRESHOLD:
                continue

            # Calculer le ratio de taille (pénaliser les changements brusques)
            size_ratio = min(area, self.target.area) / max(area, self.target.area)

            # Score combiné (pondéré)
            # IoU est le plus important, puis distance, puis taille
            distance_score = max(0, 1 - distance / FaceTrackerConfig.MAX_CENTER_DISTANCE)
            score = 0.5 * iou + 0.3 * distance_score + 0.2 * size_ratio

            if score > best_score:
                best_score = score
                best_face = face

        # Vérifier que le meilleur score est acceptable
        if best_score < 0.2:
            return None

        return best_face

    def reset(self):
        """Réinitialise le tracker."""
        self.target = None
        self._initialized = False
        self.frames_without_target = 0
        print("[TRACKER] Reset complet")

    def get_target_info(self) -> Optional[Dict]:
        """Retourne les infos du target actuel."""
        if self.target is None:
            return None

        return {
            "box": self.target.box,
            "center": self.target.center,
            "area": self.target.area,
            "age": time.time() - self.target.last_seen
        }


# Singleton
face_tracker = FaceTracker()
