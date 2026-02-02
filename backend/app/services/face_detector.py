# -*- coding: utf-8 -*-
"""
Service de détection de visage avec YuNet.

YuNet est un détecteur de visage léger et robuste intégré à OpenCV.
Il est plus résistant aux occlusions partielles (cheveux, lunettes, etc.)
que MTCNN.

Référence: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import urllib.request

# URL du modèle YuNet
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


class YuNetDetector:
    """
    Détecteur de visage basé sur YuNet (OpenCV).

    Plus robuste aux occlusions partielles que MTCNN.
    """

    def __init__(self, conf_threshold: float = 0.7, nms_threshold: float = 0.3):
        """
        Initialise le détecteur YuNet.

        Args:
            conf_threshold: Seuil de confiance (0-1)
            nms_threshold: Seuil NMS pour supprimer les détections redondantes
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.detector = None
        self._model_path = None
        self._input_size = (320, 320)  # Taille d'entrée par défaut

        # Charger le modèle
        self._load_model()

    def _get_model_path(self) -> str:
        """Retourne le chemin du modèle, le télécharge si nécessaire."""
        # Chercher dans le dossier models
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        models_dir = os.path.join(base_dir, "models")
        model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")

        # Créer le dossier models si nécessaire
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Télécharger le modèle si absent
        if not os.path.exists(model_path):
            print(f"[YuNet] Téléchargement du modèle...")
            try:
                urllib.request.urlretrieve(YUNET_MODEL_URL, model_path)
                print(f"[YuNet] Modèle téléchargé: {model_path}")
            except Exception as e:
                print(f"[YuNet ERROR] Échec téléchargement: {e}")
                raise RuntimeError(f"Impossible de télécharger le modèle YuNet: {e}")

        return model_path

    def _load_model(self):
        """Charge le modèle YuNet."""
        try:
            self._model_path = self._get_model_path()

            # Créer le détecteur YuNet
            self.detector = cv2.FaceDetectorYN.create(
                self._model_path,
                "",
                self._input_size,
                self.conf_threshold,
                self.nms_threshold
            )

            print(f"[YuNet] Modèle chargé avec succès")

        except Exception as e:
            print(f"[YuNet ERROR] Échec chargement: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Détecte les visages dans une image.

        Args:
            image: Image BGR (numpy array)

        Returns:
            Liste de boxes (x, y, w, h)
        """
        if self.detector is None:
            return []

        # Redimensionner si nécessaire pour la détection
        h, w = image.shape[:2]

        # Mettre à jour la taille d'entrée du détecteur
        self.detector.setInputSize((w, h))

        # Détecter les visages
        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        boxes = []
        for face in faces:
            # YuNet retourne: x, y, w, h, landmarks..., confidence
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

            # Vérifier les limites
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            if fw > 10 and fh > 10:  # Ignorer les détections trop petites
                boxes.append((x, y, fw, fh))

        return boxes

    def detect_with_confidence(self, image: np.ndarray) -> List[dict]:
        """
        Détecte les visages avec leurs scores de confiance.

        Args:
            image: Image BGR

        Returns:
            Liste de dicts {'box': (x,y,w,h), 'confidence': float, 'landmarks': [...]}
        """
        if self.detector is None:
            return []

        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))

        _, faces = self.detector.detect(image)

        if faces is None:
            return []

        results = []
        for face in faces:
            x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            confidence = float(face[14]) if len(face) > 14 else 1.0

            # Extraire les landmarks (5 points: yeux, nez, coins de bouche)
            landmarks = []
            if len(face) >= 14:
                for i in range(5):
                    lx, ly = face[4 + i * 2], face[5 + i * 2]
                    landmarks.append((int(lx), int(ly)))

            # Vérifier les limites
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            if fw > 10 and fh > 10:
                results.append({
                    'box': (x, y, fw, fh),
                    'confidence': confidence,
                    'landmarks': landmarks
                })

        # Trier par confiance décroissante
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results


# Singleton
yunet_detector = YuNetDetector()
