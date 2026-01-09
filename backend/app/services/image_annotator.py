# -*- coding: utf-8 -*-
"""
Service d'annotation des images avec les émotions détectées.
Dessine les rectangles, labels et barres de résumé sur les images.
"""
import cv2
import numpy as np
from typing import List, TYPE_CHECKING

from app.config import AnnotationConfig, EmotionConfig

if TYPE_CHECKING:
    from app.services.emotion_service import FaceEmotion


class ImageAnnotator:
    """Annotateur d'images avec les émotions détectées."""

    def annotate_all_faces(
        self,
        image: np.ndarray,
        faces: List["FaceEmotion"]
    ) -> np.ndarray:
        """
        Annote l'image avec tous les visages détectés et leurs émotions.

        Args:
            image: Image source (numpy array BGR)
            faces: Liste des visages détectés avec leurs émotions

        Returns:
            Image annotée
        """
        annotated = image.copy()

        if not faces:
            return annotated

        for i, face in enumerate(faces):
            self._draw_face_annotation(annotated, face, i)

        # Dessiner la barre de résumé en haut
        self._draw_summary(annotated, faces)

        return annotated

    def _draw_face_annotation(
        self,
        image: np.ndarray,
        face: "FaceEmotion",
        index: int
    ) -> None:
        """Dessine l'annotation pour un visage."""
        x, y, w, h = face.box
        # Utiliser l'émotion lissée si disponible
        emotion = getattr(face, 'smoothed_dominant', face.dominant)
        emotions_dict = face.emotions
        color = AnnotationConfig.get_emotion_color(emotion)

        # Rectangle autour du visage
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            AnnotationConfig.BORDER_THICKNESS
        )

        # Préparer le label
        label = EmotionConfig.get_french_label(emotion)

        # Ajouter la confiance
        if emotions_dict and face.dominant in emotions_dict:
            confidence = emotions_dict[face.dominant] * 100
            label = f"#{index + 1} {label} ({confidence:.0f}%)"
        else:
            label = f"#{index + 1} {label}"

        # Calculer la position du label
        label_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            AnnotationConfig.FONT_SCALE,
            AnnotationConfig.TEXT_THICKNESS
        )
        label_w, label_h = label_size

        # S'assurer que le label reste dans l'image
        label_y = max(y - AnnotationConfig.LABEL_MARGIN_Y, label_h + 10)

        # Fond du label
        cv2.rectangle(
            image,
            (x, label_y - label_h - AnnotationConfig.LABEL_PADDING_Y),
            (x + label_w + AnnotationConfig.LABEL_PADDING_X, label_y + AnnotationConfig.LABEL_PADDING_Y),
            color,
            -1
        )

        # Texte du label
        cv2.putText(
            image,
            label,
            (x + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            AnnotationConfig.FONT_SCALE,
            AnnotationConfig.TEXT_COLOR,
            AnnotationConfig.TEXT_THICKNESS
        )

    def _draw_summary(self, image: np.ndarray, faces: List["FaceEmotion"]) -> None:
        """Dessine la barre de résumé en haut de l'image."""
        total = len(faces)

        # Compter les visages confortables (utiliser l'émotion lissée)
        comfortable = sum(
            1 for f in faces
            if EmotionConfig.is_comfortable(getattr(f, 'smoothed_dominant', f.dominant))
        )
        uncomfortable = total - comfortable

        # Texte du résumé
        summary = f"Visages: {total} | Confort: {comfortable} | Inconfort: {uncomfortable}"

        # Barre de fond
        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1], AnnotationConfig.SUMMARY_BAR_HEIGHT),
            AnnotationConfig.SUMMARY_BAR_COLOR,
            -1
        )

        # Texte
        cv2.putText(
            image,
            summary,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            AnnotationConfig.SUMMARY_FONT_SCALE,
            (255, 255, 255),
            AnnotationConfig.SUMMARY_TEXT_THICKNESS
        )

        # Barre de ratio de confort
        if total > 0:
            self._draw_comfort_bar(image, comfortable / total)

    def _draw_comfort_bar(self, image: np.ndarray, ratio: float) -> None:
        """Dessine la barre de ratio de confort."""
        bar_x = image.shape[1] - AnnotationConfig.COMFORT_BAR_WIDTH - 10
        bar_y1 = 8
        bar_y2 = 24

        # Fond de la barre
        cv2.rectangle(
            image,
            (bar_x, bar_y1),
            (bar_x + AnnotationConfig.COMFORT_BAR_WIDTH, bar_y2),
            AnnotationConfig.COMFORT_BAR_BG_COLOR,
            -1
        )

        # Barre de confort (verte)
        comfort_width = int(AnnotationConfig.COMFORT_BAR_WIDTH * ratio)
        if comfort_width > 0:
            cv2.rectangle(
                image,
                (bar_x, bar_y1),
                (bar_x + comfort_width, bar_y2),
                AnnotationConfig.COMFORT_BAR_FILL_COLOR,
                -1
            )


# Instance singleton
image_annotator = ImageAnnotator()
