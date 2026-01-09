# -*- coding: utf-8 -*-
"""
Configuration de l'annotation visuelle.
Couleurs, styles, et paramètres d'affichage sur les images.
"""
from typing import Dict, Tuple


# Type alias pour les couleurs BGR (Blue, Green, Red) utilisées par OpenCV
BGRColor = Tuple[int, int, int]


class AnnotationConfig:
    """Paramètres pour l'annotation des images avec les émotions détectées."""

    # --- Couleurs des émotions (format BGR pour OpenCV) ---
    EMOTION_COLORS: Dict[str, BGRColor] = {
        "happy": (0, 255, 0),      # Vert
        "sad": (255, 0, 0),        # Bleu
        "angry": (0, 0, 255),      # Rouge
        "fear": (255, 0, 255),     # Magenta
        "surprise": (0, 255, 255), # Jaune
        "disgust": (128, 0, 128),  # Violet
        "neutral": (255, 255, 0),  # Cyan
    }
    DEFAULT_COLOR: BGRColor = (255, 255, 255)  # Blanc

    # --- Rectangle de détection ---
    BORDER_THICKNESS: int = 2

    # --- Texte des labels ---
    FONT_SCALE: float = 0.6
    TEXT_THICKNESS: int = 2
    TEXT_COLOR: BGRColor = (0, 0, 0)  # Noir
    LABEL_PADDING_X: int = 10
    LABEL_PADDING_Y: int = 5
    LABEL_MARGIN_Y: int = 5

    # --- Barre de résumé ---
    SUMMARY_BAR_HEIGHT: int = 30
    SUMMARY_BAR_COLOR: BGRColor = (50, 50, 50)  # Gris foncé
    SUMMARY_FONT_SCALE: float = 0.6
    SUMMARY_TEXT_THICKNESS: int = 2
    SUMMARY_TEXT_TEMPLATE: str = "Visages: {total} | Confort: {comfortable}/{total}"

    # --- Barre de ratio de confort ---
    COMFORT_BAR_WIDTH: int = 150
    COMFORT_BAR_HEIGHT: int = 20
    COMFORT_BAR_BG_COLOR: BGRColor = (100, 100, 100)  # Gris
    COMFORT_BAR_FILL_COLOR: BGRColor = (0, 255, 0)    # Vert

    @classmethod
    def get_emotion_color(cls, emotion: str) -> BGRColor:
        """Retourne la couleur associée à une émotion."""
        return cls.EMOTION_COLORS.get(emotion, cls.DEFAULT_COLOR)
