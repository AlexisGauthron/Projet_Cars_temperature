# -*- coding: utf-8 -*-
"""
Configuration de la détection d'émotions.
Seuils, catégories, et paramètres de lissage.
"""
from typing import Set


class EmotionConfig:
    """Paramètres pour la détection et le traitement des émotions."""

    # --- Catégories d'émotions ---
    COMFORT_EMOTIONS: Set[str] = {"happy", "surprise", "neutral"}
    DISCOMFORT_EMOTIONS: Set[str] = {"sad", "angry", "fear", "disgust"}
    ALL_EMOTIONS: Set[str] = COMFORT_EMOTIONS | DISCOMFORT_EMOTIONS

    # --- Historique des émotions ---
    # Taille maximale de l'historique (nombre de frames conservées)
    HISTORY_MAX_SIZE: int = 15

    # Nombre minimum d'émotions avant de pouvoir analyser
    HISTORY_MIN_SIZE: int = 5

    # --- Seuils de détection ---
    # Pourcentage de visages inconfortables pour déclencher l'état global "inconfortable"
    # Ex: 0.5 = plus de 50% des visages doivent être inconfortables
    COMFORT_MAJORITY_THRESHOLD: float = 0.5

    # --- Lissage temporel ---
    # Taille du buffer pour le lissage (nombre de frames à moyenner)
    SMOOTHING_BUFFER_SIZE: int = 5

    # Seuil de confiance minimum pour considérer une émotion valide (0.0 - 1.0)
    MIN_CONFIDENCE_THRESHOLD: float = 0.4

    # --- Labels en français ---
    EMOTION_LABELS_FR: dict = {
        "happy": "Heureux",
        "sad": "Triste",
        "angry": "En colère",
        "fear": "Peur",
        "surprise": "Surpris",
        "disgust": "Dégoût",
        "neutral": "Neutre",
    }

    @classmethod
    def get_french_label(cls, emotion: str) -> str:
        """Retourne le label français pour une émotion."""
        return cls.EMOTION_LABELS_FR.get(emotion, emotion.capitalize())

    @classmethod
    def is_comfortable(cls, emotion: str) -> bool:
        """Vérifie si une émotion est dans la catégorie confortable."""
        return emotion in cls.COMFORT_EMOTIONS
