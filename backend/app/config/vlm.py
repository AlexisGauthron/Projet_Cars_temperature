# -*- coding: utf-8 -*-
"""
Configuration du système VLM (Vision Language Model).
Questions, seuils de déclenchement, et réponses.
"""
from typing import Dict, List


class VLMConfig:
    """Paramètres pour le système de questions VLM."""

    # --- Seuils de déclenchement ---
    # Fenêtre d'analyse pour les questions (dernières N émotions)
    ANALYSIS_WINDOW: int = 8

    # Nombre minimum d'émotions inconfortables pour déclencher une question
    # Ex: 5 sur 8 = 62.5% d'inconfort requis
    DISCOMFORT_THRESHOLD: int = 5

    # --- Réponses acceptées ---
    HOT_RESPONSES: List[str] = ["trop chaud", "chaud", "hot", "baisser"]
    COLD_RESPONSES: List[str] = ["trop froid", "froid", "cold", "augmenter"]
    OK_RESPONSES: List[str] = ["ça va", "ca va", "c'est bon", "ok", "bon"]

    # --- Questions contextuelles ---
    QUESTIONS: Dict[str, Dict] = {
        "sad": {
            "question": "Vous semblez inconfortable. La température vous convient-elle ?",
            "options": ["Trop chaud", "Trop froid", "Ça va"],
        },
        "angry": {
            "question": "Souhaitez-vous ajuster la climatisation ?",
            "options": ["Baisser", "Augmenter", "C'est bon"],
        },
        "fear": {
            "question": "Tout va bien ? La température est-elle confortable ?",
            "options": ["Trop chaud", "Trop froid", "Ça va"],
        },
        "disgust": {
            "question": "L'atmosphère vous convient-elle ?",
            "options": ["Trop chaud", "Trop froid", "Ça va"],
        },
        "default": {
            "question": "La température vous convient-elle ?",
            "options": ["Trop chaud", "Trop froid", "Ça va"],
        },
    }

    # --- Messages de confirmation ---
    MESSAGES: Dict[str, str] = {
        "temp_decreased": "Température baissée à {temp}°C",
        "temp_increased": "Température augmentée à {temp}°C",
        "temp_maintained": "Température maintenue",
    }

    @classmethod
    def get_question(cls, emotion: str) -> Dict:
        """Retourne la question appropriée pour une émotion."""
        return cls.QUESTIONS.get(emotion, cls.QUESTIONS["default"])

    @classmethod
    def parse_response(cls, response: str) -> str:
        """
        Parse la réponse utilisateur et retourne la direction.

        Returns:
            "chaud", "froid", ou "ok"
        """
        response_lower = response.lower().strip()

        if response_lower in cls.HOT_RESPONSES:
            return "chaud"
        elif response_lower in cls.COLD_RESPONSES:
            return "froid"
        else:
            return "ok"
