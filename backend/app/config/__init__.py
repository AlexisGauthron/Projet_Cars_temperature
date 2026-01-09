# -*- coding: utf-8 -*-
"""
Configuration centralisée de l'application ProjectCare.

Usage:
    from app.config import Settings, EmotionConfig, TemperatureConfig, VLMConfig, AnnotationConfig

Tous les paramètres de l'application sont centralisés ici pour faciliter
la maintenance et les ajustements.
"""

from app.config.settings import Settings
from app.config.emotion import EmotionConfig
from app.config.temperature import TemperatureConfig
from app.config.vlm import VLMConfig
from app.config.annotation import AnnotationConfig

__all__ = [
    "Settings",
    "EmotionConfig",
    "TemperatureConfig",
    "VLMConfig",
    "AnnotationConfig",
]
