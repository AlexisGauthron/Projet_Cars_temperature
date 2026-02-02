# -*- coding: utf-8 -*-
"""
Module core pour le benchmark de classification d'émotions.
"""

from .structures import EmotionLabel, EmotionPrediction, BenchmarkMetrics
from .metrics import compute_metrics, compute_confusion_matrix

__all__ = [
    "EmotionLabel",
    "EmotionPrediction",
    "BenchmarkMetrics",
    "compute_metrics",
    "compute_confusion_matrix",
]
