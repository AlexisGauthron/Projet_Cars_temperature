# -*- coding: utf-8 -*-
"""
Configuration partagée pour les benchmarks.

Ce module contient les configurations communes à tous les benchmarks:
- Mode strict (warmup, passes, etc.)
- Profils de benchmark prédéfinis
"""

from .strict import (
    PROFILES,
    DEFAULT_PROFILE,
    WARMUP_IMAGES,
    NUM_PASSES,
    CLEAR_GPU_MEMORY,
    TIMER_FUNCTION,
    GPU_SYNC_BEFORE_TIMING,
    GC_BETWEEN_PASSES,
    GC_BETWEEN_MODELS,
    get_profile,
    list_profiles,
    print_profiles,
)

__all__ = [
    "PROFILES",
    "DEFAULT_PROFILE",
    "WARMUP_IMAGES",
    "NUM_PASSES",
    "CLEAR_GPU_MEMORY",
    "TIMER_FUNCTION",
    "GPU_SYNC_BEFORE_TIMING",
    "GC_BETWEEN_PASSES",
    "GC_BETWEEN_MODELS",
    "get_profile",
    "list_profiles",
    "print_profiles",
]
