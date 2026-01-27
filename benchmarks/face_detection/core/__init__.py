# -*- coding: utf-8 -*-
"""
Core module - Structures de données et métriques.
"""

from .structures import BBox, DetectionResult, BenchmarkMetrics
from .metrics import compute_iou, match_detections, get_difficulty_level, compute_ap
from .results import (
    print_results,
    export_results,
    load_results_from_json,
    get_results_filepath,
    load_existing_results,
    merge_and_save_results
)
from .logger import get_logger, setup_logging

__all__ = [
    # Structures
    "BBox",
    "DetectionResult",
    "BenchmarkMetrics",
    # Métriques
    "compute_iou",
    "match_detections",
    "get_difficulty_level",
    "compute_ap",
    # Résultats
    "print_results",
    "export_results",
    "load_results_from_json",
    "get_results_filepath",
    "load_existing_results",
    "merge_and_save_results",
    # Logging
    "get_logger",
    "setup_logging",
]
