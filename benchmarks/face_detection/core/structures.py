# -*- coding: utf-8 -*-
"""
Structures de données pour le benchmark de détection de visage.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class BBox:
    """Bounding box avec métadonnées."""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    # Attributs WIDER FACE
    blur: int = 0       # 0=clear, 1=normal, 2=heavy
    occlusion: int = 0  # 0=none, 1=partial, 2=heavy
    pose: int = 0       # 0=typical, 1=atypical
    invalid: int = 0    # 0=valid, 1=invalid

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class DetectionResult:
    """Résultat de détection pour une image."""
    image_path: str
    ground_truth: List[BBox]
    predictions: List[BBox]
    detection_time_ms: float
    # Métriques calculées
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class BenchmarkMetrics:
    """Métriques globales du benchmark."""
    detector_name: str
    total_images: int = 0
    total_gt_faces: int = 0
    total_detected: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_time_ms: float = 0.0
    # Par difficulté
    easy: Dict = field(default_factory=lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    medium: Dict = field(default_factory=lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    hard: Dict = field(default_factory=lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    # Pour courbe PR
    all_scores: List[Tuple[float, bool]] = field(default_factory=list)
    # Statistiques de timing (mode strict)
    time_min_ms: float = 0.0
    time_max_ms: float = 0.0
    time_std_ms: float = 0.0
    all_times_ms: List[float] = field(default_factory=list)
    # Métadonnées mode strict
    strict_mode: bool = False
    warmup_images: int = 0
    num_passes: int = 1

    @property
    def precision(self) -> float:
        if self.total_tp + self.total_fp == 0:
            return 0.0
        return self.total_tp / (self.total_tp + self.total_fp)

    @property
    def recall(self) -> float:
        if self.total_tp + self.total_fn == 0:
            return 0.0
        return self.total_tp / (self.total_tp + self.total_fn)

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def avg_time_ms(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.total_time_ms / self.total_images
