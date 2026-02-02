# -*- coding: utf-8 -*-
"""
Structures de données pour le benchmark de classification d'émotions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class EmotionLabel(Enum):
    """Labels d'émotions standard (FER2013 / AffectNet)."""
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    SAD = 4
    SURPRISE = 5
    NEUTRAL = 6
    CONTEMPT = 7  # AffectNet uniquement

    @classmethod
    def from_string(cls, name: str) -> "EmotionLabel":
        """Convertit un string en EmotionLabel."""
        name_upper = name.upper().strip()
        mapping = {
            "ANGRY": cls.ANGRY,
            "ANGER": cls.ANGRY,
            "DISGUST": cls.DISGUST,
            "FEAR": cls.FEAR,
            "HAPPY": cls.HAPPY,
            "HAPPINESS": cls.HAPPY,
            "SAD": cls.SAD,
            "SADNESS": cls.SAD,
            "SURPRISE": cls.SURPRISE,
            "SURPRISED": cls.SURPRISE,
            "NEUTRAL": cls.NEUTRAL,
            "CONTEMPT": cls.CONTEMPT,
        }
        return mapping.get(name_upper, cls.NEUTRAL)

    @classmethod
    def fer2013_labels(cls) -> List["EmotionLabel"]:
        """Retourne les 7 labels FER2013."""
        return [cls.ANGRY, cls.DISGUST, cls.FEAR, cls.HAPPY, cls.SAD, cls.SURPRISE, cls.NEUTRAL]

    @classmethod
    def affectnet_labels(cls) -> List["EmotionLabel"]:
        """Retourne les 8 labels AffectNet."""
        return [cls.NEUTRAL, cls.HAPPY, cls.SAD, cls.SURPRISE, cls.FEAR, cls.DISGUST, cls.ANGRY, cls.CONTEMPT]


@dataclass
class EmotionPrediction:
    """Représente une prédiction d'émotion."""
    label: EmotionLabel
    confidence: float
    probabilities: Optional[Dict[EmotionLabel, float]] = None

    def __repr__(self) -> str:
        return f"EmotionPrediction({self.label.name}, conf={self.confidence:.2f})"


@dataclass
class EmotionSample:
    """Représente un échantillon du dataset."""
    image_path: str
    ground_truth: EmotionLabel
    face_bbox: Optional[tuple] = None  # (x, y, w, h) si disponible


@dataclass
class BenchmarkMetrics:
    """Métriques de benchmark pour un classifieur d'émotions."""
    classifier_name: str

    # Compteurs globaux
    total_samples: int = 0
    correct_predictions: int = 0
    total_time_ms: float = 0.0

    # Par classe
    per_class_correct: Dict[EmotionLabel, int] = field(default_factory=dict)
    per_class_total: Dict[EmotionLabel, int] = field(default_factory=dict)
    per_class_predicted: Dict[EmotionLabel, int] = field(default_factory=dict)

    # Matrice de confusion: confusion[gt][pred] = count
    confusion_matrix: Dict[EmotionLabel, Dict[EmotionLabel, int]] = field(default_factory=dict)

    # Scores de confiance
    all_confidences: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Initialise les dictionnaires pour chaque classe."""
        for label in EmotionLabel.fer2013_labels():
            if label not in self.per_class_correct:
                self.per_class_correct[label] = 0
            if label not in self.per_class_total:
                self.per_class_total[label] = 0
            if label not in self.per_class_predicted:
                self.per_class_predicted[label] = 0
            if label not in self.confusion_matrix:
                self.confusion_matrix[label] = {l: 0 for l in EmotionLabel.fer2013_labels()}

    @property
    def accuracy(self) -> float:
        """Accuracy globale."""
        if self.total_samples == 0:
            return 0.0
        return self.correct_predictions / self.total_samples

    @property
    def avg_time_ms(self) -> float:
        """Temps moyen par image en ms."""
        if self.total_samples == 0:
            return 0.0
        return self.total_time_ms / self.total_samples

    @property
    def fps(self) -> float:
        """Images par seconde."""
        if self.avg_time_ms == 0:
            return 0.0
        return 1000.0 / self.avg_time_ms

    def per_class_accuracy(self, label: EmotionLabel) -> float:
        """Accuracy pour une classe spécifique."""
        total = self.per_class_total.get(label, 0)
        if total == 0:
            return 0.0
        return self.per_class_correct.get(label, 0) / total

    def per_class_precision(self, label: EmotionLabel) -> float:
        """Précision pour une classe (TP / (TP + FP))."""
        predicted = self.per_class_predicted.get(label, 0)
        if predicted == 0:
            return 0.0
        return self.per_class_correct.get(label, 0) / predicted

    def per_class_recall(self, label: EmotionLabel) -> float:
        """Recall pour une classe (TP / (TP + FN))."""
        return self.per_class_accuracy(label)  # Équivalent au recall

    def per_class_f1(self, label: EmotionLabel) -> float:
        """F1-score pour une classe."""
        precision = self.per_class_precision(label)
        recall = self.per_class_recall(label)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def macro_f1(self) -> float:
        """Macro F1-score (moyenne des F1 par classe)."""
        f1_scores = [self.per_class_f1(label) for label in EmotionLabel.fer2013_labels()]
        return sum(f1_scores) / len(f1_scores)

    @property
    def weighted_f1(self) -> float:
        """Weighted F1-score (pondéré par le support)."""
        total = sum(self.per_class_total.values())
        if total == 0:
            return 0.0
        weighted_sum = sum(
            self.per_class_f1(label) * self.per_class_total.get(label, 0)
            for label in EmotionLabel.fer2013_labels()
        )
        return weighted_sum / total

    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour export."""
        return {
            "classifier": self.classifier_name,
            "total_samples": self.total_samples,
            "accuracy": round(self.accuracy * 100, 2),
            "macro_f1": round(self.macro_f1 * 100, 2),
            "weighted_f1": round(self.weighted_f1 * 100, 2),
            "avg_time_ms": round(self.avg_time_ms, 2),
            "fps": round(self.fps, 1),
            "per_class": {
                label.name: {
                    "accuracy": round(self.per_class_accuracy(label) * 100, 2),
                    "precision": round(self.per_class_precision(label) * 100, 2),
                    "recall": round(self.per_class_recall(label) * 100, 2),
                    "f1": round(self.per_class_f1(label) * 100, 2),
                    "support": self.per_class_total.get(label, 0),
                }
                for label in EmotionLabel.fer2013_labels()
            }
        }

    def __repr__(self) -> str:
        return (
            f"BenchmarkMetrics({self.classifier_name}: "
            f"acc={self.accuracy:.1%}, f1={self.macro_f1:.1%}, "
            f"{self.avg_time_ms:.1f}ms/img)"
        )
