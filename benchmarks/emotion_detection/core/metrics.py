# -*- coding: utf-8 -*-
"""
Métriques pour le benchmark de classification d'émotions.
"""

from typing import Dict, List, Tuple
import numpy as np

from .structures import EmotionLabel, EmotionPrediction, BenchmarkMetrics


def compute_metrics(
    predictions: List[EmotionPrediction],
    ground_truths: List[EmotionLabel],
    classifier_name: str = "Unknown"
) -> BenchmarkMetrics:
    """
    Calcule les métriques à partir des prédictions et ground truths.

    Args:
        predictions: Liste des prédictions
        ground_truths: Liste des labels ground truth
        classifier_name: Nom du classifieur

    Returns:
        BenchmarkMetrics avec toutes les métriques calculées
    """
    metrics = BenchmarkMetrics(classifier_name=classifier_name)
    metrics.total_samples = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        # Mise à jour compteurs globaux
        if pred.label == gt:
            metrics.correct_predictions += 1
            metrics.per_class_correct[gt] = metrics.per_class_correct.get(gt, 0) + 1

        # Compteurs par classe
        metrics.per_class_total[gt] = metrics.per_class_total.get(gt, 0) + 1
        metrics.per_class_predicted[pred.label] = metrics.per_class_predicted.get(pred.label, 0) + 1

        # Matrice de confusion
        if gt not in metrics.confusion_matrix:
            metrics.confusion_matrix[gt] = {}
        metrics.confusion_matrix[gt][pred.label] = metrics.confusion_matrix[gt].get(pred.label, 0) + 1

        # Confidences
        metrics.all_confidences.append(pred.confidence)

    return metrics


def compute_confusion_matrix(
    metrics: BenchmarkMetrics,
    labels: List[EmotionLabel] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Génère la matrice de confusion sous forme numpy.

    Args:
        metrics: BenchmarkMetrics avec la confusion matrix
        labels: Liste des labels à inclure (défaut: FER2013)

    Returns:
        Tuple (matrice numpy, liste des noms de labels)
    """
    if labels is None:
        labels = EmotionLabel.fer2013_labels()

    n_classes = len(labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i, gt_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if gt_label in metrics.confusion_matrix:
                matrix[i, j] = metrics.confusion_matrix[gt_label].get(pred_label, 0)

    label_names = [label.name for label in labels]
    return matrix, label_names


def print_confusion_matrix(metrics: BenchmarkMetrics, labels: List[EmotionLabel] = None):
    """Affiche la matrice de confusion formatée."""
    matrix, label_names = compute_confusion_matrix(metrics, labels)

    # Header
    header = "GT\\Pred".ljust(10) + " ".join(name[:7].center(7) for name in label_names)
    print(header)
    print("-" * len(header))

    # Lignes
    for i, name in enumerate(label_names):
        row = name[:10].ljust(10)
        row += " ".join(str(matrix[i, j]).center(7) for j in range(len(label_names)))
        print(row)


def print_classification_report(metrics: BenchmarkMetrics):
    """Affiche un rapport de classification style sklearn."""
    print(f"\n{'='*60}")
    print(f"Classification Report: {metrics.classifier_name}")
    print(f"{'='*60}")

    print(f"\n{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 54)

    labels = EmotionLabel.fer2013_labels()
    for label in labels:
        precision = metrics.per_class_precision(label) * 100
        recall = metrics.per_class_recall(label) * 100
        f1 = metrics.per_class_f1(label) * 100
        support = metrics.per_class_total.get(label, 0)

        print(f"{label.name:<12} {precision:>9.1f}% {recall:>9.1f}% {f1:>9.1f}% {support:>10}")

    print("-" * 54)
    print(f"{'Accuracy':<12} {'':<10} {'':<10} {metrics.accuracy*100:>9.1f}% {metrics.total_samples:>10}")
    print(f"{'Macro Avg':<12} {'':<10} {'':<10} {metrics.macro_f1*100:>9.1f}% {metrics.total_samples:>10}")
    print(f"{'Weighted Avg':<12} {'':<10} {'':<10} {metrics.weighted_f1*100:>9.1f}% {metrics.total_samples:>10}")

    print(f"\nPerformance: {metrics.avg_time_ms:.2f} ms/image ({metrics.fps:.1f} FPS)")
