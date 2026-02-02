# -*- coding: utf-8 -*-
"""
Moteur d'exécution du benchmark de classification d'émotions.
"""

import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import EmotionLabel, EmotionPrediction, BenchmarkMetrics
from datasets.base import BaseEmotionDataset
from classifiers.base import BaseEmotionClassifier

# Tqdm optionnel
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def run_benchmark(
    classifier: BaseEmotionClassifier,
    dataset: BaseEmotionDataset,
    limit: Optional[int] = None,
    warmup: int = 5,
    quiet: bool = False,
    progress_callback: Optional[callable] = None
) -> BenchmarkMetrics:
    """
    Exécute le benchmark pour un classifieur sur un dataset.

    Args:
        classifier: Instance du classifieur à tester
        dataset: Instance du dataset à utiliser
        limit: Nombre maximum d'images à traiter
        warmup: Nombre d'itérations de warmup
        quiet: Si True, supprime les outputs
        progress_callback: Callback appelé avec (current, total)

    Returns:
        BenchmarkMetrics avec tous les résultats
    """
    metrics = BenchmarkMetrics(classifier_name=classifier.name)

    # Vérifier disponibilité
    if not classifier.is_available():
        if not quiet:
            print(f"  {classifier.name}: Not available")
        return metrics

    if not dataset.is_available():
        if not quiet:
            print(f"  Dataset {dataset.name}: Not available")
        return metrics

    # Charger les échantillons
    samples = dataset.get_samples()
    if limit:
        samples = samples[:limit]

    total_samples = len(samples)

    if not quiet:
        print(f"\n  {classifier.name} on {dataset.name} ({total_samples} samples)")

    # Warmup
    if warmup > 0 and not quiet:
        print(f"  Warmup ({warmup} iterations)...", end=" ", flush=True)

    classifier.warmup(warmup)

    if warmup > 0 and not quiet:
        print("Done")

    # Benchmark
    if HAS_TQDM and not quiet:
        iterator = tqdm(samples, desc=f"  {classifier.name}", ncols=80)
    else:
        iterator = samples

    for i, sample in enumerate(iterator):
        # Callback de progression
        if progress_callback:
            progress_callback(i + 1, total_samples)

        # Charger l'image
        image = dataset.get_image(sample)
        if image is None:
            continue

        # Mesurer le temps de prédiction
        start_time = time.perf_counter()
        prediction = classifier.predict(image)
        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        # Mettre à jour les métriques
        metrics.total_samples += 1
        metrics.total_time_ms += inference_time

        gt_label = sample.ground_truth
        pred_label = prediction.label

        # Accuracy
        if pred_label == gt_label:
            metrics.correct_predictions += 1
            metrics.per_class_correct[gt_label] = metrics.per_class_correct.get(gt_label, 0) + 1

        # Compteurs par classe
        metrics.per_class_total[gt_label] = metrics.per_class_total.get(gt_label, 0) + 1
        metrics.per_class_predicted[pred_label] = metrics.per_class_predicted.get(pred_label, 0) + 1

        # Matrice de confusion
        if gt_label not in metrics.confusion_matrix:
            metrics.confusion_matrix[gt_label] = {}
        metrics.confusion_matrix[gt_label][pred_label] = \
            metrics.confusion_matrix[gt_label].get(pred_label, 0) + 1

        # Confiances
        metrics.all_confidences.append(prediction.confidence)

        # Affichage progression (sans tqdm)
        if not HAS_TQDM and not quiet and (i + 1) % 100 == 0:
            print(f"\r  {classifier.name}: {i+1}/{total_samples} "
                  f"({metrics.accuracy*100:.1f}% acc)", end="", flush=True)

    if not HAS_TQDM and not quiet:
        print()

    # Résumé
    if not quiet:
        print(f"  Result: {metrics.accuracy*100:.1f}% accuracy, "
              f"{metrics.avg_time_ms:.2f}ms/img ({metrics.fps:.1f} FPS)")

    return metrics


def _worker_init():
    """Initialisation du worker pour multiprocessing."""
    import warnings
    warnings.filterwarnings('ignore')


def _benchmark_worker(args) -> Optional[BenchmarkMetrics]:
    """Worker pour benchmark parallèle."""
    classifier_name, dataset_name, limit, warmup = args

    try:
        # Imports locaux dans le worker
        from classifiers import get_classifier
        from datasets import get_dataset

        classifier = get_classifier(classifier_name)
        dataset = get_dataset(dataset_name)

        return run_benchmark(
            classifier, dataset,
            limit=limit, warmup=warmup, quiet=True
        )

    except Exception as e:
        print(f"  Error with {classifier_name}: {e}")
        return None


def run_benchmark_parallel(
    classifiers: List[BaseEmotionClassifier],
    dataset: BaseEmotionDataset,
    limit: Optional[int] = None,
    warmup: int = 5,
    num_workers: Optional[int] = None
) -> Dict[str, BenchmarkMetrics]:
    """
    Exécute le benchmark en parallèle pour plusieurs classifieurs.

    ⚠️ Note: Les temps mesurés en parallèle ne sont PAS objectifs
    pour comparer les performances. Utilisez le mode séquentiel
    pour des mesures de temps fiables.

    Args:
        classifiers: Liste des classifieurs à tester
        dataset: Dataset à utiliser
        limit: Nombre maximum d'images
        warmup: Itérations de warmup
        num_workers: Nombre de workers

    Returns:
        Dict[classifier_name, BenchmarkMetrics]
    """
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(classifiers))

    print(f"\n Mode PARALLELE - {len(classifiers)} classifieurs")
    print(f"   Workers: {num_workers}")
    print("   Note: Les temps ne sont pas objectifs en mode parallele!\n")

    # Trouver le nom du dataset
    dataset_name = None
    from datasets import DATASET_REGISTRY
    for name, cls in DATASET_REGISTRY.items():
        if isinstance(dataset, cls):
            dataset_name = name
            break

    if dataset_name is None:
        dataset_name = "fer2013"

    # Préparer les arguments
    worker_args = [
        (c.name.lower().replace(" ", "_").replace("-", "_"), dataset_name, limit, warmup)
        for c in classifiers
    ]

    results = {}

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as executor:
        futures = {
            executor.submit(_benchmark_worker, args): classifiers[i].name
            for i, args in enumerate(worker_args)
        }

        for future in futures:
            classifier_name = futures[future]
            try:
                metrics = future.result(timeout=600)
                if metrics:
                    results[classifier_name] = metrics
                    print(f"  {classifier_name}: {metrics.accuracy*100:.1f}% accuracy")
            except Exception as e:
                print(f"  {classifier_name}: Error - {e}")

    return results


def run_benchmark_sequential(
    classifiers: List[BaseEmotionClassifier],
    dataset: BaseEmotionDataset,
    limit: Optional[int] = None,
    warmup: int = 5
) -> Dict[str, BenchmarkMetrics]:
    """
    Exécute le benchmark séquentiellement (temps objectifs).

    Args:
        classifiers: Liste des classifieurs à tester
        dataset: Dataset à utiliser
        limit: Nombre maximum d'images
        warmup: Itérations de warmup

    Returns:
        Dict[classifier_name, BenchmarkMetrics]
    """
    print(f"\n Mode SEQUENTIEL - {len(classifiers)} classifieurs")
    print("   Les temps sont objectifs et comparables.\n")

    results = {}

    for classifier in classifiers:
        metrics = run_benchmark(
            classifier, dataset,
            limit=limit, warmup=warmup, quiet=False
        )
        results[classifier.name] = metrics

    return results
