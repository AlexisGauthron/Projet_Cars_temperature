# -*- coding: utf-8 -*-
"""
Moteur d'exécution STRICT du benchmark de classification d'émotions.

Ce module implémente un benchmark rigoureux avec:
- Warmup pour éliminer les temps d'initialisation
- time.perf_counter() pour une mesure précise
- Multi-passes par image pour statistiques (mean, std, min, max)
- Libération mémoire GPU entre les modèles
- Exécution séquentielle garantie
"""

import gc
import time
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.structures import EmotionLabel, EmotionPrediction, BenchmarkMetrics
from datasets.base import BaseEmotionDataset
from classifiers.base import BaseEmotionClassifier

# Import config partagée (depuis benchmarks/shared_config/)
try:
    from shared_config.strict import (
        WARMUP_IMAGES as DEFAULT_WARMUP,
        NUM_PASSES as DEFAULT_PASSES,
        CLEAR_GPU_MEMORY as DEFAULT_CLEAR_GPU,
    )
except ImportError:
    DEFAULT_WARMUP = 10
    DEFAULT_PASSES = 3
    DEFAULT_CLEAR_GPU = True


def clear_gpu_memory():
    """Libère la mémoire GPU (PyTorch, TensorFlow, ONNX)."""
    gc.collect()

    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass

    # TensorFlow
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except (ImportError, AttributeError):
        pass

    gc.collect()


def run_benchmark_strict(
    classifier: BaseEmotionClassifier,
    dataset: BaseEmotionDataset,
    limit: Optional[int] = None,
    warmup_images: int = DEFAULT_WARMUP,
    num_passes: int = DEFAULT_PASSES,
    clear_gpu: bool = DEFAULT_CLEAR_GPU,
    quiet: bool = False,
) -> BenchmarkMetrics:
    """
    Exécute le benchmark en mode STRICT pour des mesures rigoureuses.

    Args:
        classifier: Instance du classifieur à tester
        dataset: Instance du dataset à utiliser
        limit: Nombre maximum d'images à traiter
        warmup_images: Nombre d'images de warmup (ignorées pour le timing)
        num_passes: Nombre de passes par image pour les statistiques
        clear_gpu: Libérer la mémoire GPU avant de commencer
        quiet: Si True, supprime les outputs

    Returns:
        BenchmarkMetrics avec statistiques de timing détaillées
    """
    # 1. Libérer la mémoire GPU si demandé
    if clear_gpu:
        if not quiet:
            print(f"    Libération mémoire GPU...")
        clear_gpu_memory()

    # 2. Initialiser les métriques
    metrics = BenchmarkMetrics(classifier_name=classifier.name)

    # Ajouter attributs pour mode strict
    metrics.strict_mode = True
    metrics.warmup_images = warmup_images
    metrics.num_passes = num_passes
    metrics.time_min_ms = float('inf')
    metrics.time_max_ms = 0.0
    metrics.time_std_ms = 0.0
    metrics.all_times_ms = []

    # 3. Vérifier disponibilité
    if not classifier.is_available():
        if not quiet:
            print(f"    {classifier.name}: Non disponible")
        return metrics

    if not dataset.is_available():
        if not quiet:
            print(f"    Dataset non disponible")
        return metrics

    # 4. Charger les échantillons
    samples = dataset.get_samples()
    if limit:
        samples = samples[:limit]

    total_samples = len(samples)

    if not quiet:
        print(f"    Mode STRICT: {warmup_images} warmup, {num_passes} passes/image")
        print(f"    {total_samples} images à traiter")

    # 5. Phase de WARMUP
    if warmup_images > 0:
        if not quiet:
            print(f"    Warmup ({warmup_images} images)...", end="", flush=True)

        warmup_count = min(warmup_images, total_samples)
        for i in range(warmup_count):
            sample = samples[i % total_samples]
            image = dataset.get_image(sample)
            if image is not None:
                _ = classifier.predict(image)

        if not quiet:
            print(" OK")

    # 6. Phase de BENCHMARK
    all_image_times = []

    for i, sample in enumerate(samples):
        if not quiet and ((i + 1) % 100 == 0 or i == 0):
            print(f"\r    {classifier.name}: {i+1}/{total_samples}", end="", flush=True)

        image = dataset.get_image(sample)
        if image is None:
            continue

        gt_label = sample.ground_truth
        metrics.total_samples += 1

        # Multi-passes avec time.perf_counter()
        pass_times = []
        prediction = None

        for _ in range(num_passes):
            t_start = time.perf_counter()
            prediction = classifier.predict(image)
            t_end = time.perf_counter()
            pass_times.append((t_end - t_start) * 1000)  # ms

        # Statistiques pour cette image
        mean_time = statistics.mean(pass_times)
        all_image_times.append(mean_time)
        metrics.all_times_ms.append(mean_time)
        metrics.total_time_ms += mean_time

        # Mettre à jour min/max
        if mean_time < metrics.time_min_ms:
            metrics.time_min_ms = mean_time
        if mean_time > metrics.time_max_ms:
            metrics.time_max_ms = mean_time

        # Accuracy (utilise la dernière prédiction)
        if prediction:
            pred_label = prediction.label

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

    if not quiet:
        print()  # Nouvelle ligne

    # 7. Calculer les statistiques globales de timing
    if all_image_times:
        if len(all_image_times) > 1:
            metrics.time_std_ms = statistics.stdev(all_image_times)
        else:
            metrics.time_std_ms = 0.0

    return metrics


def run_all_classifiers_strict(
    classifiers: List[BaseEmotionClassifier],
    dataset: BaseEmotionDataset,
    limit: Optional[int] = None,
    warmup_images: int = DEFAULT_WARMUP,
    num_passes: int = DEFAULT_PASSES,
) -> Dict[str, BenchmarkMetrics]:
    """
    Exécute le benchmark strict pour tous les classifieurs séquentiellement.

    Garantit:
    - Un seul classifieur à la fois
    - Libération mémoire GPU entre chaque classifieur
    - Conditions identiques pour chaque modèle

    Args:
        classifiers: Liste des classifieurs à tester
        dataset: Dataset à utiliser
        limit: Nombre maximum d'images
        warmup_images: Nombre d'images de warmup
        num_passes: Nombre de passes par image

    Returns:
        Dict[classifier_name, BenchmarkMetrics]
    """
    results = {}
    total_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"BENCHMARK STRICT - {len(classifiers)} classifieurs")
    print(f"{'='*60}")
    print(f"  Warmup: {warmup_images} images")
    print(f"  Passes: {num_passes} par image")
    print(f"  Images: {limit or 'toutes'}")
    print(f"{'='*60}\n")

    for idx, classifier in enumerate(classifiers, 1):
        print(f"\n[{idx}/{len(classifiers)}] {classifier.name}")
        print(f"{'-'*40}")

        classifier_start = time.perf_counter()

        metrics = run_benchmark_strict(
            classifier=classifier,
            dataset=dataset,
            limit=limit,
            warmup_images=warmup_images,
            num_passes=num_passes,
            clear_gpu=True,
            quiet=False
        )

        classifier_time = time.perf_counter() - classifier_start
        results[classifier.name] = metrics

        # Afficher résumé
        print(f"    Terminé en {classifier_time:.1f}s")
        print(f"    Temps moyen: {metrics.avg_time_ms:.2f}ms (±{metrics.time_std_ms:.2f}ms)")
        print(f"    Min/Max: {metrics.time_min_ms:.2f}ms / {metrics.time_max_ms:.2f}ms")
        print(f"    Accuracy: {metrics.accuracy*100:.1f}%")

    total_time = time.perf_counter() - total_start

    print(f"\n{'='*60}")
    print(f"BENCHMARK STRICT TERMINÉ")
    print(f"  Temps total: {total_time:.1f}s")
    print(f"{'='*60}\n")

    return results
