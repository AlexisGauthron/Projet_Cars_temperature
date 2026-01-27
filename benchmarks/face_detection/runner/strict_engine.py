# -*- coding: utf-8 -*-
"""
Moteur d'exécution STRICT du benchmark de détection de visage.

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

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox, BenchmarkMetrics
from core.metrics import match_detections, get_difficulty_level
from datasets.base import BaseDataset
from detectors.base import BaseDetector
from config import STRICT_WARMUP_IMAGES, STRICT_NUM_PASSES, STRICT_CLEAR_GPU_MEMORY


def clear_gpu_memory():
    """Libère la mémoire GPU (PyTorch, TensorFlow, ONNX)."""
    # Garbage collection Python
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

    # ONNX Runtime - pas de clear explicite, mais gc devrait aider
    gc.collect()


def run_benchmark_strict(
    detector: BaseDetector,
    dataset: BaseDataset,
    limit: Optional[int] = None,
    iou_threshold: float = 0.5,
    warmup_images: int = STRICT_WARMUP_IMAGES,
    num_passes: int = STRICT_NUM_PASSES,
    clear_gpu: bool = STRICT_CLEAR_GPU_MEMORY,
    quiet: bool = False,
) -> BenchmarkMetrics:
    """
    Exécute le benchmark en mode STRICT pour des mesures rigoureuses.

    Args:
        detector: Instance du détecteur à tester
        dataset: Instance du dataset à utiliser
        limit: Nombre maximum d'images à traiter
        iou_threshold: Seuil IoU pour considérer un match
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
            print(f"  🧹 Libération mémoire GPU...")
        clear_gpu_memory()

    # 2. Initialiser les métriques
    metrics = BenchmarkMetrics(
        detector_name=detector.name,
        strict_mode=True,
        warmup_images=warmup_images,
        num_passes=num_passes
    )

    # 3. Charger les annotations et images
    annotations = dataset.get_annotations()
    image_paths = list(annotations.keys())

    if limit:
        image_paths = image_paths[:limit]

    total_images = len(image_paths)

    if not quiet:
        print(f"  📊 Mode STRICT: {warmup_images} warmup, {num_passes} passes/image")
        print(f"  📷 {total_images} images à traiter")

    # 4. Phase de WARMUP
    if warmup_images > 0 and not quiet:
        print(f"  🔥 Warmup ({warmup_images} images)...", end="", flush=True)

    warmup_count = min(warmup_images, total_images)
    for i in range(warmup_count):
        rel_path = image_paths[i % total_images]
        image_path = dataset.get_image_path(rel_path)
        if image_path:
            image = cv2.imread(str(image_path))
            if image is not None:
                _ = detector.detect(image)

    if warmup_images > 0 and not quiet:
        print(" OK")

    # 5. Phase de BENCHMARK
    all_image_times = []  # Temps moyen par image (sur les passes)

    for i, rel_path in enumerate(image_paths):
        if not quiet and ((i + 1) % 50 == 0 or i == 0):
            print(f"\r  ⏱️  {detector.name}: {i+1}/{total_images}", end="", flush=True)

        # Trouver l'image
        image_path = dataset.get_image_path(rel_path)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        gt_boxes = annotations[rel_path]
        metrics.total_images += 1
        metrics.total_gt_faces += len([b for b in gt_boxes if not b.invalid])

        # Multi-passes avec time.perf_counter()
        pass_times = []
        predictions = None

        for pass_idx in range(num_passes):
            t_start = time.perf_counter()
            predictions = detector.detect(image)
            t_end = time.perf_counter()
            pass_times.append((t_end - t_start) * 1000)  # ms

        # Statistiques pour cette image
        mean_time = statistics.mean(pass_times)
        all_image_times.append(mean_time)
        metrics.all_times_ms.append(mean_time)
        metrics.total_time_ms += mean_time

        # Comptage des détections (utilise la dernière passe)
        if predictions:
            metrics.total_detected += len(predictions)

            # Matching
            tp, fp, fn, scores = match_detections(predictions, gt_boxes, iou_threshold)

            metrics.total_tp += tp
            metrics.total_fp += fp
            metrics.total_fn += fn
            metrics.all_scores.extend(scores)

            # Par difficulté
            difficulty = get_difficulty_level(gt_boxes)
            metrics.__dict__[difficulty]["tp"] += tp
            metrics.__dict__[difficulty]["fp"] += fp
            metrics.__dict__[difficulty]["fn"] += fn
            metrics.__dict__[difficulty]["total"] += len([b for b in gt_boxes if not b.invalid])

    if not quiet:
        print()  # Nouvelle ligne

    # 6. Calculer les statistiques globales de timing
    if all_image_times:
        metrics.time_min_ms = min(all_image_times)
        metrics.time_max_ms = max(all_image_times)
        if len(all_image_times) > 1:
            metrics.time_std_ms = statistics.stdev(all_image_times)
        else:
            metrics.time_std_ms = 0.0

    return metrics


def run_all_detectors_strict(
    detectors: List[BaseDetector],
    dataset: BaseDataset,
    limit: Optional[int] = None,
    iou_threshold: float = 0.5,
    warmup_images: int = STRICT_WARMUP_IMAGES,
    num_passes: int = STRICT_NUM_PASSES,
) -> Dict[str, BenchmarkMetrics]:
    """
    Exécute le benchmark strict pour tous les détecteurs séquentiellement.

    Garantit:
    - Un seul détecteur à la fois
    - Libération mémoire GPU entre chaque détecteur
    - Conditions identiques pour chaque modèle

    Args:
        detectors: Liste des détecteurs à tester
        dataset: Dataset à utiliser
        limit: Nombre maximum d'images
        iou_threshold: Seuil IoU
        warmup_images: Nombre d'images de warmup
        num_passes: Nombre de passes par image

    Returns:
        Dict[detector_name, BenchmarkMetrics]
    """
    results = {}
    total_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"🔬 BENCHMARK STRICT - {len(detectors)} détecteurs")
    print(f"{'='*60}")
    print(f"   Warmup: {warmup_images} images")
    print(f"   Passes: {num_passes} par image")
    print(f"   Images: {limit or 'toutes'}")
    print(f"{'='*60}\n")

    for idx, detector in enumerate(detectors, 1):
        print(f"\n[{idx}/{len(detectors)}] {detector.name}")
        print(f"{'-'*40}")

        detector_start = time.perf_counter()

        metrics = run_benchmark_strict(
            detector=detector,
            dataset=dataset,
            limit=limit,
            iou_threshold=iou_threshold,
            warmup_images=warmup_images,
            num_passes=num_passes,
            clear_gpu=True,
            quiet=False
        )

        detector_time = time.perf_counter() - detector_start
        results[detector.name] = metrics

        # Afficher résumé pour ce détecteur
        print(f"  ✅ Terminé en {detector_time:.1f}s")
        print(f"     Temps moyen: {metrics.avg_time_ms:.2f}ms (±{metrics.time_std_ms:.2f}ms)")
        print(f"     Min/Max: {metrics.time_min_ms:.2f}ms / {metrics.time_max_ms:.2f}ms")
        print(f"     Precision: {metrics.precision*100:.1f}%  Recall: {metrics.recall*100:.1f}%")

    total_time = time.perf_counter() - total_start

    print(f"\n{'='*60}")
    print(f"✅ BENCHMARK STRICT TERMINÉ")
    print(f"   Temps total: {total_time:.1f}s")
    print(f"{'='*60}\n")

    return results
