# -*- coding: utf-8 -*-
"""
Moteur d'exécution du benchmark de détection de visage.
Supporte l'exécution séquentielle et parallèle.
"""

import time
import sys
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import Manager

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox, BenchmarkMetrics
from core.metrics import match_detections, get_difficulty_level
from core.logger import get_logger
from datasets.base import BaseDataset
from detectors.base import BaseDetector
from detectors import DETECTOR_REGISTRY

# Logger pour ce module
logger = get_logger(__name__)

# Tqdm pour les barres de progression (optionnel)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def run_benchmark(
    detector: BaseDetector,
    dataset: BaseDataset,
    limit: Optional[int] = None,
    iou_threshold: float = 0.5,
    quiet: bool = False,
    progress_dict: Optional[Dict] = None,
    timeout_per_image: Optional[float] = None
) -> BenchmarkMetrics:
    """
    Exécute le benchmark pour un détecteur sur un dataset.

    Args:
        detector: Instance du détecteur à tester
        dataset: Instance du dataset à utiliser
        limit: Nombre maximum d'images à traiter
        iou_threshold: Seuil IoU pour considérer un match
        quiet: Si True, supprime les outputs
        progress_dict: Dict partagé pour le suivi de progression (multiprocessing)
        timeout_per_image: Timeout par image en secondes

    Returns:
        BenchmarkMetrics avec tous les résultats
    """
    metrics = BenchmarkMetrics(detector_name=detector.name)
    annotations = dataset.get_annotations()
    image_paths = list(annotations.keys())

    if limit:
        image_paths = image_paths[:limit]

    total_images = len(image_paths)
    skipped_timeout = 0

    for i, rel_path in enumerate(image_paths):
        # Mise à jour de la progression
        if progress_dict is not None:
            progress_dict[detector.name] = (i + 1, total_images)

        if not quiet and ((i + 1) % 100 == 0 or i == 0):
            print(f"\r  {detector.name}: {i+1}/{len(image_paths)}", end="", flush=True)

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

        # Détection avec timeout optionnel
        start_time = time.time()
        predictions = detector.detect(image)
        detection_time = (time.time() - start_time) * 1000

        # Vérifier si le temps dépasse le timeout (pour stats)
        if timeout_per_image is not None and detection_time > timeout_per_image * 1000:
            skipped_timeout += 1

        metrics.total_detected += len(predictions)
        metrics.total_time_ms += detection_time

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
        if skipped_timeout > 0:
            print(f"  ⚠️  {detector.name}: {skipped_timeout} images ont dépassé le timeout")

    return metrics


def run_benchmark_worker(args_tuple):
    """
    Worker function pour multiprocessing.
    Initialise le détecteur et le dataset dans le process enfant.
    """
    (detector_name, dataset_name, limit, iou_threshold,
     progress_dict, timeout) = args_tuple

    # Supprimer les outputs parasites
    import os
    import warnings
    from io import StringIO

    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Configurer TensorFlow si installé
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        try:
            tf.keras.utils.disable_interactive_logging()
        except AttributeError:
            pass  # Méthode non disponible dans certaines versions de TF
    except ImportError:
        pass  # TensorFlow non installé

    # Rediriger stdout/stderr
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = devnull
    sys.stderr = devnull

    try:
        # Importer les modules (dans le worker)
        from datasets import get_dataset
        from detectors import DETECTOR_REGISTRY

        # Créer le dataset
        dataset = get_dataset(dataset_name)
        if not dataset.is_available():
            return None

        # Créer le détecteur
        detector_class = DETECTOR_REGISTRY.get(detector_name)
        if detector_class is None:
            return None

        detector = detector_class()
        if not detector.is_available():
            return None

        # Initialiser la progression
        if progress_dict is not None:
            annotations = dataset.get_annotations()
            total = limit if limit else len(annotations)
            progress_dict[detector_name] = (0, total)

        # Exécuter le benchmark
        metrics = run_benchmark(
            detector, dataset, limit, iou_threshold,
            quiet=True, progress_dict=progress_dict, timeout_per_image=timeout
        )

        return metrics
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()


def run_benchmark_parallel(
    detectors: List[BaseDetector],
    dataset: BaseDataset,
    limit: Optional[int] = None,
    iou_threshold: float = 0.5,
    timeout: Optional[float] = None,
    num_workers: Optional[int] = None
) -> Dict[str, BenchmarkMetrics]:
    """
    Exécute le benchmark en parallèle pour plusieurs détecteurs.

    Args:
        detectors: Liste des détecteurs à tester
        dataset: Dataset à utiliser
        limit: Nombre maximum d'images
        iou_threshold: Seuil IoU
        timeout: Timeout par image
        num_workers: Nombre de workers (défaut: nb CPU)

    Returns:
        Dict[detector_name, BenchmarkMetrics]
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = min(num_workers, multiprocessing.cpu_count(), len(detectors))

    print(f"\n⚡ Mode PARALLÈLE - {len(detectors)} détecteurs simultanés")
    print(f"   Workers: {num_workers}")

    # Créer un dictionnaire partagé pour le suivi de progression
    manager = Manager()
    progress_dict = manager.dict()

    # Trouver le nom du dataset
    from datasets import DATASET_REGISTRY
    dataset_name = None
    for name, cls in DATASET_REGISTRY.items():
        if isinstance(dataset, cls):
            dataset_name = name
            break

    if dataset_name is None:
        dataset_name = "wider_face"
        logger.warning(f"Dataset type not found in registry, using fallback: {dataset_name}")

    worker_args = [
        (d.name, dataset_name, limit, iou_threshold, progress_dict, timeout)
        for d in detectors
    ]

    results = {}
    completed_detectors = set()
    errors = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_detector = {
            executor.submit(run_benchmark_worker, arg): arg[0]
            for arg in worker_args
        }

        print("\n📊 Progression:")

        if HAS_TQDM:
            # Mode tqdm
            annotations = dataset.get_annotations()
            total_images = limit or len(annotations)
            pbar_dict = {}
            for i, d in enumerate(detectors):
                pbar_dict[d.name] = tqdm(
                    total=total_images,
                    desc=f"  {d.name:<12}",
                    position=i,
                    leave=True,
                    ncols=80,
                    bar_format='{desc} {bar:20} {percentage:5.1f}% [{n_fmt}/{total_fmt}]'
                )

            all_done = False
            while not all_done:
                # Mettre à jour les barres
                for d in detectors:
                    name = d.name
                    if name in progress_dict and name not in completed_detectors:
                        current, total = progress_dict[name]
                        pbar = pbar_dict[name]
                        pbar.n = current
                        pbar.refresh()

                # Vérifier les futures terminées
                for future in list(future_to_detector.keys()):
                    if future.done():
                        detector_name = future_to_detector[future]
                        if detector_name not in completed_detectors:
                            completed_detectors.add(detector_name)
                            pbar_dict[detector_name].n = pbar_dict[detector_name].total
                            pbar_dict[detector_name].refresh()
                            try:
                                metrics = future.result()
                                if metrics is not None:
                                    results[detector_name] = metrics
                            except Exception as e:
                                errors[detector_name] = str(e)

                all_done = len(completed_detectors) == len(detectors)
                if not all_done:
                    time.sleep(0.2)

            for pbar in pbar_dict.values():
                pbar.close()
            print()

        else:
            # Mode simple
            all_done = False
            while not all_done:
                status_parts = []
                for d in detectors:
                    name = d.name
                    if name in completed_detectors:
                        status_parts.append(f"{name}:✅")
                    elif name in progress_dict:
                        current, total = progress_dict[name]
                        pct = int(current / total * 100) if total > 0 else 0
                        status_parts.append(f"{name}:{pct}%")
                    else:
                        status_parts.append(f"{name}:⏳")

                line = " | ".join(status_parts)
                sys.stdout.write(f"\r  {line[:100]:<100}")
                sys.stdout.flush()

                for future in list(future_to_detector.keys()):
                    if future.done():
                        detector_name = future_to_detector[future]
                        if detector_name not in completed_detectors:
                            completed_detectors.add(detector_name)
                            try:
                                metrics = future.result()
                                if metrics is not None:
                                    results[detector_name] = metrics
                            except Exception as e:
                                errors[detector_name] = str(e)

                all_done = len(completed_detectors) == len(detectors)
                if not all_done:
                    time.sleep(0.3)

            print()
            for d in detectors:
                name = d.name
                if name in results:
                    print(f"  ✅ {name}")
                elif name in errors:
                    print(f"  ❌ {name}")
                else:
                    print(f"  ❌ {name}")

    if errors:
        print("\n⚠️  Erreurs:")
        for name, err in errors.items():
            print(f"   {name}: {err}")

    parallel_time = time.time() - start_time
    print(f"\n⏱️  Temps total: {parallel_time:.1f}s")

    return results
