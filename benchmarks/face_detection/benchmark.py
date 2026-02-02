#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de détection de visage.

CLI unifié avec les mêmes commandes que emotion_detection.

Usage:
    # Commandes de base
    python benchmark.py --list                    # Lister modèles + datasets
    python benchmark.py --list-models             # Lister les modèles
    python benchmark.py --list-datasets           # Lister les datasets
    python benchmark.py --list-profiles           # Lister les profils strict

    # Benchmark standard
    python benchmark.py -d wider_face -l 300      # 300 images de WIDER FACE
    python benchmark.py -m YuNet SCRFD -l 300     # Modèles spécifiques

    # Options communes
    python benchmark.py -e MTCNN RetinaFace       # Exclure des modèles
    python benchmark.py -p --workers 4            # Mode parallèle
    python benchmark.py -v                        # Mode verbeux
    python benchmark.py -f                        # Forcer recalcul

    # Mode strict (benchmark rigoureux)
    python benchmark.py --strict                  # Profil standard
    python benchmark.py --strict --profile publication
    python benchmark.py --strict -w 20 --passes 5 # Override manuel

    # Options spécifiques face detection
    python benchmark.py --iou 0.75                # Seuil IoU personnalisé
    python benchmark.py --timeout 5.0             # Timeout par image
"""

import argparse
import sys
from pathlib import Path

# Ajouter les chemins pour les imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports locaux (depuis face_detection/)
from config import RESULTS_DIR, SLOW_DETECTORS, DEFAULT_IOU_THRESHOLD
from config import STRICT_WARMUP_IMAGES, STRICT_NUM_PASSES
from core import (
    print_results,
    get_results_filepath,
    load_existing_results,
    merge_and_save_results,
)
from datasets import get_dataset, list_datasets
from detectors import get_all_detectors, list_detectors
from runner import run_benchmark, run_benchmark_parallel, run_all_detectors_strict

# Import config partagée (depuis benchmarks/shared_config/)
try:
    from shared_config import strict as shared_strict_config
    HAS_SHARED_CONFIG = True
except ImportError:
    HAS_SHARED_CONFIG = False


def get_strict_params(args):
    """Récupère les paramètres du mode strict selon le profil ou les overrides."""
    if HAS_SHARED_CONFIG:
        # Utiliser la config partagée
        if args.profile:
            profile = shared_strict_config.get_profile(args.profile)
            warmup = profile["warmup_images"]
            passes = profile["num_passes"]
        else:
            warmup = shared_strict_config.WARMUP_IMAGES
            passes = shared_strict_config.NUM_PASSES
    else:
        # Fallback sur config locale
        warmup = STRICT_WARMUP_IMAGES
        passes = STRICT_NUM_PASSES

    # Override manuel si spécifié
    if args.warmup is not None:
        warmup = args.warmup
    if args.passes is not None:
        passes = args.passes

    return warmup, passes


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de détection de visage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python benchmark.py -d wider_face -l 300
  python benchmark.py -m YuNet SCRFD -l 300
  python benchmark.py --strict --profile publication
  python benchmark.py --list
        """
    )

    # =========================================================================
    # ARGUMENTS COMMUNS (identiques à emotion_detection)
    # =========================================================================

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="wider_face",
        help="Dataset à utiliser (défaut: wider_face)"
    )

    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        help="Modèles à tester (défaut: tous)"
    )

    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Nombre maximum d'images à traiter"
    )

    parser.add_argument(
        "--exclude", "-e",
        type=str,
        nargs="+",
        help="Exclure des modèles spécifiques"
    )

    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Mode parallèle (défaut: séquentiel)"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Nombre de workers pour le mode parallèle"
    )

    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Chemin personnalisé pour l'export JSON"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Recalculer même si les résultats existent"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Affichage détaillé"
    )

    # =========================================================================
    # MODE STRICT (commun)
    # =========================================================================

    parser.add_argument(
        "--strict", "-s",
        action="store_true",
        help="Mode strict: warmup + multi-passes + stats timing"
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["quick", "standard", "publication"],
        help="Profil de benchmark strict"
    )

    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=None,
        help="Override: nombre d'images de warmup"
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=None,
        help="Override: nombre de passes par image"
    )

    # =========================================================================
    # LISTING (commun)
    # =========================================================================

    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister modèles et datasets disponibles"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lister les modèles disponibles"
    )

    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Lister les datasets disponibles"
    )

    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Lister les profils de benchmark strict"
    )

    # =========================================================================
    # SPÉCIFIQUE FACE DETECTION
    # =========================================================================

    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"Seuil IoU pour le matching (défaut: {DEFAULT_IOU_THRESHOLD})"
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout par image en secondes"
    )

    args = parser.parse_args()

    # =========================================================================
    # MODE LISTING
    # =========================================================================

    if args.list_profiles:
        if HAS_SHARED_CONFIG:
            shared_strict_config.print_profiles()
        else:
            print("Config partagée non disponible")
        return 0

    if args.list or args.list_models:
        print("\n" + "=" * 60)
        print("MODÈLES DISPONIBLES")
        print("=" * 60)
        detectors_info = list_detectors()
        for name, info in detectors_info.items():
            status = "OK" if info["is_available"] else "NOT AVAILABLE"
            print(f"  {name:<20} [{status}]")
        available = sum(1 for info in detectors_info.values() if info["is_available"])
        print(f"\nTotal: {available}/{len(detectors_info)} modèles disponibles")

        if not args.list_models:  # Si --list, afficher aussi les datasets
            args.list_datasets = True

    if args.list or args.list_datasets:
        print("\n" + "=" * 60)
        print("DATASETS DISPONIBLES")
        print("=" * 60)
        datasets_info = list_datasets()
        for name, info in datasets_info.items():
            status = "OK" if info["is_available"] else "NOT FOUND"
            print(f"  {name:<20} [{status}] - {info.get('description', '')}")

    if args.list or args.list_models or args.list_datasets:
        return 0

    # =========================================================================
    # BENCHMARK
    # =========================================================================

    print("\n" + "=" * 70)
    print("BENCHMARK - DÉTECTION DE VISAGE")
    print("=" * 70)

    # Charger le dataset
    dataset_name = args.dataset
    try:
        dataset = get_dataset(dataset_name)
    except ValueError as e:
        print(f"Erreur: {e}")
        return 1

    if not dataset.is_available():
        print(f"Dataset '{dataset_name}' non disponible")
        print("  Exécutez: python scripts/download_datasets.py")
        return 1

    print(f"\nDataset: {dataset.name}")
    annotations = dataset.get_annotations()
    total_faces = sum(len(boxes) for boxes in annotations.values())
    print(f"  {len(annotations)} images, {total_faces} visages annotés")

    # Déterminer le fichier de résultats
    results_filepath = Path(args.export) if args.export else get_results_filepath(
        dataset_name, args.limit, strict_mode=args.strict
    )
    print(f"\nFichier résultats: {results_filepath}")

    # Charger les résultats existants
    existing_results = load_existing_results(results_filepath)
    if existing_results:
        print(f"  Résultats existants: {list(existing_results.keys())}")

    # Charger les détecteurs
    all_detectors = get_all_detectors()
    print(f"\nModèles disponibles: {len(all_detectors)}")

    if args.verbose:
        for d in all_detectors:
            print(f"  - {d.name}")

    # Filtrer par modèles demandés
    if args.models:
        requested = [m.lower() for m in args.models]
        detectors = [d for d in all_detectors if d.name.lower() in requested]
        if not detectors:
            print(f"Erreur: Aucun modèle trouvé parmi: {args.models}")
            return 1
    else:
        detectors = all_detectors

    # Exclure des modèles
    if args.exclude:
        exclude_lower = [e.lower() for e in args.exclude]
        before = len(detectors)
        detectors = [d for d in detectors if d.name.lower() not in exclude_lower]
        print(f"  Exclusion: {args.exclude} ({before} -> {len(detectors)} modèles)")

    # Filtrer les modèles déjà calculés (sauf si --force)
    if existing_results and not args.force:
        already_done = [d.name for d in detectors if d.name in existing_results]
        detectors = [d for d in detectors if d.name not in existing_results]
        if already_done:
            print(f"  Déjà calculés (skip): {already_done}")
        if not detectors:
            print("\nTous les modèles demandés ont déjà des résultats!")
            print("  Utilisez --force pour recalculer.")
            return 0

    print(f"\nModèles à tester: {[d.name for d in detectors]}")

    # Configuration
    print(f"\nConfiguration:")
    print(f"  Limite: {args.limit or 'toutes'} images")
    print(f"  IoU: {args.iou}")
    if args.timeout:
        print(f"  Timeout: {args.timeout}s")

    # Exécuter le benchmark
    results = {}

    if args.strict:
        # Mode STRICT
        warmup, passes = get_strict_params(args)
        profile_name = args.profile or "standard"
        print(f"  Mode: STRICT (profil: {profile_name})")
        print(f"  Warmup: {warmup} images, Passes: {passes}")

        results = run_all_detectors_strict(
            detectors=detectors,
            dataset=dataset,
            limit=args.limit,
            iou_threshold=args.iou,
            warmup_images=warmup,
            num_passes=passes
        )

    elif args.parallel:
        # Mode parallèle
        print(f"  Mode: PARALLÈLE")
        print(f"  Note: Les temps ne sont pas objectifs en mode parallèle!")

        results = run_benchmark_parallel(
            detectors=detectors,
            dataset=dataset,
            limit=args.limit,
            iou_threshold=args.iou,
            timeout=args.timeout,
            num_workers=args.workers
        )

    else:
        # Mode séquentiel (défaut)
        print(f"  Mode: SÉQUENTIEL")

        import time
        start_time = time.time()

        for detector in detectors:
            print(f"\n  Testing {detector.name}...")
            metrics = run_benchmark(
                detector=detector,
                dataset=dataset,
                limit=args.limit,
                iou_threshold=args.iou,
                timeout_per_image=args.timeout
            )
            results[detector.name] = metrics

        elapsed = time.time() - start_time
        print(f"\n  Temps total: {elapsed:.1f}s")

    # Afficher les résultats
    if results:
        print_results(results, verbose=args.verbose)

    # Sauvegarder
    RESULTS_DIR.mkdir(exist_ok=True)
    merge_and_save_results(results, existing_results, results_filepath, dataset_name, args.limit)

    print("\nBenchmark terminé!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
