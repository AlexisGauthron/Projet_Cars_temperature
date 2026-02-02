#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de classification d'émotions faciales.

CLI unifié avec les mêmes commandes que face_detection.

Usage:
    # Commandes de base
    python benchmark.py --list                    # Lister modèles + datasets
    python benchmark.py --list-models             # Lister les modèles
    python benchmark.py --list-datasets           # Lister les datasets
    python benchmark.py --list-profiles           # Lister les profils strict

    # Benchmark standard
    python benchmark.py -d fer2013 -l 500         # 500 images de FER2013
    python benchmark.py -m hsemotion deepface     # Modèles spécifiques

    # Options communes
    python benchmark.py -e rmn fer_pytorch        # Exclure des modèles
    python benchmark.py -p --workers 4            # Mode parallèle
    python benchmark.py -v                        # Mode verbeux
    python benchmark.py -f                        # Forcer recalcul

    # Mode strict (benchmark rigoureux)
    python benchmark.py --strict                  # Profil standard
    python benchmark.py --strict --profile publication
    python benchmark.py --strict -w 20 --passes 5 # Override manuel
"""

import argparse
import sys
from pathlib import Path

# Ajouter les chemins pour les imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DEFAULT_DATASET, DEFAULT_LIMIT
from datasets import get_dataset, list_datasets, DATASET_REGISTRY
from classifiers import get_classifier, list_classifiers, get_available_classifiers, CLASSIFIER_REGISTRY
from runner.engine import run_benchmark, run_benchmark_sequential, run_benchmark_parallel
from runner.strict_engine import run_all_classifiers_strict
from core.results import save_results, print_results_table, print_per_class_comparison
from core.metrics import print_classification_report, print_confusion_matrix

# Import config partagée (depuis benchmarks/shared_config/)
try:
    from shared_config import strict as shared_strict_config
    HAS_SHARED_CONFIG = True
except ImportError:
    HAS_SHARED_CONFIG = False
    # Valeurs par défaut
    DEFAULT_WARMUP = 10
    DEFAULT_PASSES = 3


def get_strict_params(args):
    """Récupère les paramètres du mode strict selon le profil ou les overrides."""
    if HAS_SHARED_CONFIG:
        if args.profile:
            profile = shared_strict_config.get_profile(args.profile)
            warmup = profile["warmup_images"]
            passes = profile["num_passes"]
        else:
            warmup = shared_strict_config.WARMUP_IMAGES
            passes = shared_strict_config.NUM_PASSES
    else:
        warmup = DEFAULT_WARMUP
        passes = DEFAULT_PASSES

    # Override manuel si spécifié
    if args.warmup is not None:
        warmup = args.warmup
    if args.passes is not None:
        passes = args.passes

    return warmup, passes


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de classification d'émotions faciales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python benchmark.py -d fer2013 -l 500
  python benchmark.py -m hsemotion deepface -l 500
  python benchmark.py --strict --profile publication
  python benchmark.py --list
        """
    )

    # =========================================================================
    # ARGUMENTS COMMUNS (identiques à face_detection)
    # =========================================================================

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset à utiliser (défaut: {DEFAULT_DATASET})"
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
        default=DEFAULT_LIMIT,
        help=f"Nombre maximum d'images à traiter (défaut: {DEFAULT_LIMIT})"
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
        help="Affichage détaillé (rapport par classe, matrice de confusion)"
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

    args = parser.parse_args()

    # =========================================================================
    # MODE LISTING
    # =========================================================================

    if args.list_profiles:
        if HAS_SHARED_CONFIG:
            shared_strict_config.print_profiles()
        else:
            print("Config partagée non disponible")
            print("Profils: quick, standard, publication")
        return 0

    if args.list or args.list_models:
        print("\n" + "=" * 60)
        print("MODÈLES DISPONIBLES")
        print("=" * 60)
        for name, info in list_classifiers().items():
            status = "OK" if info.get("is_available") else "NOT AVAILABLE"
            desc = info.get("description", "")
            print(f"  {name:<20} [{status}] {desc}")

    if args.list or args.list_datasets:
        print("\n" + "=" * 60)
        print("DATASETS DISPONIBLES")
        print("=" * 60)
        for name, info in list_datasets().items():
            status = "OK" if info.get("is_available") else "NOT FOUND"
            desc = info.get("description", "")
            print(f"  {name:<20} [{status}] {desc}")

    if args.list or args.list_models or args.list_datasets:
        return 0

    # =========================================================================
    # BENCHMARK
    # =========================================================================

    print("\n" + "=" * 70)
    print("BENCHMARK - CLASSIFICATION D'ÉMOTIONS")
    print("=" * 70)

    # Charger le dataset
    print(f"\nDataset: {args.dataset}")
    try:
        dataset = get_dataset(args.dataset)
        if not dataset.is_available():
            print(f"  Dataset '{args.dataset}' non disponible!")
            print(f"  Exécutez: python scripts/download_datasets.py {args.dataset}")
            return 1
        stats = dataset.get_stats()
        print(f"  Échantillons: {stats['total_samples']}")
        print(f"  Classes: {stats['num_classes']}")
    except Exception as e:
        print(f"  Erreur: {e}")
        return 1

    # Charger les classifieurs
    if args.models:
        classifier_names = args.models
    else:
        # Tous les classifieurs disponibles
        classifier_names = [name for name, info in list_classifiers().items()
                           if info.get("is_available")]

    # Exclure des modèles
    if args.exclude:
        exclude_lower = [e.lower() for e in args.exclude]
        before = len(classifier_names)
        classifier_names = [n for n in classifier_names if n.lower() not in exclude_lower]
        print(f"\n  Exclusion: {args.exclude} ({before} -> {len(classifier_names)} modèles)")

    classifiers = []
    print(f"\nModèles:")
    for name in classifier_names:
        try:
            classifier = get_classifier(name)
            if classifier.is_available():
                classifiers.append(classifier)
                print(f"  - {name}: OK")
            else:
                print(f"  - {name}: Non disponible")
        except Exception as e:
            print(f"  - {name}: Erreur - {e}")

    if not classifiers:
        print("\nAucun classifieur disponible!")
        print("  pip install deepface hsemotion fer rmn")
        return 1

    # Configuration
    print(f"\nConfiguration:")
    print(f"  Limite: {args.limit or 'toutes'} images")

    # Exécuter le benchmark
    results = {}

    if args.strict:
        # Mode STRICT
        warmup, passes = get_strict_params(args)
        profile_name = args.profile or "standard"
        print(f"  Mode: STRICT (profil: {profile_name})")
        print(f"  Warmup: {warmup} images, Passes: {passes}")

        results = run_all_classifiers_strict(
            classifiers=classifiers,
            dataset=dataset,
            limit=args.limit,
            warmup_images=warmup,
            num_passes=passes
        )

    elif args.parallel:
        # Mode parallèle
        print(f"  Mode: PARALLÈLE")
        print(f"  Note: Les temps ne sont pas objectifs en mode parallèle!")

        results = run_benchmark_parallel(
            classifiers, dataset,
            limit=args.limit,
            warmup=args.warmup or 5,
            num_workers=args.workers
        )

    else:
        # Mode séquentiel (défaut)
        print(f"  Mode: SÉQUENTIEL")

        results = run_benchmark_sequential(
            classifiers, dataset,
            limit=args.limit,
            warmup=args.warmup or 5
        )

    # Afficher les résultats
    print_results_table(results)

    if args.verbose:
        print_per_class_comparison(results)

        for name, metrics in results.items():
            print_classification_report(metrics)
            print("\nMatrice de confusion:")
            print_confusion_matrix(metrics)

    # Sauvegarder les résultats
    output_dir = Path(args.export).parent if args.export else None
    save_results(results, args.dataset, output_dir)

    print("\nBenchmark terminé!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
