#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark PROFESSIONNEL des détecteurs de visage.

Point d'entrée unique pour exécuter les benchmarks.

Métriques calculées:
- Precision, Recall, F1-Score
- AP (Average Precision) - Standard PASCAL VOC
- IoU (Intersection over Union)
- Séparation par difficulté (Easy/Medium/Hard)

FONCTIONNALITÉ CLÉ: Résultats incrémentaux
- Les résultats sont sauvegardés par (dataset, limit) dans un fichier unique
- Ajouter un nouveau modèle → il s'ajoute aux résultats existants
- Utilisez --force pour recalculer des modèles déjà présents

Usage:
    # Benchmark de base sur WIDER FACE
    python benchmark.py --dataset wider_face --limit 300

    # Ajouter un modèle aux résultats existants (INCRÉMENTAL)
    python benchmark.py --dataset wider_face --limit 300 --models SCRFD

    # Tester plusieurs modèles spécifiques
    python benchmark.py --dataset wider_face --limit 300 --models YuNet SCRFD YOLOv8-face

    # Recalculer un modèle existant
    python benchmark.py --dataset wider_face --limit 300 --models YuNet --force

    # Lister les modèles et datasets disponibles
    python benchmark.py --list-models
    python benchmark.py --list-datasets

    # Options pour les détecteurs lents
    python benchmark.py --fast                       # Exclure MTCNN et RetinaFace
    python benchmark.py --exclude mtcnn retinaface   # Exclure des détecteurs spécifiques

    # Mode STRICT (benchmark rigoureux pour comparaison de temps juste)
    python benchmark.py --benchmark-strict           # Warmup + multi-passes + stats
    python benchmark.py --strict --warmup 20 --passes 5  # Personnaliser
"""

import argparse
import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RESULTS_DIR, SLOW_DETECTORS, DEFAULT_IOU_THRESHOLD,
    STRICT_WARMUP_IMAGES, STRICT_NUM_PASSES
)
from core import (
    print_results,
    get_results_filepath,
    load_existing_results,
    merge_and_save_results,
)
from datasets import get_dataset, list_datasets
from detectors import get_all_detectors, list_detectors
from runner import run_benchmark, run_benchmark_parallel, run_all_detectors_strict


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark professionnel de détection de visage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python benchmark.py --dataset wider_face --limit 300
  python benchmark.py --dataset wider_face --limit 300 --models YuNet SCRFD
  python benchmark.py --dataset wider_face --limit 300 --models YOLOv8-face  # Ajoute au fichier existant
  python benchmark.py --list-models
  python benchmark.py --list-datasets
        """
    )

    # Dataset et images
    parser.add_argument("--dataset", type=str, default="wider_face",
                        help="Dataset à utiliser (défaut: wider_face)")
    parser.add_argument("--limit", type=int, help="Limiter le nombre d'images")

    # Modèles
    parser.add_argument("--models", type=str, nargs="+",
                        help="Modèles à tester (défaut: tous). Ex: --models YuNet SCRFD")
    parser.add_argument("--fast", action="store_true",
                        help="Seulement les détecteurs rapides (exclut MTCNN, RetinaFace)")
    parser.add_argument("--exclude", type=str, nargs="+",
                        help="Exclure des détecteurs. Ex: --exclude mtcnn retinaface")

    # Options de benchmark
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD,
                        help=f"Seuil IoU (défaut: {DEFAULT_IOU_THRESHOLD})")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Timeout par image en secondes")
    parser.add_argument("--sequential", action="store_true",
                        help="Mode séquentiel (défaut: parallèle)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Nombre de workers parallèles")

    # Mode benchmark strict (mesures rigoureuses)
    parser.add_argument("--benchmark-strict", "--strict", action="store_true",
                        help="Mode strict: warmup + multi-passes + stats timing")
    parser.add_argument("--warmup", type=int, default=STRICT_WARMUP_IMAGES,
                        help=f"Nombre d'images de warmup (défaut: {STRICT_WARMUP_IMAGES})")
    parser.add_argument("--passes", type=int, default=STRICT_NUM_PASSES,
                        help=f"Nombre de passes par image (défaut: {STRICT_NUM_PASSES})")

    # Export
    parser.add_argument("--export", type=str, help="Chemin personnalisé pour l'export JSON")
    parser.add_argument("--force", action="store_true",
                        help="Recalculer même si les résultats existent déjà")

    # Listing
    parser.add_argument("--list-models", action="store_true",
                        help="Lister les modèles disponibles")
    parser.add_argument("--list-datasets", action="store_true",
                        help="Lister les datasets disponibles")

    args = parser.parse_args()

    # Mode: Lister les modèles
    if args.list_models:
        print("\n📋 MODÈLES DISPONIBLES")
        print("=" * 50)
        detectors_info = list_detectors()
        for name, info in detectors_info.items():
            status = "✓" if info["is_available"] else "✗"
            print(f"  {status} {name}")
        available = sum(1 for info in detectors_info.values() if info["is_available"])
        print(f"\nTotal: {available}/{len(detectors_info)} modèles disponibles")
        return

    # Mode: Lister les datasets
    if args.list_datasets:
        print("\n📋 DATASETS DISPONIBLES")
        print("=" * 50)
        datasets_info = list_datasets()
        for name, info in datasets_info.items():
            status = "✓" if info["is_available"] else "✗"
            print(f"  {status} {name:<20} - {info['description']}")
        return

    print("=" * 80)
    print("🔬 BENCHMARK PROFESSIONNEL - DÉTECTION DE VISAGE")
    print("=" * 80)

    # Charger le dataset
    dataset_name = args.dataset
    try:
        dataset = get_dataset(dataset_name)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not dataset.is_available():
        print(f"❌ Dataset '{dataset_name}' non disponible")
        print(f"   Annotations: {dataset.annotation_file}")
        print(f"   Images: {dataset.images_dir}")
        print("   Exécutez: python scripts/download_datasets.py")
        sys.exit(1)

    print(f"\n📂 Dataset: {dataset.name}")
    print(f"   Chargement des annotations...")
    annotations = dataset.get_annotations()
    total_faces = sum(len(boxes) for boxes in annotations.values())
    print(f"   {len(annotations)} images, {total_faces} visages annotés")

    # Déterminer le fichier de résultats
    results_filepath = Path(args.export) if args.export else get_results_filepath(
        dataset_name, args.limit, strict_mode=args.benchmark_strict
    )
    print(f"\n📄 Fichier résultats: {results_filepath}")

    # Charger les résultats existants
    existing_results = load_existing_results(results_filepath)
    if existing_results:
        print(f"   Résultats existants: {list(existing_results.keys())}")

    # Charger les détecteurs
    all_detectors = get_all_detectors()
    print(f"\n🔧 Détecteurs disponibles: {len(all_detectors)}")
    for d in all_detectors:
        print(f"   ✓ {d.name}")

    # Filtrer par modèles demandés
    if args.models:
        requested_models = [m.lower() for m in args.models]
        detectors = [d for d in all_detectors if d.name.lower() in requested_models]
        if not detectors:
            print(f"❌ Aucun modèle trouvé parmi: {args.models}")
            print(f"   Disponibles: {[d.name for d in all_detectors]}")
            sys.exit(1)
        print(f"\n   🎯 Modèles sélectionnés: {[d.name for d in detectors]}")
    else:
        detectors = all_detectors

    # Mode rapide: exclure les détecteurs lents
    if args.fast:
        before_count = len(detectors)
        detectors = [d for d in detectors if d.name not in SLOW_DETECTORS]
        print(f"   ⚡ Mode FAST: exclusion de {SLOW_DETECTORS} ({before_count} → {len(detectors)} détecteurs)")

    # Exclure des détecteurs spécifiques
    if args.exclude:
        exclude_lower = [e.lower() for e in args.exclude]
        before_count = len(detectors)
        detectors = [d for d in detectors if d.name.lower() not in exclude_lower]
        print(f"   🚫 Exclusion: {args.exclude} ({before_count} → {len(detectors)} détecteurs)")

    # Filtrer les modèles déjà calculés (sauf si --force)
    if existing_results and not args.force:
        already_done = [d.name for d in detectors if d.name in existing_results]
        detectors = [d for d in detectors if d.name not in existing_results]
        if already_done:
            print(f"\n   ✅ Déjà calculés (skip): {already_done}")
        if not detectors:
            print(f"\n✅ Tous les modèles demandés ont déjà des résultats!")
            print(f"   Utilisez --force pour recalculer.")
            return

    print(f"\n   🚀 Modèles à calculer: {[d.name for d in detectors]}")

    # Exécuter le benchmark
    print(f"\n🚀 Démarrage du benchmark (IoU={args.iou})...")
    if args.limit:
        print(f"   Limite: {args.limit} images")

    results = {}

    # Mode STRICT: benchmark rigoureux avec warmup et multi-passes
    if args.benchmark_strict:
        results = run_all_detectors_strict(
            detectors=detectors,
            dataset=dataset,
            limit=args.limit,
            iou_threshold=args.iou,
            warmup_images=args.warmup,
            num_passes=args.passes
        )
    # Mode parallèle par défaut (sauf si --sequential ou un seul détecteur)
    elif not args.sequential and len(detectors) > 1:
        results = run_benchmark_parallel(
            detectors=detectors,
            dataset=dataset,
            limit=args.limit,
            iou_threshold=args.iou,
            timeout=args.timeout,
            num_workers=args.workers
        )
    else:
        # Mode séquentiel standard
        if len(detectors) == 1:
            print(f"\n🔄 Mode SÉQUENTIEL (1 seul détecteur)")
        else:
            print(f"\n🔄 Mode SÉQUENTIEL (--sequential)")

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

        sequential_time = time.time() - start_time
        print(f"\n⏱️  Temps total: {sequential_time:.1f}s")

    # Afficher les résultats des nouveaux modèles
    if results:
        print_results(results)

    # Fusionner et sauvegarder les résultats (incrémental)
    RESULTS_DIR.mkdir(exist_ok=True)
    merge_and_save_results(results, existing_results, results_filepath, dataset_name, args.limit)


if __name__ == "__main__":
    main()
