# -*- coding: utf-8 -*-
"""
Gestion des résultats du benchmark.
Export JSON, chargement, fusion incrémentale.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional

from .structures import BenchmarkMetrics
from .metrics import compute_ap

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR


def print_results(results: Dict[str, BenchmarkMetrics]):
    """Affiche les résultats du benchmark."""

    print("\n" + "="*80)
    print("📊 RÉSULTATS DU BENCHMARK")
    print("="*80)

    # Vérifier si mode strict (au moins un résultat en mode strict)
    is_strict = any(m.strict_mode for m in results.values())

    if is_strict:
        # Tableau avec statistiques de timing détaillées
        print(f"\n{'Détecteur':<20} {'Precision':>9} {'Recall':>9} {'F1':>9} {'Temps Moy':>11} {'±Std':>8} {'Min':>9} {'Max':>9}")
        print("-"*95)

        for name, m in results.items():
            print(f"{name:<20} {m.precision*100:>8.1f}% {m.recall*100:>8.1f}% "
                  f"{m.f1_score*100:>8.1f}% {m.avg_time_ms:>9.2f}ms "
                  f"±{m.time_std_ms:>6.2f} {m.time_min_ms:>7.2f}ms {m.time_max_ms:>7.2f}ms")
    else:
        # Tableau standard
        print(f"\n{'Détecteur':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AP':>10} {'Temps':>12}")
        print("-"*70)

        for name, m in results.items():
            ap = compute_ap(m.all_scores)
            print(f"{name:<15} {m.precision*100:>9.1f}% {m.recall*100:>9.1f}% "
                  f"{m.f1_score*100:>9.1f}% {ap*100:>9.1f}% {m.avg_time_ms:>10.1f}ms")

    # Par difficulté
    print("\n" + "="*80)
    print("📈 RÉSULTATS PAR DIFFICULTÉ")
    print("="*80)

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\n{difficulty.upper()}:")
        print(f"  {'Détecteur':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*50}")

        for name, m in results.items():
            d = m.__dict__[difficulty]
            if d["tp"] + d["fp"] == 0:
                p = 0
            else:
                p = d["tp"] / (d["tp"] + d["fp"])

            if d["tp"] + d["fn"] == 0:
                r = 0
            else:
                r = d["tp"] / (d["tp"] + d["fn"])

            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            print(f"  {name:<15} {p*100:>9.1f}% {r*100:>9.1f}% {f1*100:>9.1f}%")

    # Classement
    print("\n" + "="*80)
    print("🏆 CLASSEMENT")
    print("="*80)

    # Par AP
    sorted_by_ap = sorted(results.items(), key=lambda x: compute_ap(x[1].all_scores), reverse=True)
    print("\nPar AP (Average Precision):")
    for i, (name, m) in enumerate(sorted_by_ap[:5], 1):
        ap = compute_ap(m.all_scores)
        medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else "  "
        print(f"  {medal} {i}. {name}: {ap*100:.1f}%")

    # Par vitesse
    sorted_by_speed = sorted(results.items(), key=lambda x: x[1].avg_time_ms)
    print("\nPar vitesse:")
    for i, (name, m) in enumerate(sorted_by_speed[:5], 1):
        medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else "  "
        print(f"  {medal} {i}. {name}: {m.avg_time_ms:.1f}ms")


def export_results(results: Dict[str, BenchmarkMetrics], output_path: Path):
    """Exporte les résultats en JSON."""
    export_data = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {}
    }

    for name, m in results.items():
        export_data["results"][name] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "ap": compute_ap(m.all_scores),
            "avg_time_ms": m.avg_time_ms,
            "total_images": m.total_images,
            "total_gt_faces": m.total_gt_faces,
            "total_detected": m.total_detected,
            "tp": m.total_tp,
            "fp": m.total_fp,
            "fn": m.total_fn,
            "by_difficulty": {
                "easy": m.easy,
                "medium": m.medium,
                "hard": m.hard
            },
            "pr_scores": [[score, is_tp] for score, is_tp in m.all_scores]
        }

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"📁 Résultats exportés: {output_path}")


def load_results_from_json(json_path: Path) -> Dict[str, dict]:
    """Charge les résultats depuis un fichier JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_results_filepath(dataset: str, limit: Optional[int], strict_mode: bool = False) -> Path:
    """Génère le chemin du fichier de résultats basé sur le dataset et la limite."""
    RESULTS_DIR.mkdir(exist_ok=True)
    limit_str = f"_{limit}img" if limit else "_full"
    strict_str = "_strict" if strict_mode else ""
    return RESULTS_DIR / f"benchmark_{dataset}{limit_str}{strict_str}.json"


def load_existing_results(filepath: Path) -> Dict[str, dict]:
    """Charge les résultats existants depuis un fichier JSON."""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data.get("results", {})
        except Exception as e:
            print(f"  ⚠️ Erreur lecture fichier existant: {e}")
    return {}


def merge_and_save_results(
    new_results: Dict[str, BenchmarkMetrics],
    existing_results: Dict[str, dict],
    filepath: Path,
    dataset: str,
    limit: Optional[int],
    strict_mode: bool = False,
    warmup_images: int = 0,
    num_passes: int = 1
):
    """Fusionne les nouveaux résultats avec les existants et sauvegarde."""
    # Détecter automatiquement le mode strict depuis les résultats
    is_strict = strict_mode or any(m.strict_mode for m in new_results.values())

    export_data = {
        "benchmark_info": {
            "dataset": dataset,
            "limit": limit,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "strict_mode": is_strict
        },
        "results": {}
    }

    # Ajouter les paramètres strict si activé
    if is_strict:
        # Récupérer les params depuis le premier résultat strict
        for m in new_results.values():
            if m.strict_mode:
                warmup_images = m.warmup_images
                num_passes = m.num_passes
                break
        export_data["benchmark_info"]["strict_config"] = {
            "warmup_images": warmup_images,
            "num_passes": num_passes,
            "timing_method": "time.perf_counter()",
            "gpu_memory_cleared": True
        }

    # Ajouter les résultats existants
    for name, data in existing_results.items():
        export_data["results"][name] = data

    # Ajouter/Mettre à jour avec les nouveaux résultats
    for name, m in new_results.items():
        result_entry = {
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "ap": compute_ap(m.all_scores),
            "avg_time_ms": m.avg_time_ms,
            "total_images": m.total_images,
            "total_gt_faces": m.total_gt_faces,
            "total_detected": m.total_detected,
            "tp": m.total_tp,
            "fp": m.total_fp,
            "fn": m.total_fn,
            "by_difficulty": {
                "easy": m.easy,
                "medium": m.medium,
                "hard": m.hard
            },
            "pr_scores": [[score, is_tp] for score, is_tp in m.all_scores],
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Ajouter les statistiques de timing si mode strict
        if m.strict_mode:
            result_entry["strict_mode"] = True
            result_entry["timing_stats"] = {
                "time_min_ms": m.time_min_ms,
                "time_max_ms": m.time_max_ms,
                "time_std_ms": m.time_std_ms,
                "warmup_images": m.warmup_images,
                "num_passes": m.num_passes
            }

        export_data["results"][name] = result_entry

    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"📁 Résultats sauvegardés: {filepath}")
    print(f"   Total modèles: {len(export_data['results'])}")
