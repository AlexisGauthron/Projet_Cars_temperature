# -*- coding: utf-8 -*-
"""
Gestion des résultats du benchmark d'émotions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .structures import BenchmarkMetrics, EmotionLabel
from .metrics import compute_confusion_matrix

RESULTS_DIR = Path(__file__).parent.parent / "results"


def save_results(
    results: Dict[str, BenchmarkMetrics],
    dataset_name: str,
    output_dir: Path = None
) -> Path:
    """
    Sauvegarde les résultats du benchmark.

    Args:
        results: Dict[classifier_name, BenchmarkMetrics]
        dataset_name: Nom du dataset utilisé
        output_dir: Dossier de sortie (défaut: results/)

    Returns:
        Chemin du fichier JSON sauvegardé
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Générer le nom du fichier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_benchmark_{dataset_name}_{timestamp}.json"
    filepath = output_dir / filename

    # Construire les données
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "num_classifiers": len(results),
        },
        "results": {
            name: metrics.to_dict()
            for name, metrics in results.items()
        }
    }

    # Sauvegarder
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n Results saved to: {filepath}")
    return filepath


def load_results(filepath: Path) -> Dict:
    """Charge les résultats depuis un fichier JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_results_table(results: Dict[str, BenchmarkMetrics]):
    """Affiche un tableau récapitulatif des résultats."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS - EMOTION CLASSIFICATION")
    print("=" * 80)

    # Header
    header = f"{'Classifier':<20} {'Accuracy':>10} {'Macro-F1':>10} {'W-F1':>10} {'Time(ms)':>10} {'FPS':>8}"
    print(header)
    print("-" * 80)

    # Trier par accuracy décroissante
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].accuracy,
        reverse=True
    )

    for name, metrics in sorted_results:
        print(
            f"{name:<20} "
            f"{metrics.accuracy*100:>9.1f}% "
            f"{metrics.macro_f1*100:>9.1f}% "
            f"{metrics.weighted_f1*100:>9.1f}% "
            f"{metrics.avg_time_ms:>10.2f} "
            f"{metrics.fps:>8.1f}"
        )

    print("-" * 80)

    # Meilleur modèle
    if sorted_results:
        best_name, best_metrics = sorted_results[0]
        print(f"\n Best Classifier: {best_name} (Accuracy: {best_metrics.accuracy:.1%})")


def print_per_class_comparison(results: Dict[str, BenchmarkMetrics]):
    """Affiche une comparaison par classe entre les classifieurs."""
    labels = EmotionLabel.fer2013_labels()
    classifiers = list(results.keys())

    print("\n" + "=" * 80)
    print("PER-CLASS ACCURACY COMPARISON")
    print("=" * 80)

    # Header
    header = f"{'Emotion':<12}" + "".join(f"{c[:10]:>12}" for c in classifiers)
    print(header)
    print("-" * (12 + 12 * len(classifiers)))

    for label in labels:
        row = f"{label.name:<12}"
        for classifier in classifiers:
            acc = results[classifier].per_class_accuracy(label) * 100
            row += f"{acc:>11.1f}%"
        print(row)


def generate_html_report(
    results: Dict[str, BenchmarkMetrics],
    dataset_name: str,
    output_dir: Path = None
) -> Path:
    """Génère un rapport HTML interactif."""
    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_report_{dataset_name}_{timestamp}.html"
    filepath = output_dir / filename

    # Préparer les données pour le graphique
    classifiers = list(results.keys())
    accuracies = [results[c].accuracy * 100 for c in classifiers]
    f1_scores = [results[c].macro_f1 * 100 for c in classifiers]
    times = [results[c].avg_time_ms for c in classifiers]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Classification Benchmark - {dataset_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-container {{ height: 400px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .best {{ background: #e8f5e9 !important; font-weight: bold; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .label {{ font-size: 12px; color: #666; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; }}
        .stat-box {{ text-align: center; padding: 15px; background: #f9f9f9; border-radius: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Emotion Classification Benchmark</h1>
        <p>Dataset: <strong>{dataset_name}</strong> | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="card">
            <h2>Summary</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="metric">{len(classifiers)}</div>
                    <div class="label">Classifiers</div>
                </div>
                <div class="stat-box">
                    <div class="metric">{max(accuracies):.1f}%</div>
                    <div class="label">Best Accuracy</div>
                </div>
                <div class="stat-box">
                    <div class="metric">{max(f1_scores):.1f}%</div>
                    <div class="label">Best F1-Score</div>
                </div>
                <div class="stat-box">
                    <div class="metric">{min(times):.1f}ms</div>
                    <div class="label">Fastest</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Accuracy Comparison</h2>
            <div class="chart-container">
                <canvas id="accuracyChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Results Table</h2>
            <table>
                <tr>
                    <th>Classifier</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Weighted F1</th>
                    <th>Time (ms)</th>
                    <th>FPS</th>
                </tr>
                {"".join(f'''
                <tr class="{'best' if results[c].accuracy == max(r.accuracy for r in results.values()) else ''}">
                    <td>{c}</td>
                    <td>{results[c].accuracy*100:.1f}%</td>
                    <td>{results[c].macro_f1*100:.1f}%</td>
                    <td>{results[c].weighted_f1*100:.1f}%</td>
                    <td>{results[c].avg_time_ms:.2f}</td>
                    <td>{results[c].fps:.1f}</td>
                </tr>''' for c in classifiers)}
            </table>
        </div>

        <div class="card">
            <h2>Per-Class Performance</h2>
            <div class="chart-container">
                <canvas id="perClassChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Accuracy Chart
        new Chart(document.getElementById('accuracyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(classifiers)},
                datasets: [
                    {{
                        label: 'Accuracy (%)',
                        data: {json.dumps(accuracies)},
                        backgroundColor: 'rgba(76, 175, 80, 0.7)',
                    }},
                    {{
                        label: 'Macro F1 (%)',
                        data: {json.dumps(f1_scores)},
                        backgroundColor: 'rgba(33, 150, 243, 0.7)',
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
            }}
        }});

        // Per-class Chart
        const emotions = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL'];
        const perClassData = {{
            labels: emotions,
            datasets: {json.dumps([
                {
                    "label": c,
                    "data": [results[c].per_class_accuracy(EmotionLabel[e]) * 100 for e in ["ANGRY", "DISGUST", "FEAR", "HAPPY", "SAD", "SURPRISE", "NEUTRAL"]]
                }
                for c in classifiers
            ])}
        }};

        new Chart(document.getElementById('perClassChart'), {{
            type: 'radar',
            data: perClassData,
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{ r: {{ beginAtZero: true, max: 100 }} }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f" HTML report saved to: {filepath}")
    return filepath
