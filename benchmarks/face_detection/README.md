# Benchmark Détection de Visage - Application Automobile

Benchmark professionnel de modèles de détection de visage optimisé pour les applications automobiles (Driver Monitoring System).

## Structure du Projet

```
benchmarks/face_detection/
├── benchmark.py              # Point d'entrée principal (modulaire)
├── config.py                 # Configuration globale
├── core/                     # Structures et métriques
│   ├── structures.py         # BBox, DetectionResult, BenchmarkMetrics
│   ├── metrics.py            # IoU, matching, AP
│   └── results.py            # Export/import JSON, affichage
├── datasets/                 # Loaders de datasets
│   ├── base.py               # Interface BaseDataset
│   └── wider_face.py         # WIDER FACE loader
├── detectors/                # Détecteurs de visage
│   ├── base.py               # Interface BaseDetector
│   ├── yunet.py              # YuNet (OpenCV)
│   ├── opencv_dnn.py         # OpenCV DNN (SSD ResNet)
│   ├── haar.py               # Haar Cascade
│   ├── mtcnn.py              # MTCNN
│   ├── retinaface.py         # RetinaFace
│   ├── scrfd.py              # SCRFD (InsightFace)
│   ├── yolov8_face.py        # YOLOv8-face
│   └── yolov11_face.py       # YOLOv11-face
├── runner/                   # Moteur d'exécution
│   └── engine.py             # run_benchmark, run_benchmark_parallel
├── models/                   # Fichiers modèles (.onnx, .caffemodel, etc.)
├── datasets/                 # Images des datasets
├── annotations/              # Ground truth
├── scripts/                  # Scripts utilitaires
│   ├── download_models.py
│   └── download_datasets.py
└── results/                  # Résultats JSON des benchmarks
```

---

## Installation Rapide

```bash
# 1. Télécharger les modèles
python scripts/download_models.py

# 2. Télécharger WIDER FACE + annotations
python scripts/download_datasets.py

# 3. Installer les dépendances optionnelles
pip install insightface ultralytics huggingface_hub  # SCRFD, YOLOv8, YOLOv11
pip install mtcnn retinaface                          # MTCNN, RetinaFace
```

---

## CLI - benchmark.py

Point d'entrée principal du benchmark avec architecture modulaire.

### Commandes de Base

| Commande | Description |
|----------|-------------|
| `python benchmark.py --list-models` | Lister les modèles disponibles |
| `python benchmark.py --list-datasets` | Lister les datasets disponibles |
| `python benchmark.py --dataset wider_face` | Benchmark sur WIDER FACE |
| `python benchmark.py --limit 300` | Limiter à 300 images |
| `python benchmark.py --models YuNet SCRFD` | Tester des modèles spécifiques |

### Options Avancées

| Option | Description |
|--------|-------------|
| `--fast` | Exclure MTCNN et RetinaFace (lents) |
| `--exclude <noms>` | Exclure des détecteurs spécifiques |
| `--sequential` | Mode séquentiel (défaut: parallèle) |
| `--workers N` | Nombre de workers parallèles |
| `--iou <seuil>` | Seuil IoU (défaut: 0.5) |
| `--timeout <sec>` | Timeout par image |
| `--force` | Recalculer même si résultats existants |
| `--export <path>` | Chemin personnalisé pour l'export JSON |

### Fonctionnalité Clé: Résultats Incrémentaux

Les résultats sont sauvegardés par `(dataset, limit)` dans un fichier unique. Ajouter un nouveau modèle met à jour le fichier existant sans recalculer les autres.

```bash
# Premier benchmark avec 3 modèles
python benchmark.py --dataset wider_face --limit 300 --models YuNet OpenCV-DNN Haar

# Ajouter SCRFD aux résultats existants (ne recalcule pas YuNet, OpenCV-DNN, Haar)
python benchmark.py --dataset wider_face --limit 300 --models SCRFD

# Forcer le recalcul de YuNet
python benchmark.py --dataset wider_face --limit 300 --models YuNet --force
```

### Exemples

```bash
# Lister les modèles disponibles
python benchmark.py --list-models

# Benchmark rapide (sans modèles lents)
python benchmark.py --dataset wider_face --limit 300 --fast

# Benchmark complet sur WIDER FACE
python benchmark.py --dataset wider_face

# Tester uniquement les modèles YOLO
python benchmark.py --dataset wider_face --limit 500 --models YOLOv8-face YOLOv11-face

# Mode séquentiel avec timeout
python benchmark.py --dataset wider_face --sequential --timeout 5

# Exclure des modèles spécifiques
python benchmark.py --dataset wider_face --exclude haar mtcnn
```

---

## Métriques Calculées

| Métrique | Description | Formule |
|----------|-------------|---------|
| **Precision** | Exactitude des détections | TP / (TP + FP) |
| **Recall** | Taux de détection | TP / (TP + FN) |
| **F1-Score** | Moyenne harmonique | 2 × P × R / (P + R) |
| **AP** | Average Precision (PASCAL VOC) | Aire sous courbe PR |
| **IoU** | Intersection over Union | Seuil par défaut: 0.5 |
| **Temps moyen** | Latence par image | ms/image |

### Résultats par Difficulté

Le benchmark calcule automatiquement les métriques séparées selon les attributs WIDER FACE:

| Niveau | Critères |
|--------|----------|
| **Easy** | Visages clairs, sans occlusion, pose typique |
| **Medium** | Flou léger ou occlusion partielle |
| **Hard** | Flou fort, occlusion forte, pose atypique |

---

## Architecture Modulaire

### Ajouter un Nouveau Dataset

1. Créer `datasets/mon_dataset.py`:

```python
from .base import BaseDataset

class MonDataset(BaseDataset):
    name = "Mon Dataset"
    description = "Description du dataset"

    @property
    def annotation_file(self) -> Path:
        return ANNOTATIONS_DIR / "mon_dataset" / "annotations.txt"

    @property
    def images_dir(self) -> Path:
        return DATASETS_DIR / "mon_dataset"

    def load_annotations(self) -> Dict[str, List[BBox]]:
        # Charger les annotations au format BBox
        pass
```

2. L'ajouter au registry dans `datasets/__init__.py`:

```python
from .mon_dataset import MonDataset

DATASET_REGISTRY = {
    "wider_face": WiderFaceDataset,
    "mon_dataset": MonDataset,  # Ajouter ici
}
```

### Ajouter un Nouveau Détecteur

1. Créer `detectors/mon_detecteur.py`:

```python
from .base import BaseDetector

class MonDetecteur(BaseDetector):
    name = "MonDetecteur"

    def __init__(self):
        # Charger le modèle
        pass

    def detect(self, image: np.ndarray) -> List[BBox]:
        # Retourner les détections
        pass

    def is_available(self) -> bool:
        return self.model is not None
```

2. L'ajouter au registry dans `detectors/__init__.py`.

---

## Scripts Utilitaires

### download_models.py

```bash
# Lister les modèles
python scripts/download_models.py --list

# Télécharger tous les modèles
python scripts/download_models.py

# Télécharger un modèle spécifique
python scripts/download_models.py --model yunet
python scripts/download_models.py --model scrfd
```

### download_datasets.py

```bash
# Voir le statut
python scripts/download_datasets.py --list

# Télécharger WIDER FACE + annotations
python scripts/download_datasets.py
```

---

## Résultats

Les résultats sont exportés en JSON dans `results/`.

### Format des Fichiers

- `benchmark_wider_face_full.json` : Benchmark complet
- `benchmark_wider_face_300img.json` : Benchmark limité à 300 images

### Exemple de Sortie JSON

```json
{
  "benchmark_info": {
    "dataset": "wider_face",
    "limit": null,
    "last_updated": "2026-01-24 18:48:11"
  },
  "results": {
    "YuNet": {
      "precision": 0.929,
      "recall": 0.485,
      "f1_score": 0.637,
      "ap": 0.988,
      "avg_time_ms": 62.9,
      "by_difficulty": {
        "easy": {"tp": 2847, "fp": 140, "fn": 265},
        "medium": {"tp": 3888, "fp": 158, "fn": 632},
        "hard": {"tp": 12273, "fp": 1151, "fn": 19325}
      }
    }
  }
}
```

---

## Recommandations par Cas d'Usage

| Cas d'usage | Modèle recommandé | Raison |
|-------------|-------------------|--------|
| **Embarqué temps réel** | YuNet | 230KB, ~1ms, robuste |
| **Meilleur compromis** | OpenCV-DNN | 99% précision, ~35ms |
| **State-of-the-art** | SCRFD / YOLOv11-face | Meilleur AP, GPU recommandé |
| **Haute précision** | RetinaFace | 97% précision, lent |
| **CPU uniquement** | YuNet ou OpenCV-DNN | Pas besoin de GPU |

---

## Documentation

- [MODELS.md](MODELS.md) - Détails sur chaque modèle de détection
- [DATASETS.md](DATASETS.md) - Détails sur les datasets disponibles
- [BENCHMARKS.md](BENCHMARKS.md) - Résultats de benchmarks publiés (état de l'art)

---

## Ressources

- [WIDER FACE Benchmark](http://shuoyang1213.me/WIDERFACE/)
- [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [OpenCV Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
