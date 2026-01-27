# Datasets pour le Benchmark de Détection de Visage

Ce document liste tous les datasets utilisés pour évaluer les performances des détecteurs de visage.

## Vue d'ensemble

| Dataset | Images | Résolution | Difficulté | Taille | Status |
|---------|--------|------------|------------|--------|--------|
| WIDER FACE | 3,226 | Variable | Easy/Medium/Hard | ~850MB | Principal |
| LFW | 13,233 | 250×250 | Facile | ~180MB | Optionnel |
| UTKFace | 20,000+ | 200×200 | Facile | ~120MB | Optionnel |
| FDDB | 2,845 | Variable | Moyen | ~50MB | Optionnel |
| DARK FACE | 6,000 | Variable | Low-light | ~2GB | Manuel |
| MAFA | 30,811 | Variable | Masques/Occlusions | ~4GB | Manuel |

---

## Dataset Principal

### WIDER FACE (Recommandé)

Le dataset de référence pour le benchmark de détection de visage, utilisé par défaut.

| Caractéristique | Valeur |
|-----------------|--------|
| **Images** | 3,226 (validation set) |
| **Visages annotés** | 39,112 |
| **Résolution** | Variable (de 50×50 à 4K) |
| **Annotations** | Bounding boxes + attributs |
| **Difficultés** | Easy, Medium, Hard |

**Attributs annotés :**

| Attribut | Valeurs | Description |
|----------|---------|-------------|
| `blur` | 0, 1, 2 | Clair / Normal / Fort |
| `occlusion` | 0, 1, 2 | Aucune / Partielle / Forte |
| `pose` | 0, 1 | Typique / Atypique |
| `invalid` | 0, 1 | Valide / Invalide |

**Distribution par difficulté :**

| Difficulté | Visages | Critères |
|------------|---------|----------|
| Easy | 3,112 | blur=0, occlusion=0, pose=0 |
| Medium | 4,520 | blur=1 ou occlusion=1 |
| Hard | 31,480 | blur=2, occlusion=2, ou pose=1 |

**Téléchargement :**
```bash
python scripts/download_datasets.py
```

**Source :** [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) | [HuggingFace](https://huggingface.co/datasets/wider_face)

---

## Datasets Optionnels

### LFW (Labeled Faces in the Wild)

Dataset classique pour la reconnaissance faciale, utile pour tester la détection sur des visages bien cadrés.

| Caractéristique | Valeur |
|-----------------|--------|
| **Images** | 13,233 |
| **Personnes** | 5,749 |
| **Résolution** | 250×250 |

**Caractéristiques :**
- Visages centrés et alignés
- Conditions "in the wild" (non contrôlées)
- Principalement utilisé pour la reconnaissance

**Source :** [UMass Vision](http://vis-www.cs.umass.edu/lfw/)

---

### FDDB (Face Detection Data Set and Benchmark)

Dataset standard pour la détection de visages avec annotations elliptiques.

| Caractéristique | Valeur |
|-----------------|--------|
| **Images** | 2,845 |
| **Visages** | 5,171 |
| **Annotations** | Ellipses |

**Source :** [FDDB](http://vis-www.cs.umass.edu/fddb/)

---

### UTKFace

Dataset avec annotations démographiques pour tester la robustesse.

| Caractéristique | Valeur |
|-----------------|--------|
| **Images** | 20,000+ |
| **Résolution** | 200×200 |
| **Annotations** | Âge, genre, ethnie |

**Source :** [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

---

## Datasets Difficiles (Hard)

Ces datasets contiennent des cas challenging pour évaluer la robustesse des détecteurs.

### DARK FACE

| Caractéristique | Valeur |
|-----------------|--------|
| **Images** | 6,000 |
| **Difficultés** | Low-light, nuit, ombres |
| **Taille** | ~2GB |

**Cas couverts :**
- Scènes nocturnes
- Sous-exposition
- Ombres fortes
- Contre-jour

**Source :** [DARK FACE](https://flyywh.github.io/CVPRW2019LowLight/)

---

### MAFA (Masked Faces)

| Caractéristique | Valeur |
|-----------------|--------|
| **Images** | 30,811 |
| **Difficultés** | Masques, occlusions, accessoires |
| **Taille** | ~4GB |

**Cas couverts :**
- Masques médicaux
- Lunettes de soleil
- Écharpes, foulards

**Source :** [MAFA](https://imsg.ac.cn/research/maskedface.html)

---

### UFDD (Unconstrained Face Detection)

| Caractéristique | Valeur |
|-----------------|--------|
| **Images** | 6,000 |
| **Difficultés** | Météo, flou, conditions adverses |

**Cas couverts :**
- Pluie, neige, brouillard
- Flou de mouvement
- Bruit

**Source :** [UFDD](https://ufdd.info/)

---

## Matrice des Difficultés

| Dataset | Low-light | Occlusion | Masques | Flou | Pose | Petits visages |
|---------|:---------:|:---------:|:-------:|:----:|:----:|:--------------:|
| WIDER FACE | - | ✓ | - | ✓ | ✓ | ✓ |
| DARK FACE | ✓✓ | - | - | - | - | - |
| MAFA | - | ✓✓ | ✓✓ | - | - | - |
| UFDD | - | - | - | ✓✓ | - | - |
| LFW | - | - | - | - | - | - |

**Légende :** ✓ = présent, ✓✓ = focus principal

---

## Usage avec le Benchmark

### Lister les datasets disponibles

```bash
python benchmark.py --list-datasets
```

### Benchmark sur WIDER FACE

```bash
# Benchmark rapide (300 images)
python benchmark.py --dataset wider_face --limit 300

# Benchmark complet
python benchmark.py --dataset wider_face
```

### Ajouter un nouveau dataset

1. Créer le loader dans `datasets/`:

```python
# datasets/fddb.py
from .base import BaseDataset

class FDDBDataset(BaseDataset):
    name = "FDDB"
    description = "Face Detection Data Set and Benchmark"

    @property
    def annotation_file(self) -> Path:
        return ANNOTATIONS_DIR / "fddb" / "annotations.txt"

    @property
    def images_dir(self) -> Path:
        return DATASETS_DIR / "fddb"

    def load_annotations(self) -> Dict[str, List[BBox]]:
        # Parser le format FDDB (ellipses → rectangles)
        pass
```

2. L'enregistrer dans `datasets/__init__.py`:

```python
DATASET_REGISTRY = {
    "wider_face": WiderFaceDataset,
    "fddb": FDDBDataset,  # Ajouter ici
}
```

---

## Références

1. **WIDER FACE** - Yang et al., "WIDER FACE: A Face Detection Benchmark", CVPR 2016
2. **LFW** - Huang et al., "Labeled Faces in the Wild", 2007
3. **FDDB** - Jain & Learned-Miller, "FDDB: A Benchmark for Face Detection in Unconstrained Settings", 2010
4. **DARK FACE** - Yuan et al., "DARK FACE Dataset", CVPRW 2019
5. **MAFA** - Ge et al., "Detecting Masked Faces in the Wild", CVPR 2017
6. **UFDD** - Nada et al., "Pushing the Limits of Unconstrained Face Detection", 2018
