# État de l'Art - Benchmarks de Reconnaissance d'Émotions Faciales

> Documentation mise à jour : Janvier 2025

## Table des Matières

1. [Vue d'ensemble des Datasets](#vue-densemble-des-datasets)
2. [FER2013 Benchmark](#fer2013-benchmark)
3. [AffectNet Benchmark](#affectnet-benchmark)
4. [RAF-DB Benchmark](#raf-db-benchmark)
5. [CK+ Benchmark](#ck-benchmark)
6. [Comparaison Multi-Dataset](#comparaison-multi-dataset)
7. [Modèles State-of-the-Art](#modèles-state-of-the-art)
8. [Références](#références)

---

## Vue d'ensemble des Datasets

| Dataset | Images | Classes | Résolution | Type | Année |
|---------|--------|---------|------------|------|-------|
| **FER2013** | 35,887 | 7 | 48×48 grayscale | Wild | 2013 |
| **FER+** | 35,887 | 8 | 48×48 grayscale | Wild | 2016 |
| **AffectNet** | 1,000,000+ | 8 | Variable RGB | Wild | 2017 |
| **RAF-DB** | ~30,000 | 7/12 | Variable RGB | Wild | 2017 |
| **CK+** | 593 séq. | 7 | 640×490 | Lab | 2010 |
| **ExpW** | 91,793 | 7 | Variable RGB | Wild | 2015 |
| **JAFFE** | 213 | 7 | 256×256 | Lab | 1998 |

### Classes d'émotions standard (7 classes FER2013)
- 😠 **Angry** (Colère)
- 🤢 **Disgust** (Dégoût)
- 😨 **Fear** (Peur)
- 😊 **Happy** (Joie)
- 😢 **Sad** (Tristesse)
- 😲 **Surprise**
- 😐 **Neutral** (Neutre)

---

## FER2013 Benchmark

### Informations du Dataset
- **Taille**: 35,887 images (28,709 train / 3,589 public test / 3,589 private test)
- **Résolution**: 48×48 pixels, niveaux de gris
- **Source**: Images web avec étiquetage crowdsourcé
- **Précision humaine**: ~65.5% (±5%)

### Défis du Dataset
- Forte variation intra-classe (pose, éclairage, occlusion)
- Faible séparabilité inter-classe (disgust vs. anger, fear vs. sadness)
- Déséquilibre des classes (disgust et fear sous-représentés)
- Bruit dans les labels (~10-15% d'erreurs d'annotation)

### Leaderboard FER2013 (Top Modèles)

| Rang | Modèle | Accuracy | Année | Notes |
|------|--------|----------|-------|-------|
| 1 | **Synthetic Augmentation (SD)** | **96.47%** | 2024 | +Data synthétique Stable Diffusion |
| 2 | Gabor+LBP AlexNet | 98.10%* | 2024 | *Features engineering |
| 3 | MAEL-FER | 85.78% | 2024 | Multi-Attention Enhanced |
| 4 | POSTER++ | 80.76% | 2023 | Transformer-based |
| 5 | DAN | 79.27% | 2021 | Distract Your Attention |
| 6 | EmoNeXt-XLarge | 76.12% | 2023 | ConvNeXt-based |
| 7 | Segmentation VGG-19 | 75.97% | 2023 | - |
| 8 | CNNs + BOVW | 75.42% | 2022 | Bag of Visual Words |
| 9 | EmoNeXt-Large | 75.57% | 2023 | - |
| 10 | LHC-Net | 74.42% | 2022 | - |
| 11 | **RMN** | 74.14% | 2021 | Residual Masking Network |
| 12 | EmoNeXt-Small | 74.33% | 2023 | - |
| 13 | VGGNet (optimized) | 73.28% | 2021 | State-of-the-art sans extra data |
| 14 | SE-Net50 | 72.50% | 2020 | Squeeze-and-Excitation |
| 15 | Ad-Corre | 72.03% | 2021 | - |

### Modèles Classiques (Référence)

| Modèle | Accuracy | FPS* | Notes |
|--------|----------|------|-------|
| VGG16 | 73.28% | ~15 | Baseline robuste |
| ResNet50 | 73.20% | ~20 | - |
| Inception | 71.60% | ~25 | - |
| Deep Emotion | 70.02% | ~30 | Léger |
| GoogleNet | 65.20% | ~35 | - |
| **DeepFace** | ~70-75% | **238** | Production-ready |
| **ViT-FER** | ~71-73% | ~18 | Transformer |

*FPS approximatif sur CPU

---

## AffectNet Benchmark

### Informations du Dataset
- **Taille**: 1,000,000+ images (~450,000 manuellement annotées)
- **Classes**: 8 émotions (inclut Contempt)
- **Annotations**: Valence/Arousal + catégories discrètes
- **Split**: 287,401 train / 4,000 validation

### Défis Spécifiques
- Dataset partiellement annoté automatiquement
- ~60% d'accord inter-annotateur seulement
- Grande variabilité des conditions "in-the-wild"

### Leaderboard AffectNet (8 classes)

| Rang | Modèle | Accuracy | Année |
|------|--------|----------|-------|
| 1 | Data-Centric Approach | **89.17%** | 2023 |
| 2 | MAEL-FER | 69.08% | 2024 |
| 3 | SFER-MDFAE | 67.86% | 2024 |
| 4 | EfficientNet-B2 | ~66% | 2022 |
| 5 | FCCA | 65.51% | 2023 |
| 6 | ResNet-50 baseline | ~58-60% | - |

### Résultats par Émotion (AffectNet-7)

| Émotion | Accuracy Moyenne | Difficulté |
|---------|-----------------|------------|
| Happy | ~85% | Facile |
| Surprise | ~75% | Moyen |
| Neutral | ~70% | Moyen |
| Sad | ~55% | Difficile |
| Angry | ~50% | Difficile |
| Fear | ~45% | Très difficile |
| Disgust | ~40% | Très difficile |

---

## RAF-DB Benchmark

### Informations du Dataset
- **Taille**: ~30,000 images (12,271 train / 3,068 test)
- **Classes**: 7 basiques + 12 composées
- **Annotation**: 40 annotateurs par image
- **Qualité**: Haute qualité, diversité réelle

### Leaderboard RAF-DB

| Rang | Modèle | Accuracy | Année |
|------|--------|----------|-------|
| 1 | **POSTER++** | **92.21%** | 2023 |
| 2 | POSTER | 92.05% | 2022 |
| 3 | SFER-MDFAE | 92.93% | 2024 |
| 4 | ResNet50+CBAM+TCN | 91.86% | 2024 |
| 5 | FCCA | 91.30% | 2023 |
| 6 | EAC | 90.35% | 2022 |
| 7 | MANet | ~89% | 2021 |
| 8 | VGG16 (improved) | 87.84% | 2023 |
| 9 | FARNet | 87.65% | 2023 |
| 10 | SCN | 87.03% | 2020 |
| 11 | RAN | 86.90% | 2020 |
| 12 | **CLCM (Lightweight)** | 84.00% | 2024 |

---

## CK+ Benchmark

### Informations du Dataset
- **Taille**: 593 séquences vidéo (327 avec labels émotions)
- **Sujets**: 123 personnes (18-50 ans)
- **Type**: Expressions posées, progression neutre→apex
- **Environnement**: Contrôlé (laboratoire)

### Leaderboard CK+

| Rang | Modèle | Accuracy | Notes |
|------|--------|----------|-------|
| 1 | **AA-DCN** | **99.26%** | 2024 |
| 2 | MAEL-FER | 96.98% | 2024 |
| 3 | Combined Training | 94.70% | 2024 |
| 4 | SFER-MDFAE | ~95% | 2024 |
| 5 | VGG-based | ~94% | - |
| 6 | ResNet-50 | ~93% | - |

> ⚠️ **Note**: CK+ est un dataset de laboratoire avec des expressions posées.
> Les performances élevées (~95-100%) ne se transfèrent pas aux conditions réelles.

---

## Comparaison Multi-Dataset

### Performance des Modèles Récents sur Plusieurs Datasets

| Modèle | FER2013 | AffectNet | RAF-DB | CK+ | Année |
|--------|---------|-----------|--------|-----|-------|
| **MAEL-FER** | 85.78% | 69.08% | 94.83% | 96.98% | 2024 |
| **SFER-MDFAE** | 76.18% | 67.86% | 92.93% | - | 2024 |
| **POSTER++** | 80.76% | - | 92.21% | - | 2023 |
| **DAN** | 79.27% | - | 89.70% | - | 2021 |
| **EAC** | ~78% | - | 90.35% | - | 2022 |
| **SCN** | ~75% | - | 87.03% | - | 2020 |
| **RAN** | ~74% | - | 86.90% | - | 2020 |

### Classement des Datasets par Difficulté

1. **AffectNet** (le plus difficile) - ~60-70% accuracy
2. **FER2013** - ~70-85% accuracy
3. **RAF-DB** - ~85-92% accuracy
4. **CK+** (le plus facile) - ~95-99% accuracy

---

## Modèles State-of-the-Art

### Top Architectures 2024-2025

#### 1. POSTER++ (Transformer-based)
- **Architecture**: Vision Transformer + Cross-Attention
- **Points forts**: Attention multi-échelle, robuste aux occlusions
- **FER2013**: 80.76% | **RAF-DB**: 92.21%

#### 2. MAEL-FER (Multi-Attention Enhanced)
- **Architecture**: CNN + Multi-head Attention
- **Points forts**: Meilleure généralisation cross-dataset
- **Multi-dataset**: Performances équilibrées

#### 3. EAC (Erasing Attention Consistency)
- **Architecture**: ResNet + Attention flipping
- **Points forts**: Robuste au bruit de labels
- **RAF-DB**: 90.35%

#### 4. RMN (Residual Masking Network)
- **Architecture**: CNN + Residual Masking
- **Points forts**: Bon ratio accuracy/vitesse
- **FER2013**: 74.14%

#### 5. HSEmotion (EfficientNet-based)
- **Architecture**: EfficientNet-B0/B2
- **Points forts**: Très rapide, mobile-friendly
- **Vitesse**: ~50+ FPS

### Architectures par Cas d'Usage

| Cas d'Usage | Modèle Recommandé | Accuracy | Vitesse |
|-------------|-------------------|----------|---------|
| **Production haute vitesse** | DeepFace | ~75% | ⚡⚡⚡ |
| **Mobile/Edge** | HSEmotion-ONNX | ~70% | ⚡⚡⚡ |
| **Meilleure accuracy** | POSTER++ | ~92% | ⚡ |
| **Équilibré** | ViT-FER | ~73% | ⚡⚡ |
| **Recherche** | MAEL-FER | Variable | ⚡ |

---

## Tendances et Insights

### Évolution des Performances (FER2013)

```
2013: 71.2% (Kaggle winner)
2017: 73.0% (VGG optimisé)
2020: 74.0% (RMN, SCN)
2022: 76.0% (EmoNeXt)
2023: 80.7% (POSTER++)
2024: 85.8% (MAEL-FER)
2024: 96.5% (avec data synthétique*)
```

*Avec augmentation par Stable Diffusion

### Facteurs Clés de Performance

1. **Qualité des données** > Architecture du modèle
2. **Pré-entraînement** sur grands datasets (MS-Celeb-1M, VGGFace2)
3. **Augmentation de données** (mixup, cutout, synthetic)
4. **Gestion du bruit de labels** (EAC, SCN)
5. **Attention mechanisms** (POSTER, Transformers)

### Limitations Actuelles

- **Gap Lab/Wild**: Modèles CK+ ne généralisent pas bien
- **Biais culturels**: Datasets majoritairement occidentaux
- **Émotions subtiles**: Disgust, Fear restent difficiles
- **Temps réel**: Trade-off accuracy vs latence

---

## Références

### Papers Clés

1. [Facial Emotion Recognition: State of the Art Performance on FER2013](https://arxiv.org/abs/2105.03588) - arXiv 2021
2. [POSTER++: A simpler and stronger facial expression recognition network](https://arxiv.org/abs/2301.12149) - 2023
3. [AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild](https://arxiv.org/abs/1708.03985) - IEEE 2017
4. [Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf) - CVPR 2017 (RAF-DB)

### Leaderboards

- [Papers with Code - FER2013](https://paperswithcode.com/sota/facial-expression-recognition-on-fer2013)
- [Papers with Code - AffectNet](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet)
- [Papers with Code - RAF-DB](https://paperswithcode.com/sota/facial-expression-recognition-on-raf-db)

### Ressources Supplémentaires

- [FER2013 Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- [AffectNet Official](http://mohammadmahoor.com/databases-codes/)
- [RAF-DB Official](http://www.whdeng.cn/RAF/model1.html)
- [Improved facial emotion recognition model (2024)](https://www.nature.com/articles/s41598-024-79167-8)
- [Benchmarking deep networks for FER in the wild](https://link.springer.com/article/10.1007/s11042-022-12790-7)

---

*Cette documentation est générée pour le projet ProjectCare - Benchmark Emotion Detection*
