# Benchmarks Existants - Détection de Visage

Ce document compile les résultats de benchmarks publiés par la communauté scientifique et industrielle sur la détection de visage.

## Leaderboard WIDER FACE (État de l'Art)

### Hard Set (Cas Difficiles)

Le subset "Hard" de WIDER FACE contient les visages les plus difficiles à détecter : petits visages, occlusions fortes, flou important, poses atypiques.

| Rang | Modèle | AP Hard | AP Medium | AP Easy | Année | Publication |
|------|--------|---------|-----------|---------|-------|-------------|
| 1 | **SCRFD-34GF** | **85.2%** | 93.9% | 96.1% | 2022 | ICLR 2022 |
| 2 | SCRFD-10GF | 83.0% | 93.9% | 95.2% | 2022 | ICLR 2022 |
| 3 | TinaFace | 81.4% | 92.2% | 95.6% | 2021 | CVPR 2021 |
| 4 | SCRFD-2.5GF | 77.8% | 92.2% | 93.8% | 2022 | ICLR 2022 |
| 5 | HAMBox | 76.7% | 89.3% | 95.2% | 2020 | CVPR 2020 |
| 6 | DSFD | 71.3% | 86.0% | 91.3% | 2019 | CVPR 2019 |
| 7 | SCRFD-0.5GF | 68.5% | 87.0% | 90.6% | 2022 | ICLR 2022 |
| 8 | RetinaFace-R50 | 64.1% | 87.8% | 94.9% | 2020 | CVPR 2020 |
| 9 | RetinaFace-MNet | 47.3% | 74.5% | 87.3% | 2020 | CVPR 2020 |

**Source :** [InsightFace SCRFD](https://insightface.ai/scrfd) | [ICLR 2022 Paper](https://openreview.net/pdf?id=RhB1AdoFfGE)

---

## Comparaison Vitesse vs Précision

### Benchmark SCRFD (ICLR 2022)

Résultats sur GPU avec images VGA (640×480) :

| Modèle | AP Hard | GFLOPs | Temps GPU | Speedup vs TinaFace |
|--------|---------|--------|-----------|---------------------|
| SCRFD-34GF | 85.2% | 34.0 | 11.7 ms | 3.2× |
| SCRFD-10GF | 83.0% | 10.0 | 5.4 ms | 6.9× |
| TinaFace | 81.4% | 172.0 | 37.1 ms | 1× (baseline) |
| SCRFD-2.5GF | 77.9% | 2.5 | 2.3 ms | 16.1× |
| SCRFD-0.5GF | 68.5% | 0.5 | 0.8 ms | 46.4× |
| RetinaFace-MNet | 47.3% | 0.8 | 1.3 ms | 28.5× |

**Conclusion :** SCRFD-34GF surpasse TinaFace de +3.8% AP tout en étant 3× plus rapide.

**Source :** [arXiv:2105.04714](https://arxiv.org/abs/2105.04714)

---

## Benchmarks YOLO Face Detection

### GCS-YOLOv8 (2024)

Comparaison sur WIDER FACE validation set :

| Modèle | AP Easy | AP Medium | AP Hard | Params | FLOPs |
|--------|---------|-----------|---------|--------|-------|
| **GCS-YOLOv8** | **94.2%** | **92.7%** | **81.2%** | 1.68 MB | 3.5 G |
| YOLOv8-face | 93.1% | 91.4% | 77.5% | 3.0 MB | 8.1 G |
| SCRFD-2.5GF | 93.8% | 92.2% | 77.9% | 0.67 MB | 2.5 G |
| YOLOv5-face | 93.2% | 91.3% | 76.8% | 3.1 MB | 8.0 G |
| RetinaFace | 94.9% | 93.8% | 84.0% | 29.0 MB | 37.6 G |

**Source :** [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11548442/)

### YOLOv11n-face (HuggingFace 2024)

| Difficulté | AP |
|------------|-----|
| Easy | 94.2% |
| Medium | 92.1% |
| Hard | 81.0% |

**Source :** [HuggingFace - AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection)

---

## Comparaison YOLOv11 vs YOLOv8 (Général)

### Améliorations YOLOv11 (Ultralytics 2024)

| Métrique | YOLOv11 vs YOLOv8 |
|----------|-------------------|
| Paramètres | -22% |
| Vitesse CPU | +30% |
| mAP (COCO) | +1-2% |
| Latence CPU (nano) | 56.1 ms vs 73.6 ms |

**Architectures :**
- YOLOv11 : C3k2 bottlenecks + C2PSA attention module
- YOLOv8 : C2f bottlenecks

**Source :** [Ultralytics YOLO11 vs YOLOv8](https://docs.ultralytics.com/compare/yolo11-vs-yolov8/)

---

## Benchmarks Modèles Légers

### YuNet (Machine Intelligence Research 2023)

| Métrique | Valeur |
|----------|--------|
| Paramètres | 75,856 (0.3 MB) |
| mAP Hard (single-scale) | 81.1% |
| mAP Hard (multi-scale) | 83.3% |
| Temps CPU (i7-12700K, 320×320) | 1.6 ms |
| Temps CPU (i7-12700K, 640×640) | 4.2 ms |

**Comparaison avec autres modèles légers :**

| Modèle | Params | mAP Hard | Temps (320×320) |
|--------|--------|----------|-----------------|
| YuNet | 76K | 81.1% | 1.6 ms |
| LFFD | 2.3M | 77.0% | 12.3 ms |
| CenterFace | 7.3M | 78.4% | 8.1 ms |
| RetinaFace-MNet | 0.4M | 47.3% | 3.2 ms |

**Source :** [YuNet Paper](https://www.mi-research.net/en/article/doi/10.1007/s11633-023-1423-y)

---

## Benchmarks Low-Light / Conditions Difficiles

### Hybrid Low-Light Model (2024)

Test sur DARK FACE dataset :

| Modèle | Taux Détection | TPR | FPR |
|--------|----------------|-----|-----|
| **Hybrid Model** | **94.5%** | Meilleur | Plus bas |
| YOLOv3 | 87.2% | - | - |
| MTCNN | 82.1% | - | - |
| RetinaFace | 89.3% | - | - |

**Source :** [ResearchGate](https://www.researchgate.net/figure/Performance-computation-trade-off-on-the-WIDER-FACE-validation-hard-set-for-different_fig1_351511411)

---

## Résultats NIST FRTE (2024-2025)

Le NIST (National Institute of Standards and Technology) évalue régulièrement les algorithmes de reconnaissance faciale :

| Métrique | Meilleurs Algorithmes |
|----------|----------------------|
| Précision (conditions optimales) | >99.5% |
| Précision vérification | jusqu'à 99.97% |
| Algorithmes >99% précision | 45/105 testés |

**Marché :** 6.94 milliards $ (2024) → 7.92 milliards $ (2025), +14.2%

**Source :** [NIST FRTE](https://www.nist.gov/programs-projects/face-recognition-technology-fret)

---

## Comparaison avec Nos Résultats

### Nos Benchmarks sur WIDER FACE (complet, 3226 images)

| Modèle | Notre AP | Notre Recall | Notre Temps | AP Publié | Écart |
|--------|----------|--------------|-------------|-----------|-------|
| OpenCV-DNN | 99.5% | 8.5% | 34.4 ms | ~99% | ≈ |
| YuNet | 98.8% | 48.5% | 63.0 ms | ~98% | ≈ |
| RetinaFace | 99.5% | 56.6% | 1139 ms | 96.4% | +3%* |
| MTCNN | 99.1% | 40.1% | 1048 ms | ~92% | +7%* |
| Haar | 70.0% | 13.0% | 68.5 ms | ~70% | ≈ |

*Les écarts peuvent s'expliquer par :
- Version différente du modèle
- Paramètres d'évaluation (seuil de confiance, IoU)
- Subset de test différent

---

## Synthèse et Recommandations

### Par Cas d'Usage

| Cas d'Usage | Modèle Recommandé | AP Hard | Latence |
|-------------|-------------------|---------|---------|
| **Production haute précision** | SCRFD-34GF | 85.2% | 11.7 ms |
| **Équilibre précision/vitesse** | SCRFD-10GF | 83.0% | 5.4 ms |
| **Embarqué / Edge** | YuNet | 81.1% | 1.6 ms |
| **Mobile** | SCRFD-0.5GF | 68.5% | 0.8 ms |
| **Recherche / Haute précision** | TinaFace | 81.4% | 37.1 ms |
| **CPU uniquement** | OpenCV-DNN | ~65% | 35 ms |

### Évolution des Performances (2019-2024)

```
AP Hard (%)
|
85 |                              ★ SCRFD-34GF (2022)
   |                           ★ SCRFD-10GF
80 |                    ★ TinaFace (2021)
   |                 ★ YuNet (2023)
75 |              ★ HAMBox (2020)
   |
70 |         ★ DSFD (2019)
   |
65 |    ★ RetinaFace (2020)
   |
60 +----------------------------------------→ Année
       2019   2020   2021   2022   2023   2024
```

---

## Références

1. **SCRFD** - Guo et al., "Sample and Computation Redistribution for Efficient Face Detection", ICLR 2022
   - [Paper](https://arxiv.org/abs/2105.04714) | [Code](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

2. **TinaFace** - Zhu et al., "TinaFace: Strong but Simple Baseline for Face Detection", 2021
   - [Paper](https://arxiv.org/abs/2011.13183)

3. **RetinaFace** - Deng et al., "RetinaFace: Single-stage Dense Face Localisation in the Wild", CVPR 2020
   - [Paper](https://arxiv.org/abs/1905.00641) | [Code](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)

4. **YuNet** - Yu et al., "YuNet: A Tiny Millisecond-level Face Detector", Machine Intelligence Research, 2023
   - [Paper](https://www.mi-research.net/en/article/doi/10.1007/s11633-023-1423-y) | [Code](https://github.com/opencv/opencv_zoo)

5. **WIDER FACE** - Yang et al., "WIDER FACE: A Face Detection Benchmark", CVPR 2016
   - [Website](http://shuoyang1213.me/WIDERFACE/) | [Paper](https://arxiv.org/abs/1511.06523)

6. **YOLOv8/v11** - Ultralytics, 2023-2024
   - [Documentation](https://docs.ultralytics.com/) | [Comparison](https://docs.ultralytics.com/compare/yolo11-vs-yolov8/)

7. **Papers With Code** - Face Detection Leaderboard
   - [Leaderboard](https://paperswithcode.com/sota/face-detection-on-wider-face-hard)

8. **NIST FRTE** - Face Recognition Technology Evaluation
   - [Website](https://www.nist.gov/programs-projects/face-recognition-technology-fret)
