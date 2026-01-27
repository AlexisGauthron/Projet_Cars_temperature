# Modèles de Détection de Visage

Ce document détaille tous les modèles de détection de visage supportés par le benchmark.

## Vue d'ensemble

| Modèle | Architecture | Taille | Vitesse | WIDER FACE AP (Hard) | GPU requis | Année |
|--------|--------------|--------|---------|----------------------|------------|-------|
| **Légers / Embarqué** |
| YuNet | CNN léger | 230 KB | ~1 ms | 77.9% | Non | 2023 |
| Haar Cascade | Cascade Boosting | 1 MB | ~0.3 ms | ~50% | Non | 2001 |
| MediaPipe | BlazeFace | 1 MB | ~5 ms | ~75% | Non | 2019 |
| **Équilibrés** |
| OpenCV-DNN | SSD + ResNet-10 | 10 MB | ~35 ms | ~70% | Non | 2017 |
| DLib-HOG | HOG + SVM | 0 KB | ~50 ms | ~60% | Non | 2009 |
| **YOLO Family** |
| YOLO5Face | YOLOv5 + landmarks | 7 MB | ~15 ms | **86.6%** | Optionnel | 2021 |
| YOLOv8-face | YOLOv8 nano | 6 MB | ~30 ms | ~80% | Optionnel | 2023 |
| YOLOv10-face | YOLOv10 NMS-free | 6 MB | ~20 ms | ~82% | Optionnel | 2024 |
| YOLOv11-face | YOLOv11 nano | 6 MB | ~25 ms | 81.0% | Optionnel | 2024 |
| YOLOv12-face | YOLOv12 attention | 6 MB | ~25 ms | ~83% | Optionnel | 2026 |
| **Haute Précision** |
| SCRFD_500M | Anchor-free | 2 MB | ~10 ms | 68.5% | Non | 2022 |
| SCRFD_2.5G | Anchor-free | 3 MB | ~15 ms | 77.9% | Optionnel | 2022 |
| SCRFD_10G | Anchor-free | 16 MB | ~25 ms | 83.0% | Optionnel | 2022 |
| SCRFD_34G | Anchor-free | 68 MB | ~50 ms | **85.3%** | Recommandé | 2022 |
| RetinaFace | RetinaNet + ResNet-50 | 30 MB | ~1100 ms | 84.1% | Recommandé | 2019 |
| MTCNN | Cascade CNN (3 stages) | 2 MB | ~1000 ms | ~80% | Optionnel | 2016 |
| DLib-CNN | MMOD CNN | 100 MB | ~500 ms | ~75% | Recommandé | 2017 |

---

## Modèles Intégrés

### YuNet

**Architecture :** CNN léger optimisé pour les appareils embarqués

| Caractéristique | Valeur |
|-----------------|--------|
| **Fichier** | `face_detection_yunet_2023mar.onnx` |
| **Taille** | 230 KB |
| **Format** | ONNX |
| **Entrée** | RGB, dimensions variables |
| **Framework** | OpenCV (cv2.FaceDetectorYN) |

**Dataset d'entraînement :**
- WIDER FACE (training set)
- Augmentation avec variations de pose, illumination, occlusion

**Spécificités :**
- Robuste aux occlusions partielles (masques, lunettes)
- Détection multi-échelle intégrée
- Sortie : bounding box + 5 landmarks + score de confiance
- Optimisé pour les processeurs ARM (mobile, embarqué)

**Publication :** Yu et al., "YuNet: A Tiny Millisecond-level Face Detector", Machine Intelligence Research, 2023

**Source :** [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)

**Installation :**
```bash
python scripts/download_models.py --model yunet
```

---

### OpenCV-DNN (SSD ResNet-10)

**Architecture :** Single Shot Detector avec backbone ResNet-10

| Caractéristique | Valeur |
|-----------------|--------|
| **Fichiers** | `deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel` |
| **Taille** | ~10 MB |
| **Format** | Caffe |
| **Entrée** | 300×300 RGB |
| **Framework** | OpenCV DNN (cv2.dnn) |

**Dataset d'entraînement :**
- WIDER FACE
- CelebA
- Images web diverses

**Spécificités :**
- Meilleur compromis précision/vitesse pour CPU
- Très stable et robuste
- Intégré nativement dans OpenCV
- Pas de dépendances externes

**Publication :** Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016

**Source :** [OpenCV Samples](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

**Installation :**
```bash
python scripts/download_models.py --model opencv_dnn
```

---

### Haar Cascade

**Architecture :** Cascade de classificateurs avec features de Haar

| Caractéristique | Valeur |
|-----------------|--------|
| **Fichier** | `haarcascade_frontalface_default.xml` |
| **Taille** | ~1 MB |
| **Format** | XML |
| **Entrée** | Grayscale |
| **Framework** | OpenCV (cv2.CascadeClassifier) |

**Dataset d'entraînement :**
- Propriétaire (non publié)
- Images de visages frontaux

**Spécificités :**
- Ultra-rapide (~0.3 ms)
- Précision limitée (visages frontaux uniquement)
- Sensible aux variations de pose
- Pas de GPU nécessaire

**Publication :** Viola & Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features", CVPR 2001

**Source :** [OpenCV Haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)

**Installation :**
```bash
python scripts/download_models.py --model haar
```

---

### MTCNN (Multi-task Cascaded CNN)

**Architecture :** Cascade de 3 réseaux CNN (P-Net, R-Net, O-Net)

| Caractéristique | Valeur |
|-----------------|--------|
| **Taille** | ~2 MB |
| **Format** | TensorFlow / PyTorch |
| **Entrée** | RGB, dimensions variables |
| **Framework** | mtcnn ou facenet-pytorch |

**Dataset d'entraînement :**
- WIDER FACE
- CelebA
- AFLW (Annotated Facial Landmarks in the Wild)

**Spécificités :**
- Détection + alignement en une passe
- Sortie : bounding box + 5 landmarks
- 3 étapes : proposition → raffinement → sortie
- Standard académique très utilisé

**Publication :** Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks", SPL 2016

**Source :**
- [mtcnn](https://github.com/ipazc/mtcnn) (TensorFlow)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (PyTorch)

**Installation :**
```bash
pip install mtcnn
# ou
pip install facenet-pytorch
```

---

### RetinaFace

**Architecture :** RetinaNet avec backbone ResNet-50 + FPN

| Caractéristique | Valeur |
|-----------------|--------|
| **Taille** | ~30 MB |
| **Format** | ONNX / PyTorch |
| **Entrée** | RGB, dimensions variables |
| **Framework** | retinaface / insightface |

**Dataset d'entraînement :**
- WIDER FACE (training set complet)
- Annotations supplémentaires de landmarks

**Spécificités :**
- Haute précision (state-of-the-art 2019)
- Détection dense (petits visages)
- Sortie : bounding box + 5 landmarks + qualité
- GPU fortement recommandé

**Publication :** Deng et al., "RetinaFace: Single-stage Dense Face Localisation in the Wild", CVPR 2020

**Source :**
- [retinaface](https://github.com/serengil/retinaface)
- [InsightFace](https://github.com/deepinsight/insightface)

**Installation :**
```bash
pip install retinaface
# ou
pip install insightface
```

---

### SCRFD (Sample and Computation Redistribution)

**Architecture :** Anchor-free detector avec redistribution de calculs

**5 variantes disponibles :**

| Variante | GFLOPs | Taille | Easy | Medium | Hard | Cas d'usage |
|----------|--------|--------|------|--------|------|-------------|
| **SCRFD_500M** | 0.5 | 2 MB | 90.57% | 88.12% | 68.51% | Mobile/Embarqué |
| **SCRFD_2.5G** | 2.5 | 3 MB | 93.78% | 92.16% | 77.87% | Desktop CPU |
| **SCRFD_10G** | 10 | 16 MB | 95.16% | 93.87% | 83.05% | Desktop GPU |
| **SCRFD_34G** | 34 | 68 MB | **96.06%** | **94.92%** | **85.29%** | Serveur/Cloud |
| **SCRFD** | - | 5 MB | - | - | - | Défaut (buffalo_sc) |

**Dataset d'entraînement :**
- WIDER FACE (complet)
- Données propriétaires InsightFace

**Spécificités :**
- **State-of-the-art ICLR 2022**
- Anchor-free avec redistribution des calculs
- Meilleur compromis précision/vitesse
- Optimisé pour la production
- Support GPU et CPU

**Publication :** Guo et al., "Sample and Computation Redistribution for Efficient Face Detection", ICLR 2022

**Source :** [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)

**Installation :**
```bash
pip install insightface onnxruntime
# Télécharger les variantes spécifiques
python scripts/download_models.py --model scrfd_10g
python scripts/download_models.py --model scrfd_34g
```

---

### YOLOv8-face

**Architecture :** YOLOv8 nano fine-tuné pour la détection de visages

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | YOLOv8n-face |
| **Taille** | ~6 MB |
| **Format** | PyTorch (.pt) |
| **Entrée** | 640×640 RGB |
| **Framework** | Ultralytics |

**Dataset d'entraînement :**
- WIDER FACE
- Annotations converties au format YOLO

**Spécificités :**
- Architecture YOLO moderne
- Inference rapide sur GPU et CPU
- Export ONNX, TensorRT, CoreML possible
- Intégration facile avec Ultralytics

**Publication :** Jocher et al., "Ultralytics YOLOv8", 2023

**Source :** [HuggingFace - arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)

**Installation :**
```bash
pip install ultralytics huggingface_hub
```

---

### YOLOv11-face

**Architecture :** YOLOv11 nano fine-tuné pour la détection de visages

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | YOLOv11n-face |
| **Taille** | ~6 MB |
| **Format** | PyTorch (.pt) |
| **Entrée** | 640×640 RGB |
| **Framework** | Ultralytics |

**Dataset d'entraînement :**
- WIDER FACE (complet)
- Fine-tuning avec augmentations modernes

**Spécificités :**
- Dernier modèle YOLO (2024)
- Architecture C3k2 améliorée
- Meilleure généralisation
- Support natif Ultralytics

**Performance WIDER FACE :**

| Difficulté | AP |
|------------|-----|
| Easy | 94.2% |
| Medium | 92.1% |
| Hard | 81.0% |

**Source :** [HuggingFace - AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection)

**Installation :**
```bash
pip install ultralytics huggingface_hub
```

---

### YOLOv10-face

**Architecture :** YOLOv10 NMS-free fine-tuné pour la détection de visages

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | YOLOv10n-face |
| **Taille** | ~6 MB |
| **Format** | PyTorch (.pt) |
| **Entrée** | 640×640 RGB |
| **Framework** | Ultralytics |

**Dataset d'entraînement :**
- WIDER FACE (complet)
- Fine-tuning avec augmentations modernes

**Spécificités :**
- **NMS-free** : Élimination du Non-Maximum Suppression via dual label assignment
- Latence end-to-end réduite
- Architecture optimisée NeurIPS 2024
- Meilleure vitesse d'inférence

**Publication :** Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", NeurIPS 2024

**Source :** [HuggingFace - akanametov/yolov10-face](https://huggingface.co/akanametov/yolov10-face)

**Installation :**
```bash
pip install ultralytics huggingface_hub
```

---

### YOLOv12-face

**Architecture :** YOLOv12 attention-centric fine-tuné pour la détection de visages

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | YOLOv12n-face |
| **Taille** | ~6 MB |
| **Format** | PyTorch (.pt) |
| **Entrée** | 640×640 RGB |
| **Framework** | Ultralytics |

**Dataset d'entraînement :**
- WIDER FACE
- Fine-tuning avec mécanismes d'attention

**Spécificités :**
- **Dernier YOLO** (2026)
- Architecture attention-centric
- Meilleure modélisation des dépendances globales
- Performance améliorée sur les visages occultés

**Source :** [HuggingFace - akanametov/yolov12-face](https://huggingface.co/akanametov/yolov12-face)

**Installation :**
```bash
pip install ultralytics huggingface_hub
```

---

### YOLO5Face

**Architecture :** YOLOv5 modifié avec détection de landmarks facials

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | YOLOv5n-face |
| **Taille** | ~7 MB |
| **Format** | PyTorch (.pt) |
| **Entrée** | 640×640 RGB |
| **Framework** | Ultralytics |

**Dataset d'entraînement :**
- WIDER FACE (complet)
- Annotations de landmarks 5 points

**Spécificités :**
- **State-of-the-art sur WIDER FACE** (2021)
- Détection simultanée bbox + 5 landmarks
- Optimisé pour la détection de visages

**Performance WIDER FACE :**

| Difficulté | AP |
|------------|-----|
| Easy | **96.67%** |
| Medium | **95.08%** |
| Hard | **86.55%** |

**Publication :** Qi et al., "YOLO5Face: Why Reinventing a Face Detector", arXiv:2105.12931

**Source :** [HuggingFace - akanametov/yolo5-face](https://huggingface.co/akanametov/yolo5-face)

**Installation :**
```bash
pip install ultralytics huggingface_hub
```

---

### MediaPipe (BlazeFace)

**Architecture :** BlazeFace - CNN léger optimisé mobile

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | BlazeFace short/full range |
| **Taille** | ~1 MB |
| **Format** | TFLite |
| **Entrée** | 128×128 / 192×192 RGB |
| **Framework** | MediaPipe |

**Dataset d'entraînement :**
- Données propriétaires Google
- Optimisé pour mobile et webcam

**Spécificités :**
- Ultra-rapide sur mobile (~5 ms)
- Deux modèles : short-range (2m) et full-range (5m)
- Intégré dans l'écosystème Google
- Landmarks 6 points

**Publication :** Bazarevsky et al., "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs", CVPR Workshop 2019

**Source :** [MediaPipe Face Detection](https://developers.google.com/mediapipe/solutions/vision/face_detector)

**Installation :**
```bash
pip install mediapipe
```

---

### DLib-HOG

**Architecture :** HOG (Histogram of Oriented Gradients) + Linear SVM

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | Frontal face detector |
| **Taille** | 0 KB (intégré) |
| **Format** | N/A |
| **Entrée** | Grayscale |
| **Framework** | DLib |

**Dataset d'entraînement :**
- Données propriétaires DLib
- Visages frontaux

**Spécificités :**
- Classique et robuste
- Pas de deep learning
- Fonctionne sur CPU uniquement
- Intégré dans DLib (pas de fichier à télécharger)

**Publication :** King, "Dlib-ml: A Machine Learning Toolkit", JMLR 2009

**Source :** [DLib](http://dlib.net/)

**Installation :**
```bash
pip install dlib
```

---

### DLib-CNN

**Architecture :** MMOD CNN (Max-Margin Object Detection)

| Caractéristique | Valeur |
|-----------------|--------|
| **Modèle** | mmod_human_face_detector.dat |
| **Taille** | ~100 MB |
| **Format** | DAT |
| **Entrée** | RGB |
| **Framework** | DLib |

**Dataset d'entraînement :**
- WIDER FACE
- Données propriétaires DLib

**Spécificités :**
- Plus précis que HOG
- GPU recommandé (lent sur CPU)
- Détecte mieux les profils et occlusions
- Score de confiance disponible

**Publication :** King, "Max-Margin Object Detection", arXiv:1502.00046

**Source :** [DLib Models](http://dlib.net/files/)

**Installation :**
```bash
pip install dlib
python scripts/download_models.py --model dlib
```

---

## Comparaison des Modèles

### Par Cas d'Usage

| Cas d'usage | Modèle recommandé | Alternative |
|-------------|-------------------|-------------|
| **Embarqué / IoT** | YuNet | MediaPipe |
| **Mobile** | MediaPipe | YuNet |
| **Desktop CPU** | YOLO5Face | OpenCV-DNN |
| **Desktop GPU** | SCRFD | YOLOv12-face |
| **Serveur / Cloud** | SCRFD 10g | RetinaFace |
| **Temps réel strict** | Haar | YuNet |
| **Haute précision** | SCRFD | YOLO5Face |
| **Webcam temps réel** | YOLOv10-face | YOLOv11-face |

### Par Critère Technique

| Critère | Meilleur modèle |
|---------|-----------------|
| **Vitesse CPU** | Haar (0.3ms), YuNet (1ms), MediaPipe (5ms) |
| **Vitesse GPU** | SCRFD, YOLO5Face, YOLOv10-face |
| **Précision globale (Hard)** | YOLO5Face (86.6%), SCRFD (85.2%) |
| **Petits visages** | RetinaFace, SCRFD |
| **Occlusions** | YuNet, YOLOv12-face |
| **Taille modèle** | YuNet (230KB), MediaPipe (1MB) |
| **Facilité d'intégration** | OpenCV-DNN, YuNet, MediaPipe |
| **Landmarks intégrés** | YOLO5Face, MediaPipe, MTCNN |
| **NMS-free** | YOLOv10-face |

---

## Téléchargement

### Tous les modèles

```bash
python scripts/download_models.py
```

### Modèle spécifique

```bash
python scripts/download_models.py --model yunet
python scripts/download_models.py --model opencv_dnn
python scripts/download_models.py --model scrfd
python scripts/download_models.py --model yolov8_face
```

### Lister les modèles

```bash
python scripts/download_models.py --list
python benchmark.py --list-models
```

---

## Références

1. **YuNet** - Yu et al., "YuNet: A Tiny Millisecond-level Face Detector", Machine Intelligence Research, 2023
2. **SSD** - Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016
3. **Haar** - Viola & Jones, "Rapid Object Detection using a Boosted Cascade", CVPR 2001
4. **MTCNN** - Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded CNN", SPL 2016
5. **RetinaFace** - Deng et al., "RetinaFace: Single-stage Dense Face Localisation", CVPR 2020
6. **SCRFD** - Guo et al., "Sample and Computation Redistribution for Efficient Face Detection", ICLR 2022
7. **YOLOv8** - Jocher et al., "Ultralytics YOLOv8", 2023
8. **YOLO5Face** - Qi et al., "YOLO5Face: Why Reinventing a Face Detector", arXiv:2105.12931, 2021
9. **YOLOv10** - Wang et al., "YOLOv10: Real-Time End-to-End Object Detection", NeurIPS 2024
10. **BlazeFace** - Bazarevsky et al., "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs", CVPR Workshop 2019
11. **DLib** - King, "Dlib-ml: A Machine Learning Toolkit", JMLR 2009
12. **MMOD** - King, "Max-Margin Object Detection", arXiv:1502.00046, 2015
