# Guide des Modeles de Detection d'Emotions

Ce document recense les modeles disponibles pour la detection d'emotions faciales, leurs datasets d'entrainement, et leurs caracteristiques.

---

## Table des Matieres

1. [Datasets de Reference](#datasets-de-reference)
2. [Modeles Disponibles](#modeles-disponibles)
3. [Comparatif des Modeles](#comparatif-des-modeles)
4. [Recommandations](#recommandations)
5. [Installation et Usage](#installation-et-usage)

---

## Datasets de Reference

### FER2013 (Facial Expression Recognition 2013)

| Caracteristique | Valeur |
|-----------------|--------|
| **Source** | Kaggle Competition 2013 |
| **Taille** | 35,887 images |
| **Resolution** | 48x48 pixels (grayscale) |
| **Classes** | 7 emotions |
| **Split** | Train: 28,709 / Test: 7,178 |

**Distribution des classes :**
```
happy:    8,989 (25.0%)  ████████████████████████
neutral:  6,198 (17.3%)  █████████████████
sad:      6,077 (16.9%)  ████████████████
fear:     5,121 (14.3%)  ██████████████
angry:    4,953 (13.8%)  █████████████
surprise: 4,002 (11.2%)  ███████████
disgust:    547 (1.5%)   █
```

**Problemes connus :**
- Dataset TRES desequilibre (disgust sous-represente)
- Images de faible resolution (48x48)
- Annotations parfois incorrectes (~10% d'erreurs)
- Biais vers "neutral" et "happy"

---

### AffectNet

| Caracteristique | Valeur |
|-----------------|--------|
| **Source** | Mohammad et al., 2017 |
| **Taille** | ~1,000,000 images |
| **Resolution** | Variable (haute qualite) |
| **Classes** | 8 emotions + valence/arousal |
| **Qualite** | Annotations manuelles |

**Distribution des classes :**
```
happy:      134,415 (59.3%)
neutral:    74,874 (33.0%)
sad:        25,459 (11.2%)
anger:      24,882 (11.0%)
surprise:   14,090 (6.2%)
fear:       6,378 (2.8%)
disgust:    3,803 (1.7%)
contempt:   3,750 (1.7%)
```

**Avantages :**
- Dataset le plus grand et le plus diversifie
- Annotations de haute qualite
- Images en conditions reelles (in-the-wild)
- Inclut valence/arousal pour emotions continues

**Utilise par :** HSEmotion, certains backends DeepFace

---

### RAF-DB (Real-world Affective Faces Database)

| Caracteristique | Valeur |
|-----------------|--------|
| **Source** | Li et al., 2017 |
| **Taille** | 29,672 images |
| **Resolution** | Variable |
| **Classes** | 7 emotions de base + 12 composees |
| **Qualite** | ~40 annotateurs par image |

**Avantages :**
- Annotations tres fiables (consensus)
- Images en conditions reelles
- Diversite ethnique

---

### CK+ (Extended Cohn-Kanade)

| Caracteristique | Valeur |
|-----------------|--------|
| **Source** | Lucey et al., 2010 |
| **Taille** | 593 sequences video |
| **Resolution** | 640x490 pixels |
| **Classes** | 7 emotions |
| **Qualite** | Laboratoire (controle) |

**Problemes :**
- Petit dataset
- Conditions artificielles (studio)
- Ne generalise pas bien aux conditions reelles

---

## Modeles Disponibles

### 1. FER (Facial Emotion Recognition) - ACTUEL

```python
from fer import FER
detector = FER(mtcnn=True)
result = detector.detect_emotions(image)
```

| Caracteristique | Valeur |
|-----------------|--------|
| **Bibliotheque** | `fer` |
| **Backend** | TensorFlow/Keras |
| **Dataset** | FER2013 |
| **Architecture** | CNN custom + MTCNN |
| **Input** | Image BGR, toute taille |
| **Output** | `[{box, emotions: {angry, disgust, fear, happy, sad, surprise, neutral}}]` |
| **Vitesse** | ~100ms/frame |
| **Taille** | ~500MB |

**Forces :**
- Facile a utiliser (plug & play)
- Detection de visage integree (MTCNN)
- Supporte plusieurs visages

**Faiblesses :**
- Entraine sur FER2013 (biais neutral/happy)
- Precision moyenne (~65% sur FER2013)
- Mal detecte "angry" et "disgust"

---

### 2. DeepFace

```python
from deepface import DeepFace
result = DeepFace.analyze(img, actions=['emotion'])
# Output: {dominant_emotion, emotion: {angry, disgust, fear, happy, sad, surprise, neutral}, region}
```

| Caracteristique | Valeur |
|-----------------|--------|
| **Bibliotheque** | `deepface` |
| **Backend** | TensorFlow |
| **Datasets** | FER2013, AffectNet (selon backend) |
| **Architectures** | VGG-Face, Facenet, OpenFace, DeepID, ArcFace |
| **Input** | Image BGR ou chemin fichier |
| **Output** | `{dominant_emotion, emotion: {...}, region: {x,y,w,h}}` |
| **Vitesse** | ~200-500ms/frame |
| **Taille** | ~1-2GB (modeles telecharges au premier run) |

**Backends disponibles :**
| Backend | Precision | Vitesse | Utilisation |
|---------|-----------|---------|-------------|
| VGG-Face | Bonne | Lent | Par defaut |
| Facenet | Tres bonne | Moyen | Recommande |
| OpenFace | Moyenne | Rapide | Temps reel |
| DeepID | Bonne | Moyen | - |
| ArcFace | Tres bonne | Lent | Haute precision |

**Forces :**
- Plusieurs backends a tester
- Fonctions supplementaires (age, genre, race)
- Bien documente

**Faiblesses :**
- Plus lent que les autres
- Gros telechargements au premier lancement
- Parfois instable

---

### 3. HSEmotion (RECOMMANDE)

```python
from hsemotion.facial_emotions import HSEmotionRecognizer
model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
emotion, scores = model.predict_emotions(face_img)
```

| Caracteristique | Valeur |
|-----------------|--------|
| **Bibliotheque** | `hsemotion` |
| **Backend** | PyTorch |
| **Dataset** | AffectNet (meilleur dataset!) |
| **Architecture** | EfficientNet-B0/B2 |
| **Input** | Image RGB, visage croppe 224x224 |
| **Output** | `(emotion_label, scores_array)` |
| **Vitesse** | ~50ms/frame |
| **Taille** | ~200MB |

**Modeles disponibles :**
| Modele | Dataset | Classes | Precision |
|--------|---------|---------|-----------|
| `enet_b0_8_best_afew` | AffectNet | 8 | ~63% |
| `enet_b0_8_va_mtl` | AffectNet | 8 + VA | ~65% |
| `enet_b2_8` | AffectNet | 8 | ~66% |

**Classes (8) :** anger, contempt, disgust, fear, happiness, neutral, sadness, surprise

**Forces :**
- Entraine sur AffectNet (meilleur dataset)
- Tres rapide (EfficientNet optimise)
- Bonne precision sur emotions subtiles
- 8 classes (inclut contempt)

**Faiblesses :**
- Necessite de cropper le visage avant
- Pas de detection de visage integree

---

### 4. facial-emotion-recognition

```python
from facial_emotion_recognition import EmotionRecognition
er = EmotionRecognition(device='cpu')
result = er.recognise_emotion(frame, return_type='list')
# Output: [(box, emotion_label), ...]
```

| Caracteristique | Valeur |
|-----------------|--------|
| **Bibliotheque** | `facial-emotion-recognition` |
| **Backend** | PyTorch |
| **Dataset** | FER2013 |
| **Architecture** | ResNet + face detection |
| **Input** | Image BGR, toute taille |
| **Output** | `[(box, emotion_label), ...]` |
| **Vitesse** | ~80ms/frame |
| **Taille** | ~300MB |

**Forces :**
- Detection de visage integree
- Simple a utiliser
- Support GPU/CPU

**Faiblesses :**
- Entraine sur FER2013 (memes biais)
- Moins precis que HSEmotion

---

### 5. Custom Model (votre training)

```python
from training.model import load_model
model = load_model('models/efficientnet_b0_fer.pth', num_classes=7)
output = model(preprocessed_image)  # Logits [7]
probs = torch.softmax(output, dim=1)
```

| Caracteristique | Valeur |
|-----------------|--------|
| **Bibliotheque** | PyTorch (votre code) |
| **Backend** | PyTorch |
| **Dataset** | FER2013 (ou custom) |
| **Architectures** | EfficientNet-B0, MobileNetV3-Small/Large |
| **Input** | Tensor RGB 224x224 normalise |
| **Output** | Logits `[batch, 7]` |
| **Vitesse** | ~30-50ms/frame |
| **Taille** | ~20-50MB |

**Avantages :**
- Controle total sur l'entrainement
- Peut fine-tuner sur vos donnees
- Optimise pour votre cas d'usage

---

## Comparatif des Modeles

### Precision par emotion (estimee)

| Emotion | FER | DeepFace | HSEmotion | Custom |
|---------|-----|----------|-----------|--------|
| **happy** | 85% | 88% | 90% | ~80% |
| **neutral** | 70% | 75% | 80% | ~70% |
| **sad** | 55% | 65% | 75% | ~60% |
| **angry** | 50% | 60% | 70% | ~55% |
| **surprise** | 75% | 80% | 85% | ~75% |
| **fear** | 45% | 55% | 65% | ~50% |
| **disgust** | 40% | 50% | 60% | ~45% |

### Tableau comparatif global

| Critere | FER | DeepFace | HSEmotion | Custom |
|---------|-----|----------|-----------|--------|
| **Precision globale** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Vitesse** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Facilite d'usage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Detection visage** | Integree | Integree | Manuelle | Manuelle |
| **Taille** | 500MB | 1-2GB | 200MB | 20-50MB |
| **Angry detection** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Matrice Modeles x Datasets

| Modele | FER2013 | AffectNet | RAF-DB | CK+ |
|--------|:-------:|:---------:|:------:|:---:|
| **FER** | ✅ Entraine | ❌ | ❌ | ❌ |
| **DeepFace** | ✅ Entraine | ⚡ Partiel | ❌ | ❌ |
| **HSEmotion** | ❌ | ✅ Entraine | ❌ | ❌ |
| **facial-emotion-recognition** | ✅ Entraine | ❌ | ❌ | ❌ |
| **Custom (votre training)** | ✅ Disponible | ❌ A obtenir | ❌ | ❌ |

**Legende :** ✅ = Entraine sur ce dataset | ⚡ = Support partiel | ❌ = Non supporte

---

### Impact du Dataset sur la Precision

| Dataset | Resolution | Qualite annotations | Diversite | Impact sur precision |
|---------|------------|---------------------|-----------|----------------------|
| **FER2013** | 48x48 (faible) | Moyenne (~90%) | Faible | Biais neutral/happy, mal sur angry |
| **AffectNet** | Variable (haute) | Excellente (>95%) | Excellente | Meilleure detection emotions negatives |
| **RAF-DB** | Variable | Tres bonne | Bonne | Bon equilibre |
| **CK+** | 640x490 | Excellente | Faible (studio) | Surgeneralise en conditions reelles |

---

### Tableau Croise : Performance Estimee par Combinaison

| | angry | disgust | fear | happy | sad | surprise | neutral |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FER + FER2013** | 50% | 40% | 45% | 85% | 55% | 75% | 70% |
| **DeepFace + FER2013** | 60% | 50% | 55% | 88% | 65% | 80% | 75% |
| **HSEmotion + AffectNet** | 70% | 60% | 65% | 90% | 75% | 85% | 80% |
| **Custom + FER2013** | 55% | 45% | 50% | 80% | 60% | 75% | 70% |
| **Custom + AffectNet** | 68% | 58% | 63% | 88% | 73% | 83% | 78% |

---

### Pourquoi HSEmotion est meilleur ?

1. **Dataset AffectNet vs FER2013**
   - AffectNet: 1M images, haute resolution, annotations pro
   - FER2013: 35K images, 48x48 pixels, annotations crowdsourcees

2. **Architecture EfficientNet**
   - Plus moderne que les CNN classiques
   - Meilleur ratio precision/vitesse

3. **8 classes au lieu de 7**
   - Inclut "contempt" (mepris)
   - Meilleure granularite des emotions

---

## Recommandations

### Pour votre cas (angry mal detecte)

| Priorite | Modele | Raison |
|----------|--------|--------|
| 1 | **HSEmotion** | Entraine sur AffectNet, meilleure detection angry |
| 2 | **DeepFace (Facenet)** | Plus robuste, plusieurs backends |
| 3 | **Custom training** | Controle total, peut augmenter angry |

### Selon le cas d'usage

| Cas d'usage | Modele recommande |
|-------------|-------------------|
| Prototype rapide | FER |
| Production (precision) | HSEmotion |
| Multi-fonctions (age, genre) | DeepFace |
| Temps reel strict (<30ms) | Custom MobileNetV3 |
| Controle total | Custom training |

---

## Installation et Usage

### Installation rapide

```bash
# Creer environnement virtuel
cd backend
python3 -m venv venv
source venv/bin/activate

# Installer les modeles
pip install fer                          # FER (actuel)
pip install deepface                     # DeepFace
pip install hsemotion                    # HSEmotion (recommande)
pip install facial-emotion-recognition   # Alternative PyTorch
```

### Exemple d'integration HSEmotion

```python
import cv2
from hsemotion.facial_emotions import HSEmotionRecognizer
from fer import FER  # Pour detection de visage

# Initialiser
face_detector = FER(mtcnn=True)
emotion_model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

# Charger image
img = cv2.imread('image.jpg')

# Detecter visages avec FER/MTCNN
faces = face_detector.detect_emotions(img)

for face in faces:
    x, y, w, h = face['box']

    # Cropper et redimensionner le visage
    face_crop = img[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (224, 224))
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # Predire avec HSEmotion
    emotion, scores = emotion_model.predict_emotions(face_rgb)
    print(f"Emotion: {emotion}, Scores: {scores}")
```

### Exemple DeepFace

```python
from deepface import DeepFace

result = DeepFace.analyze(
    img_path="image.jpg",
    actions=['emotion'],
    detector_backend='mtcnn'
)

print(f"Emotion: {result[0]['dominant_emotion']}")
print(f"Scores: {result[0]['emotion']}")
```

---

## Ressources

### Papers
- FER2013: "Challenges in Representation Learning" (Goodfellow et al., 2013)
- AffectNet: "AffectNet: A Database for Facial Expression" (Mollahosseini et al., 2017)
- EfficientNet: "EfficientNet: Rethinking Model Scaling" (Tan & Le, 2019)

### Liens
- [FER2013 Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [AffectNet](http://mohammadmahoor.com/affectnet/)
- [HSEmotion GitHub](https://github.com/HSE-asavchenko/face-emotion-recognition)
- [DeepFace GitHub](https://github.com/serengil/deepface)

---

*Document genere pour ProjectCare - Janvier 2025*
