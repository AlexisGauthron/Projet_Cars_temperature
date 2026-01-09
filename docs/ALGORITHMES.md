# Algorithmes et Étapes du POC ProjectCare

Ce document détaille les algorithmes, techniques et étapes utilisés dans le système de détection d'émotions et de contrôle climatique intelligent.

---

## Table des matières

1. [Vue d'ensemble du pipeline](#1-vue-densemble-du-pipeline)
2. [Détection de visages (MTCNN)](#2-détection-de-visages-mtcnn)
3. [Reconnaissance d'émotions (FER)](#3-reconnaissance-démotions-fer)
4. [Lissage temporel](#4-lissage-temporel)
5. [Historique des émotions](#5-historique-des-émotions)
6. [Système VLM (Questions intelligentes)](#6-système-vlm-questions-intelligentes)
7. [Contrôle de température adaptatif](#7-contrôle-de-température-adaptatif)
8. [Annotation visuelle](#8-annotation-visuelle)

---

## 1. Vue d'ensemble du pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE TRAITEMENT                            │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
    │  Capture │     │Détection │     │  Recon.  │     │ Lissage  │
    │  Frame   │────▶│ Visages  │────▶│ Émotions │────▶│ Temporel │
    │ (5 FPS)  │     │ (MTCNN)  │     │  (FER)   │     │          │
    └──────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                            │
    ┌──────────┐     ┌──────────┐     ┌──────────┐          │
    │ Ajust.   │     │ Réponse  │     │ Question │     ┌────▼─────┐
    │  Temp.   │◀────│Utilisat. │◀────│   VLM    │◀────│Historique│
    │          │     │          │     │          │     │ Émotions │
    └──────────┘     └──────────┘     └──────────┘     └──────────┘
```

### Flux de données

| Étape | Input | Output | Fréquence |
|-------|-------|--------|-----------|
| Capture | Flux caméra | Image Base64 (640x480) | 5 FPS (200ms) |
| Détection visages | Image | Bounding boxes + landmarks | Par frame |
| Reconnaissance | Région visage | 7 probabilités d'émotions | Par visage |
| Lissage | Émotion brute | Émotion stabilisée | Par visage |
| Historique | Émotion stabilisée | Buffer circulaire (15) | Par frame |
| VLM Check | Historique | Question/null | Toutes les 2s |
| Ajustement | Réponse utilisateur | Nouvelle température | Sur réponse |

---

## 2. Détection de visages (MTCNN)

### Algorithme : Multi-task Cascaded Convolutional Networks

MTCNN est un réseau de neurones en cascade composé de 3 étapes :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ARCHITECTURE MTCNN                             │
└─────────────────────────────────────────────────────────────────────────┘

    Image d'entrée
         │
         ▼
    ┌─────────┐
    │ P-Net   │  Proposal Network (12x12)
    │         │  - Génère des candidats (bounding boxes)
    │         │  - Régression rapide sur pyramide d'images
    └────┬────┘
         │ Candidats filtrés (NMS)
         ▼
    ┌─────────┐
    │ R-Net   │  Refine Network (24x24)
    │         │  - Affine les bounding boxes
    │         │  - Élimine les faux positifs
    └────┬────┘
         │ Boxes affinées (NMS)
         ▼
    ┌─────────┐
    │ O-Net   │  Output Network (48x48)
    │         │  - Détection finale précise
    │         │  - Localise 5 landmarks faciaux
    └────┬────┘
         │
         ▼
    Résultat: [x, y, w, h] + [yeux, nez, bouche]
```

### Paramètres utilisés

```python
# Configuration MTCNN dans FER
detector = FER(mtcnn=True)

# Seuils internes MTCNN
thresholds = [0.6, 0.7, 0.7]  # P-Net, R-Net, O-Net
min_face_size = 20            # Taille minimale de visage (pixels)
```

### Avantages de MTCNN

| Caractéristique | Bénéfice |
|-----------------|----------|
| Cascade | Filtrage progressif = rapidité |
| Multi-échelle | Détecte visages de toutes tailles |
| Landmarks | Permet l'alignement facial |
| Robustesse | Fonctionne avec occlusions partielles |

---

## 3. Reconnaissance d'émotions (FER)

### Architecture du modèle

FER utilise un CNN (Convolutional Neural Network) entraîné sur le dataset FER2013.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE CNN FER                              │
└─────────────────────────────────────────────────────────────────────────┘

    Visage détecté (48x48 grayscale)
              │
              ▼
    ┌─────────────────┐
    │ Conv2D (32, 3x3)│───▶ ReLU ───▶ MaxPool (2x2)
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │ Conv2D (64, 3x3)│───▶ ReLU ───▶ MaxPool (2x2)
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │Conv2D (128, 3x3)│───▶ ReLU ───▶ MaxPool (2x2)
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │    Flatten      │
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │  Dense (256)    │───▶ ReLU ───▶ Dropout (0.5)
    └─────────────────┘
              │
              ▼
    ┌─────────────────┐
    │  Dense (7)      │───▶ Softmax
    └─────────────────┘
              │
              ▼
    [angry, disgust, fear, happy, sad, surprise, neutral]
```

### Les 7 émotions détectées

| Émotion | Catégorie | Indice |
|---------|-----------|--------|
| `angry` | Inconfort | 0 |
| `disgust` | Inconfort | 1 |
| `fear` | Inconfort | 2 |
| `happy` | Confort | 3 |
| `sad` | Inconfort | 4 |
| `surprise` | Confort | 5 |
| `neutral` | Confort | 6 |

### Sortie du modèle

```python
# Exemple de sortie FER
{
    "angry": 0.02,
    "disgust": 0.01,
    "fear": 0.03,
    "happy": 0.65,    # Émotion dominante
    "sad": 0.05,
    "surprise": 0.12,
    "neutral": 0.12
}
# Confiance = 0.65 (65%)
```

---

## 4. Lissage temporel

### Problème résolu

Sans lissage, les détections sont instables :
- Micro-expressions détectées par erreur
- Bruit dans la prédiction du modèle
- Mouvements rapides du visage

### Algorithme : Vote majoritaire sur buffer circulaire

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LISSAGE TEMPOREL                                  │
└─────────────────────────────────────────────────────────────────────────┘

    Frame t-4    Frame t-3    Frame t-2    Frame t-1    Frame t
    ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐    ┌───────┐
    │ happy │    │ happy │    │neutral│    │ happy │    │ happy │
    └───────┘    └───────┘    └───────┘    └───────┘    └───────┘
         │            │            │            │            │
         └────────────┴────────────┴────────────┴────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐
                         │ Vote Majoritaire │
                         │                 │
                         │ happy: 4 votes  │
                         │ neutral: 1 vote │
                         └────────┬────────┘
                                  │
                                  ▼
                         Résultat: "happy"
```

### Implémentation

```python
class EmotionSmoother:
    def __init__(self, buffer_size=5, min_confidence=0.4):
        self.buffer_size = buffer_size
        self.min_confidence = min_confidence
        self.emotion_buffers = {}  # {face_id: deque()}

    def smooth(self, face_id: int, emotion: str, confidence: float) -> str:
        # 1. Filtrer les détections à faible confiance
        if confidence < self.min_confidence:
            return self._get_last_stable(face_id)

        # 2. Ajouter au buffer circulaire
        if face_id not in self.emotion_buffers:
            self.emotion_buffers[face_id] = deque(maxlen=self.buffer_size)

        self.emotion_buffers[face_id].append(emotion)

        # 3. Vote majoritaire
        return self._majority_vote(face_id)

    def _majority_vote(self, face_id: int) -> str:
        buffer = self.emotion_buffers[face_id]
        counts = Counter(buffer)
        return counts.most_common(1)[0][0]
```

### Paramètres configurables

| Paramètre | Valeur | Impact |
|-----------|--------|--------|
| `SMOOTHING_BUFFER_SIZE` | 5 | Plus grand = plus stable, mais plus de latence |
| `MIN_CONFIDENCE_THRESHOLD` | 0.4 | Plus haut = moins de bruit, mais plus de rejets |

### Gestion multi-visages

Chaque visage a son propre buffer identifié par `face_id` :

```python
emotion_buffers = {
    0: deque(["happy", "happy", "neutral", "happy", "happy"]),
    1: deque(["sad", "sad", "angry", "sad", "sad"]),
    2: deque(["neutral", "neutral", "neutral"])
}
```

Nettoyage automatique des buffers pour les visages qui disparaissent.

---

## 5. Historique des émotions

### Objectif

Suivre l'évolution des émotions sur une période plus longue pour détecter des patterns d'inconfort persistants.

### Structure de données

```python
class EmotionHistory:
    def __init__(self, max_size=15, min_size=5):
        self.max_size = max_size
        self.min_size = min_size
        self.history = deque(maxlen=max_size)

    def add(self, emotion: str, is_comfortable: bool):
        self.history.append({
            "emotion": emotion,
            "comfortable": is_comfortable,
            "timestamp": time.time()
        })
```

### Visualisation du buffer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HISTORIQUE DES ÉMOTIONS (15 max)                     │
└─────────────────────────────────────────────────────────────────────────┘

Position:  1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
         ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
Émotion: │ 😊 │ 😐 │ 😐 │ 😢 │ 😢 │ 😢 │ 😠 │ 😢 │ 😢 │ 😐 │ 😢 │ 😢 │ 😠 │ 😢 │ 😢 │
         └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
Confort:   ✓    ✓    ✓    ✗    ✗    ✗    ✗    ✗    ✗    ✓    ✗    ✗    ✗    ✗    ✗

                                        ▼
                         Analyse sur fenêtre glissante (8 derniers)
                                        ▼
                    Inconfort: 7/8 > Seuil (5/8) → Déclencher VLM
```

---

## 6. Système VLM (Questions intelligentes)

### Logique de déclenchement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ALGORITHME DE DÉCLENCHEMENT VLM                       │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │  Historique reçu    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ len(history) >= 5 ? │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                   Non                   Oui
                    │                     │
                    ▼                     ▼
              ┌───────────┐    ┌─────────────────────┐
              │  Attendre │    │ Analyser 8 derniers │
              └───────────┘    └──────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ Compter inconforts  │
                               │ dans la fenêtre     │
                               └──────────┬──────────┘
                                          │
                                          ▼
                               ┌─────────────────────┐
                               │ inconforts >= 5 ?   │
                               └──────────┬──────────┘
                                          │
                               ┌──────────┴──────────┐
                               │                     │
                              Non                   Oui
                               │                     │
                               ▼                     ▼
                         ┌───────────┐    ┌─────────────────────┐
                         │  Pas de   │    │ Générer question    │
                         │ question  │    │ contextuelle        │
                         └───────────┘    └─────────────────────┘
```

### Questions contextuelles

Les questions varient selon l'émotion dominante détectée :

| Émotion dominante | Question posée |
|-------------------|----------------|
| `sad` | "Vous semblez inconfortable. Avez-vous trop chaud ou trop froid ?" |
| `angry` | "Nous détectons une gêne. La température vous convient-elle ?" |
| `fear` | "Tout va bien ? Souhaitez-vous ajuster la climatisation ?" |
| `disgust` | "L'environnement vous semble-t-il confortable ?" |
| Défaut | "Comment vous sentez-vous ? Trop chaud, trop froid, ou ça va ?" |

### Options de réponse

```python
VLM_OPTIONS = [
    {"id": "hot", "label": "Trop chaud", "action": "decrease_temp"},
    {"id": "cold", "label": "Trop froid", "action": "increase_temp"},
    {"id": "ok", "label": "Ça va", "action": "no_change"}
]
```

### Parsing des réponses

```python
def parse_response(response: str) -> str:
    response_lower = response.lower()

    HOT_KEYWORDS = ["trop chaud", "chaud", "hot", "baisser", "diminuer"]
    COLD_KEYWORDS = ["trop froid", "froid", "cold", "augmenter", "monter"]

    for keyword in HOT_KEYWORDS:
        if keyword in response_lower:
            return "hot"

    for keyword in COLD_KEYWORDS:
        if keyword in response_lower:
            return "cold"

    return "ok"
```

---

## 7. Contrôle de température adaptatif

### Algorithme d'ajustement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTRÔLE DE TEMPÉRATURE                               │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │  Réponse utilisateur │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │  "hot"    │   │  "cold"   │   │   "ok"    │
        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
              │               │               │
              ▼               ▼               ▼
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │ T = T-1.5 │   │ T = T+1.5 │   │  T = T    │
        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Clamp(T, 16, 28)   │
                    └──────────┬──────────┘
                               │
                              ▼
                    ┌─────────────────────┐
                    │  Reset historique   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Retourner nouvelle T│
                    └─────────────────────┘
```

### Implémentation

```python
class TemperatureController:
    def __init__(self):
        self.current_temp = 22.0  # DEFAULT_TEMP
        self.min_temp = 16.0
        self.max_temp = 28.0
        self.step = 1.5

    def adjust(self, response: str) -> float:
        if response == "hot":
            self.current_temp -= self.step
        elif response == "cold":
            self.current_temp += self.step
        # "ok" -> pas de changement

        # Clamp dans les limites
        self.current_temp = max(self.min_temp,
                                min(self.max_temp, self.current_temp))

        return self.current_temp
```

### Exemple de scénario

```
État initial: T = 22°C

1. Détection d'inconfort persistant (7/8 frames)
2. Question: "Vous semblez inconfortable..."
3. Utilisateur: "Trop chaud"
4. Ajustement: T = 22 - 1.5 = 20.5°C
5. Reset historique
6. Nouveau cycle de détection...
```

---

## 8. Annotation visuelle

### Éléments annotés sur chaque frame

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Visages: 3 | Confort: 1 | Inconfort: 2        T: 22.5°C          │  │
│  │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  33%               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│         ┌─────────────┐                                                │
│         │             │                                                │
│         │   Visage 1  │  #1 happy (85%)                                │
│         │             │  [Rectangle VERT]                              │
│         └─────────────┘                                                │
│                                                                         │
│                   ┌─────────────┐     ┌─────────────┐                  │
│                   │             │     │             │                  │
│                   │   Visage 2  │     │   Visage 3  │                  │
│                   │             │     │             │                  │
│                   └─────────────┘     └─────────────┘                  │
│                   #2 sad (72%)        #3 angry (68%)                   │
│                   [Rectangle BLEU]    [Rectangle ROUGE]                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code couleur des émotions

| Émotion | Couleur BGR | Hex | Catégorie |
|---------|-------------|-----|-----------|
| `happy` | (0, 255, 0) | #00FF00 | Confort |
| `surprise` | (0, 255, 255) | #00FFFF | Confort |
| `neutral` | (128, 128, 128) | #808080 | Confort |
| `sad` | (255, 0, 0) | #0000FF | Inconfort |
| `angry` | (0, 0, 255) | #FF0000 | Inconfort |
| `fear` | (255, 0, 255) | #FF00FF | Inconfort |
| `disgust` | (0, 128, 128) | #808000 | Inconfort |

### Barre de confort

```python
def draw_comfort_bar(image, comfortable_count, total_count):
    ratio = comfortable_count / total_count if total_count > 0 else 0
    bar_width = 200
    filled_width = int(bar_width * ratio)

    # Fond gris
    cv2.rectangle(image, (10, 50), (10 + bar_width, 70), (50, 50, 50), -1)

    # Remplissage vert proportionnel
    cv2.rectangle(image, (10, 50), (10 + filled_width, 70), (0, 255, 0), -1)

    # Pourcentage
    cv2.putText(image, f"{int(ratio*100)}%", (220, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
```

---

## Résumé des paramètres clés

| Composant | Paramètre | Valeur | Justification |
|-----------|-----------|--------|---------------|
| Capture | FPS | 5 | Balance performance/réactivité |
| MTCNN | Min face size | 20px | Détecte visages à distance |
| FER | Confidence min | 40% | Filtre les prédictions incertaines |
| Lissage | Buffer size | 5 frames | ~1 seconde de stabilisation |
| Historique | Max size | 15 | ~3 secondes de contexte |
| Historique | Min size | 5 | Attendre avant analyse |
| VLM | Window | 8 frames | Fenêtre d'analyse |
| VLM | Threshold | 5/8 | 62.5% d'inconfort pour déclencher |
| Température | Step | 1.5°C | Ajustement perceptible |
| Température | Range | 16-28°C | Plage confort véhicule |

---

## Diagramme de séquence complet

```
┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│Frontend│     │ Frame  │     │Emotion │     │Smoother│     │  VLM   │
│        │     │ Router │     │Service │     │        │     │ Router │
└───┬────┘     └───┬────┘     └───┬────┘     └───┬────┘     └───┬────┘
    │              │              │              │              │
    │ POST /frame  │              │              │              │
    │─────────────▶│              │              │              │
    │              │ detect()     │              │              │
    │              │─────────────▶│              │              │
    │              │              │ smooth()     │              │
    │              │              │─────────────▶│              │
    │              │              │◀─────────────│              │
    │              │              │              │              │
    │              │◀─────────────│              │              │
    │◀─────────────│              │              │              │
    │              │              │              │              │
    │ GET /vlm-check              │              │              │
    │──────────────────────────────────────────────────────────▶│
    │              │              │ should_ask() │              │
    │              │              │◀─────────────────────────────│
    │◀──────────────────────────────────────────────────────────│
    │              │              │              │              │
    │ POST /vlm-response          │              │              │
    │──────────────────────────────────────────────────────────▶│
    │              │              │ clear_history()             │
    │              │              │◀─────────────────────────────│
    │◀──────────────────────────────────────────────────────────│
    │              │              │              │              │
```

---

## Références

- **FER2013 Dataset** : Challenges in Representation Learning (Kaggle)
- **MTCNN** : Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
- **OpenCV** : Open Source Computer Vision Library
- **FastAPI** : Modern, fast web framework for building APIs
