# Configuration ProjectCare

Ce document décrit l'architecture de configuration centralisée du projet.

## Vue d'ensemble

Tous les paramètres de l'application sont centralisés dans des fichiers de configuration dédiés, permettant un ajustement facile sans modifier le code métier.

```
ProjectCare/
├── backend/app/config/          # Configuration Backend (Python)
│   ├── __init__.py              # Exports centralisés
│   ├── settings.py              # Serveur, CORS, API
│   ├── emotion.py               # Détection d'émotions
│   ├── temperature.py           # Contrôle température
│   ├── vlm.py                   # Questions VLM
│   └── annotation.py            # Annotation visuelle
│
└── frontend/src/config/         # Configuration Frontend (JavaScript)
    ├── index.js                 # Exports centralisés
    ├── api.js                   # URLs et endpoints
    ├── timing.js                # Intervalles et cooldowns
    ├── camera.js                # Capture vidéo
    └── temperature.js           # Affichage température
```

---

## Backend (Python)

### Settings (`settings.py`)

Configuration du serveur FastAPI.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `API_VERSION` | `"1.0.0"` | Version de l'API |
| `HOST` | `"0.0.0.0"` | Adresse d'écoute |
| `PORT` | `8000` | Port du serveur |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Origines autorisées |
| `JPEG_QUALITY` | `80` | Qualité compression images |

### EmotionConfig (`emotion.py`)

Paramètres de détection d'émotions.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `COMFORT_EMOTIONS` | `{"happy", "surprise", "neutral"}` | Émotions confortables |
| `DISCOMFORT_EMOTIONS` | `{"sad", "angry", "fear", "disgust"}` | Émotions d'inconfort |
| `HISTORY_MAX_SIZE` | `15` | Taille max historique |
| `HISTORY_MIN_SIZE` | `5` | Minimum avant analyse VLM |
| `COMFORT_MAJORITY_THRESHOLD` | `0.5` | Seuil majorité (50%) |
| `SMOOTHING_BUFFER_SIZE` | `5` | Frames pour lissage |
| `MIN_CONFIDENCE_THRESHOLD` | `0.4` | Confiance minimale (40%) |

### TemperatureConfig (`temperature.py`)

Contrôle de température.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `MIN_TEMP` | `16.0` | Température min (°C) |
| `MAX_TEMP` | `28.0` | Température max (°C) |
| `DEFAULT_TEMP` | `22.0` | Température par défaut |
| `ADJUSTMENT_STEP` | `1.5` | Pas d'ajustement (°C) |

### VLMConfig (`vlm.py`)

Système de questions intelligentes.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `ANALYSIS_WINDOW` | `8` | Fenêtre d'analyse (frames) |
| `DISCOMFORT_THRESHOLD` | `5` | Seuil déclenchement (5/8) |
| `HOT_RESPONSES` | `["trop chaud", "chaud", ...]` | Réponses "chaud" |
| `COLD_RESPONSES` | `["trop froid", "froid", ...]` | Réponses "froid" |
| `QUESTIONS` | `{...}` | Questions par émotion |

### AnnotationConfig (`annotation.py`)

Styles d'annotation visuelle.

| Paramètre | Type | Description |
|-----------|------|-------------|
| `EMOTION_COLORS` | `Dict[str, BGR]` | Couleurs par émotion |
| `BORDER_THICKNESS` | `int` | Épaisseur rectangle |
| `FONT_SCALE` | `float` | Taille police |
| `SUMMARY_BAR_HEIGHT` | `int` | Hauteur barre résumé |
| `COMFORT_BAR_WIDTH` | `int` | Largeur barre confort |

---

## Frontend (JavaScript)

### API_CONFIG (`api.js`)

Configuration des appels API.

```javascript
{
  BASE_URL: 'http://localhost:8000/api',
  ENDPOINTS: {
    FRAME: '/frame',
    VLM_CHECK: '/vlm-check',
    VLM_RESPONSE: '/vlm-response',
  },
  TIMEOUT: {
    DEFAULT: 10000,
    FRAME: 5000,
    VLM: 3000,
  },
}
```

### TIMING_CONFIG (`timing.js`)

Intervalles de traitement.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `FRAME_INTERVAL_MS` | `200` | Intervalle capture (5 FPS) |
| `VLM_CHECK_INTERVAL_MS` | `2000` | Vérification VLM (2s) |
| `VLM_COOLDOWN_MS` | `15000` | Cooldown après question (15s) |

### CAMERA_CONFIG (`camera.js`)

Paramètres caméra.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `WIDTH` | `640` | Largeur vidéo (px) |
| `HEIGHT` | `480` | Hauteur vidéo (px) |
| `JPEG_QUALITY` | `0.8` | Qualité capture (80%) |

### TEMPERATURE_CONFIG (`temperature.js`)

Affichage température.

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `DEFAULT` | `20` | Température initiale |
| `MIN` | `16` | Minimum slider |
| `MAX` | `28` | Maximum slider |
| `STEP` | `0.5` | Pas slider |
| `GAUGE_MAX` | `50` | Max jauge visuelle |

---

## Variables d'environnement

### Backend

```bash
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

### Frontend

```bash
REACT_APP_API_HOST=localhost
REACT_APP_API_PORT=8000
```

---

## Exemples d'utilisation

### Backend

```python
from app.config import EmotionConfig, TemperatureConfig, VLMConfig

# Vérifier si une émotion est confortable
if EmotionConfig.is_comfortable("happy"):
    print("Confortable!")

# Limiter une température
temp = TemperatureConfig.clamp(35.0)  # Retourne 28.0

# Obtenir une question contextuelle
question = VLMConfig.get_question("sad")
```

### Frontend

```javascript
import { TIMING_CONFIG, TEMPERATURE_CONFIG } from './config';

// Intervalle de capture
setInterval(captureFrame, TIMING_CONFIG.FRAME_INTERVAL_MS);

// Formater température
const display = TEMPERATURE_CONFIG.format(22.5);  // "22.5°C"
```

---

## Bonnes pratiques

1. **Ne jamais hardcoder** de valeurs dans le code métier
2. **Documenter** tout nouveau paramètre ajouté
3. **Grouper** les paramètres liés dans le même fichier
4. **Utiliser des constantes** pour les valeurs fixes
5. **Prévoir des méthodes utilitaires** (format, clamp, etc.)
