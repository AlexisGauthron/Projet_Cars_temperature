# ProjectCare - Stellantis POC

Système de détection d'émotions faciales et contrôle automatique de la température pour le confort des passagers.

## Architecture

```
ProjectCare/
├── frontend/          # Application React
├── backend/           # API FastAPI + modèles IA
│   ├── app/           # Code source API
│   ├── data/          # Datasets
│   └── training/      # Scripts d'entraînement
├── docs/              # Documentation
└── requirements.txt   # Dépendances Python
```

## Installation

### 1. Cloner le projet

```bash
git clone <repo-url>
cd ProjectCare
```

### 2. Créer l'environnement Python

```bash
python3 -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate     # Sur Windows

pip install -r requirements.txt
```

### 3. Installer les dépendances frontend

```bash
cd frontend
npm install
cd ..
```

## Lancement

### Option 1 : Script de démarrage (recommandé)

```bash
# macOS/Linux
./start.sh

# Windows
start.bat
```

### Option 2 : Lancement manuel

**Terminal 1 - Backend :**
```bash
source venv/bin/activate
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend :**
```bash
cd frontend
npm start
```

### Accès

- **Frontend** : http://localhost:3000
- **Backend API** : http://localhost:8000
- **API Docs** : http://localhost:8000/docs

## Fonctionnalités

### Interface principale

| Bouton | Fonction |
|--------|----------|
| **Model** | Changer le modèle de détection (FER, HSEmotion, DeepFace) |
| **Solo/Multi** | Mode 1 visage ou multi-visages |
| **Lissé/Brut** | Activer/désactiver le lissage temporel |
| **Dataset** | Ouvrir l'outil de création de dataset |

### Modèles disponibles

| Modèle | Description | Dataset d'entraînement |
|--------|-------------|------------------------|
| FER | Modèle par défaut, rapide | FER2013 |
| HSEmotion | Haute qualité | AffectNet |
| DeepFace | Multi-backend, robuste | VGGFace2 |

## Création de Dataset

### Via l'interface (recommandé)

1. Cliquer sur "Dataset" dans le header
2. Sélectionner une émotion (angry, happy, sad, etc.)
3. Le visage est automatiquement capturé et sauvegardé

Les images sont stockées dans : `backend/data/{nom_dataset}/{emotion}/`

### Via le script CLI

```bash
cd backend
python create_dataset.py --output data/mon_dataset
```

Contrôles :
- `1-7` : Sélectionner l'émotion
- `ESPACE` : Capturer
- `Q` : Quitter

## Test des Modèles

### Test sur webcam

```bash
cd backend
python test_models.py
```

Contrôles :
- `1` : FER
- `2` : HSEmotion
- `3` : DeepFace
- `a` : Tous les modèles
- `q` : Quitter

### Test sur une image

```bash
python test_models.py --image chemin/vers/image.jpg
```

### Test sur un dataset

```bash
# Tous les modèles
python test_models.py --dataset data/my_dataset

# Un seul modèle
python test_models.py --dataset data/my_dataset --model fer
python test_models.py --dataset data/my_dataset --model hsemotion
python test_models.py --dataset data/my_dataset --model deepface
```

### Benchmark avec limite d'images

```bash
# Test rapide sur 100 images (echantillonnage equilibre)
python test_models.py --dataset data/my_dataset --limit 100

# Benchmark complet sur FER2013
python test_models.py --dataset data/fer2013/test --limit 700
```

## Structure des Datasets

```
data/
├── my_dataset/           # Dataset personnalisé
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── fer2013/              # Dataset FER2013
    ├── train/
    └── test/
```

## Configuration

### Timing (frontend)

Fichier : `frontend/src/config/timing.js`

```javascript
export const TIMING_CONFIG = {
  FRAME_INTERVAL_MS: 200,      // Fréquence d'analyse (5 FPS)
  VLM_CHECK_INTERVAL_MS: 2000, // Vérification questions VLM
};
```

### Température

Fichier : `backend/app/config/temperature.py`

## Dépendances principales

- **Backend** : FastAPI, OpenCV, FER, HSEmotion, DeepFace, PyTorch
- **Frontend** : React, Axios

## Troubleshooting

### Le venv ne fonctionne pas

```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Port déjà utilisé

```bash
# Trouver le processus
lsof -i :8000  # ou :3000

# Tuer le processus
kill -9 <PID>
```

### Erreur PyTorch avec HSEmotion

Si vous avez une erreur `weights_only`, c'est un problème de compatibilité PyTorch 2.6+. Le code inclut déjà un patch automatique.
