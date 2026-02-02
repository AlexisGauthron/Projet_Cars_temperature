#!/bin/bash
# =============================================================================
# RunPod Setup Script - Face Detection Benchmark
# =============================================================================
# Ce script configure l'environnement RunPod pour exécuter les benchmarks
# de détection de visage de manière fiable et reproductible.
#
# Usage:
#   chmod +x runpod_setup.sh
#   ./runpod_setup.sh
#
# =============================================================================

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "============================================================================="
echo "   RUNPOD SETUP - Face Detection Benchmark"
echo "============================================================================="
echo -e "${NC}"

# -----------------------------------------------------------------------------
# 1. Vérification de l'environnement
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[1/6] Vérification de l'environnement...${NC}"

# Vérifier si on est sur RunPod
if [ -d "/workspace" ]; then
    WORKSPACE="/workspace"
    echo -e "${GREEN}  ✓ RunPod détecté - Workspace: $WORKSPACE${NC}"
else
    WORKSPACE="$(pwd)"
    echo -e "${YELLOW}  ⚠ Environnement local détecté - Workspace: $WORKSPACE${NC}"
fi

# Vérifier le GPU
echo ""
echo "  GPU disponible:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}  ⚠ Aucun GPU NVIDIA détecté${NC}"
    GPU_AVAILABLE=false
fi

# Vérifier Python
echo ""
echo "  Python: $(python --version 2>&1)"
echo "  Pip: $(pip --version 2>&1 | cut -d' ' -f1-2)"

# -----------------------------------------------------------------------------
# 2. Configuration du répertoire de travail
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/6] Configuration du répertoire...${NC}"

BENCHMARK_DIR="$WORKSPACE/face_detection_benchmark"

# Si le script est exécuté depuis le repo existant
if [ -f "$(pwd)/benchmark.py" ]; then
    BENCHMARK_DIR="$(pwd)"
    echo -e "${GREEN}  ✓ Utilisation du répertoire actuel: $BENCHMARK_DIR${NC}"
else
    # Créer le répertoire si nécessaire
    mkdir -p "$BENCHMARK_DIR"
    echo -e "${GREEN}  ✓ Répertoire créé: $BENCHMARK_DIR${NC}"

    # Copier les fichiers si on est dans le repo
    if [ -f "benchmark.py" ]; then
        cp -r . "$BENCHMARK_DIR/"
    fi
fi

cd "$BENCHMARK_DIR"

# -----------------------------------------------------------------------------
# 3. Installation des dépendances
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/6] Installation des dépendances...${NC}"

# Mettre à jour pip
pip install --upgrade pip --quiet

# Installer les dépendances de base
echo "  Installation des packages de base..."
pip install numpy opencv-python scipy tqdm Pillow requests gdown --quiet

# Installer PyTorch avec CUDA si GPU disponible
if [ "$GPU_AVAILABLE" = true ]; then
    echo "  Installation PyTorch avec CUDA..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

    echo "  Installation ONNX Runtime GPU..."
    pip install onnxruntime-gpu --quiet
else
    echo "  Installation PyTorch CPU..."
    pip install torch torchvision --quiet

    echo "  Installation ONNX Runtime CPU..."
    pip install onnxruntime --quiet
fi

# Installer les frameworks de détection
echo "  Installation des frameworks de détection..."
pip install mediapipe mtcnn facenet-pytorch insightface ultralytics --quiet

# DLib (peut prendre du temps à compiler)
echo "  Installation DLib (peut prendre quelques minutes)..."
pip install dlib --quiet || echo -e "${YELLOW}  ⚠ DLib non installé (optionnel)${NC}"

echo -e "${GREEN}  ✓ Dépendances installées${NC}"

# -----------------------------------------------------------------------------
# 4. Téléchargement des modèles
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[4/6] Téléchargement des modèles...${NC}"

# Créer le répertoire models
mkdir -p models

# Télécharger les modèles via le script existant si disponible
if [ -f "scripts/download_models.py" ]; then
    python scripts/download_models.py || echo -e "${YELLOW}  ⚠ Certains modèles n'ont pas pu être téléchargés${NC}"
fi

# Télécharger YuNet (OpenCV)
if [ ! -f "models/face_detection_yunet_2023mar.onnx" ]; then
    echo "  Téléchargement YuNet..."
    wget -q -O models/face_detection_yunet_2023mar.onnx \
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
        || echo -e "${YELLOW}  ⚠ YuNet non téléchargé${NC}"
fi

echo -e "${GREEN}  ✓ Modèles prêts${NC}"

# -----------------------------------------------------------------------------
# 5. Vérification du dataset et Network Volume
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[5/6] Vérification des datasets...${NC}"

# Vérifier si un Network Volume est monté avec les données
NETWORK_VOLUME_DATA="/workspace/data"

if [ -d "$NETWORK_VOLUME_DATA/datasets" ]; then
    echo -e "${GREEN}  ✓ Network Volume détecté: $NETWORK_VOLUME_DATA${NC}"

    # Créer les symlinks si nécessaire
    if [ ! -L "datasets" ] && [ ! -d "datasets" ]; then
        ln -s "$NETWORK_VOLUME_DATA/datasets" datasets
        echo "    Symlink créé: datasets → $NETWORK_VOLUME_DATA/datasets"
    fi

    if [ -d "$NETWORK_VOLUME_DATA/models" ] && [ ! -L "models" ]; then
        rm -rf models 2>/dev/null || true
        ln -s "$NETWORK_VOLUME_DATA/models" models
        echo "    Symlink créé: models → $NETWORK_VOLUME_DATA/models"
    fi

    if [ -d "$NETWORK_VOLUME_DATA/annotations" ] && [ ! -L "annotations" ]; then
        rm -rf annotations 2>/dev/null || true
        ln -s "$NETWORK_VOLUME_DATA/annotations" annotations
        echo "    Symlink créé: annotations → $NETWORK_VOLUME_DATA/annotations"
    fi
fi

# Vérifier les datasets disponibles
echo ""
echo "  Datasets disponibles:"

# WIDER FACE
if [ -d "datasets/wider_face" ] && [ -f "annotations/wider_face_split/wider_face_val_bbx_gt.txt" ]; then
    IMAGES_COUNT=$(find datasets/wider_face -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${GREEN}    ✓ WIDER FACE ($IMAGES_COUNT images)${NC}"
else
    echo -e "${YELLOW}    ✗ WIDER FACE non trouvé${NC}"
fi

# MAFA
if [ -d "datasets/mafa" ]; then
    IMAGES_COUNT=$(find datasets/mafa -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${GREEN}    ✓ MAFA ($IMAGES_COUNT images)${NC}"
else
    echo -e "${YELLOW}    ✗ MAFA non trouvé${NC}"
fi

# SVIRO
if [ -d "datasets/sviro" ]; then
    IMAGES_COUNT=$(find datasets/sviro -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    echo -e "${GREEN}    ✓ SVIRO ($IMAGES_COUNT images)${NC}"
else
    echo -e "${YELLOW}    ✗ SVIRO non trouvé${NC}"
fi

# Créer les répertoires si nécessaire
mkdir -p datasets models annotations

# -----------------------------------------------------------------------------
# 6. Test de l'installation
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[6/6] Test de l'installation...${NC}"

# Test des imports Python
python << 'EOF'
import sys
errors = []

# Test core
try:
    import numpy as np
    import cv2
    print(f"  ✓ OpenCV {cv2.__version__}")
except ImportError as e:
    errors.append(f"OpenCV: {e}")

# Test PyTorch
try:
    import torch
    cuda_status = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"  ✓ PyTorch {torch.__version__} ({cuda_status})")
except ImportError as e:
    errors.append(f"PyTorch: {e}")

# Test Ultralytics (YOLO)
try:
    import ultralytics
    print(f"  ✓ Ultralytics {ultralytics.__version__}")
except ImportError as e:
    errors.append(f"Ultralytics: {e}")

# Test InsightFace
try:
    import insightface
    print(f"  ✓ InsightFace {insightface.__version__}")
except ImportError as e:
    errors.append(f"InsightFace: {e}")

# Test MediaPipe
try:
    import mediapipe as mp
    print(f"  ✓ MediaPipe {mp.__version__}")
except ImportError as e:
    errors.append(f"MediaPipe: {e}")

if errors:
    print("\n  Erreurs:")
    for err in errors:
        print(f"    ✗ {err}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Tous les tests passés${NC}"
else
    echo -e "${RED}  ✗ Certains tests ont échoué${NC}"
fi

# -----------------------------------------------------------------------------
# Résumé
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}   SETUP TERMINÉ${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo "  Répertoire de travail: $BENCHMARK_DIR"
echo ""
echo "  Pour lancer un benchmark:"
echo -e "    ${YELLOW}python benchmark.py --list-models${NC}              # Voir les modèles"
echo -e "    ${YELLOW}python benchmark.py --dataset wider_face --limit 100${NC}"
echo -e "    ${YELLOW}python benchmark.py --strict --limit 300${NC}       # Mode strict"
echo ""
echo "  Ou utilisez le script de benchmark:"
echo -e "    ${YELLOW}./runpod_benchmark.sh${NC}"
echo ""
echo -e "${BLUE}=============================================================================${NC}"
