#!/bin/bash
# =============================================================================
# RunPod Benchmark Script - Face Detection
# =============================================================================
# Lance les benchmarks de détection de visage en mode strict pour des
# résultats fiables et reproductibles.
#
# Usage:
#   ./runpod_benchmark.sh                    # Benchmark par défaut
#   ./runpod_benchmark.sh --quick            # Test rapide (50 images)
#   ./runpod_benchmark.sh --full             # Benchmark complet
#   ./runpod_benchmark.sh --models YuNet SCRFD  # Modèles spécifiques
#
# =============================================================================

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# -----------------------------------------------------------------------------
# Configuration par défaut
# -----------------------------------------------------------------------------
DATASET="wider_face"
LIMIT=300
WARMUP=10
PASSES=3
MODELS=""
MODE="strict"
FAST_MODE=false

# -----------------------------------------------------------------------------
# Parsing des arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick|-q)
            LIMIT=50
            WARMUP=5
            PASSES=2
            shift
            ;;
        --full|-f)
            LIMIT=""  # Toutes les images
            WARMUP=20
            PASSES=5
            shift
            ;;
        --medium|-m)
            LIMIT=300
            WARMUP=10
            PASSES=3
            shift
            ;;
        --limit|-l)
            LIMIT="$2"
            shift 2
            ;;
        --warmup|-w)
            WARMUP="$2"
            shift 2
            ;;
        --passes|-p)
            PASSES="$2"
            shift 2
            ;;
        --dataset|-d)
            DATASET="$2"
            shift 2
            ;;
        --models)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODELS="$MODELS $1"
                shift
            done
            ;;
        --fast)
            FAST_MODE=true
            shift
            ;;
        --standard)
            MODE="standard"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Presets:"
            echo "  --quick, -q       Test rapide (50 images, 5 warmup, 2 passes)"
            echo "  --medium, -m      Benchmark moyen (300 images, 10 warmup, 3 passes)"
            echo "  --full, -f        Benchmark complet (toutes images, 20 warmup, 5 passes)"
            echo ""
            echo "Options:"
            echo "  --limit, -l N     Nombre d'images à traiter"
            echo "  --warmup, -w N    Nombre d'images de warmup"
            echo "  --passes, -p N    Nombre de passes par image"
            echo "  --dataset, -d X   Dataset à utiliser (défaut: wider_face)"
            echo "  --models M1 M2    Modèles spécifiques à tester"
            echo "  --fast            Exclure les modèles lents (MTCNN, RetinaFace...)"
            echo "  --standard        Mode standard (pas strict)"
            echo ""
            echo "Exemples:"
            echo "  $0 --quick                    # Test rapide"
            echo "  $0 --full --fast              # Complet sans modèles lents"
            echo "  $0 --models YuNet SCRFD       # Seulement YuNet et SCRFD"
            echo "  $0 --limit 500 --passes 5     # 500 images, 5 passes"
            exit 0
            ;;
        *)
            echo -e "${RED}Option inconnue: $1${NC}"
            echo "Utilisez --help pour voir les options disponibles"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Affichage de la configuration
# -----------------------------------------------------------------------------
echo -e "${BLUE}"
echo "============================================================================="
echo "   BENCHMARK FACE DETECTION - RunPod"
echo "============================================================================="
echo -e "${NC}"

echo -e "${CYAN}Configuration:${NC}"
echo "  Dataset:    $DATASET"
if [ -z "$LIMIT" ]; then
    echo "  Images:     TOUTES"
else
    echo "  Images:     $LIMIT"
fi
echo "  Mode:       $MODE"
if [ "$MODE" = "strict" ]; then
    echo "  Warmup:     $WARMUP images"
    echo "  Passes:     $PASSES par image"
fi
if [ -n "$MODELS" ]; then
    echo "  Modèles:   $MODELS"
else
    echo "  Modèles:    TOUS"
fi
if [ "$FAST_MODE" = true ]; then
    echo "  Fast mode:  OUI (exclusion modèles lents)"
fi
echo ""

# -----------------------------------------------------------------------------
# Vérification GPU
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Vérification GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "${GREEN}  ✓ GPU: $GPU_NAME ($GPU_MEM)${NC}"

    # Vérifier la mémoire GPU disponible
    GPU_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "  Mémoire libre: ${GPU_FREE}MB"
else
    echo -e "${YELLOW}  ⚠ Pas de GPU détecté - Benchmark CPU${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# Préparation de la commande
# -----------------------------------------------------------------------------
CMD="python benchmark.py --dataset $DATASET"

# Ajouter la limite si spécifiée
if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Mode strict ou standard
if [ "$MODE" = "strict" ]; then
    CMD="$CMD --benchmark-strict --warmup $WARMUP --passes $PASSES"
fi

# Modèles spécifiques
if [ -n "$MODELS" ]; then
    CMD="$CMD --models$MODELS"
fi

# Fast mode
if [ "$FAST_MODE" = true ]; then
    CMD="$CMD --fast"
fi

echo -e "${CYAN}Commande:${NC}"
echo "  $CMD"
echo ""

# -----------------------------------------------------------------------------
# Confirmation
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Démarrage dans 3 secondes... (Ctrl+C pour annuler)${NC}"
sleep 3

# -----------------------------------------------------------------------------
# Exécution du benchmark
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}=============================================================================${NC}"
echo -e "${GREEN}   LANCEMENT DU BENCHMARK${NC}"
echo -e "${BLUE}=============================================================================${NC}"
echo ""

START_TIME=$(date +%s)

# Exécuter le benchmark
eval $CMD
BENCHMARK_EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# -----------------------------------------------------------------------------
# Résumé
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}=============================================================================${NC}"
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}   BENCHMARK TERMINÉ AVEC SUCCÈS${NC}"
else
    echo -e "${RED}   BENCHMARK TERMINÉ AVEC ERREURS (code: $BENCHMARK_EXIT_CODE)${NC}"
fi
echo -e "${BLUE}=============================================================================${NC}"
echo ""
echo "  Durée totale: ${MINUTES}m ${SECONDS}s"
echo ""

# Afficher les fichiers de résultats
echo -e "${CYAN}Fichiers de résultats:${NC}"
ls -la results/*.json 2>/dev/null | tail -5 || echo "  Aucun fichier trouvé"
echo ""

# Afficher le dernier fichier créé
LATEST_RESULT=$(ls -t results/*.json 2>/dev/null | head -1)
if [ -n "$LATEST_RESULT" ]; then
    echo -e "${CYAN}Dernier résultat: ${GREEN}$LATEST_RESULT${NC}"
    echo ""
    echo "Pour télécharger les résultats:"
    echo "  - Via JupyterLab: clic droit sur le fichier → Download"
    echo "  - Via SCP: scp user@host:$PWD/$LATEST_RESULT ./"
fi

echo ""
echo -e "${BLUE}=============================================================================${NC}"
