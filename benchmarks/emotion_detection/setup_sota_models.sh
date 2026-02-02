#!/bin/bash
# =============================================================================
# Script d'installation des modèles SOTA pour la reconnaissance d'émotions
# =============================================================================
#
# Ce script télécharge et configure les modèles state-of-the-art:
# - POSTER++ (92.21% sur RAF-DB)
# - DAN (89.70% sur RAF-DB)
#
# Usage:
#   ./setup_sota_models.sh poster    # Installer POSTER++
#   ./setup_sota_models.sh dan       # Installer DAN
#   ./setup_sota_models.sh all       # Installer tous les modèles
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# -----------------------------------------------------------------------------
# POSTER++ Installation
# -----------------------------------------------------------------------------
setup_poster() {
    print_info "Installation de POSTER++ (POSTER V2)"
    echo ""

    POSTER_DIR="$MODELS_DIR/poster"
    mkdir -p "$POSTER_DIR"
    cd "$POSTER_DIR"

    # Cloner le repo si nécessaire
    if [ ! -d "POSTER_V2" ]; then
        print_info "Clonage du repo POSTER_V2..."
        git clone https://github.com/Talented-Q/POSTER_V2.git
        print_success "Repo cloné"
    else
        print_info "POSTER_V2 déjà présent"
    fi

    # Installer les dépendances
    print_info "Installation des dépendances..."
    pip install timm==0.9.7 --quiet

    # Télécharger les poids pré-entraînés
    echo ""
    print_warning "Poids pré-entraînés POSTER++"
    echo ""
    echo "Les poids doivent être téléchargés manuellement depuis:"
    echo "  https://github.com/Talented-Q/POSTER_V2#pretrained-models"
    echo ""
    echo "Placez les fichiers .pth dans:"
    echo "  $POSTER_DIR/"
    echo ""
    echo "Fichiers attendus:"
    echo "  - poster_rafdb.pth     (RAF-DB, 92.21%)"
    echo "  - poster_affectnet7.pth (AffectNet-7, 67.49%)"
    echo "  - poster_affectnet8.pth (AffectNet-8, 63.77%)"
    echo ""

    # Vérifier si les poids existent
    if [ -f "poster_rafdb.pth" ]; then
        print_success "Poids RAF-DB trouvés"
    else
        print_warning "Poids RAF-DB non trouvés"
    fi

    cd "$SCRIPT_DIR"
    print_success "Installation POSTER++ terminée"
}

# -----------------------------------------------------------------------------
# DAN Installation
# -----------------------------------------------------------------------------
setup_dan() {
    print_info "Installation de DAN (Distract your Attention Network)"
    echo ""

    DAN_DIR="$MODELS_DIR/dan"
    mkdir -p "$DAN_DIR"
    cd "$DAN_DIR"

    # Cloner le repo si nécessaire
    if [ ! -d "DAN" ]; then
        print_info "Clonage du repo DAN..."
        git clone https://github.com/yaoing/DAN.git
        print_success "Repo cloné"
    else
        print_info "DAN déjà présent"
    fi

    # Télécharger les poids pré-entraînés
    echo ""
    print_warning "Poids pré-entraînés DAN"
    echo ""
    echo "Les poids doivent être téléchargés depuis:"
    echo "  https://github.com/yaoing/DAN#pretrained-models"
    echo ""
    echo "Ou via Google Drive/Baidu links dans le README du repo."
    echo ""
    echo "Placez les fichiers .pth dans:"
    echo "  $DAN_DIR/"
    echo ""
    echo "Fichiers attendus:"
    echo "  - dan_rafdb.pth        (RAF-DB, 89.70%)"
    echo "  - dan_affectnet7.pth   (AffectNet-7, 65.69%)"
    echo "  - dan_affectnet8.pth   (AffectNet-8, 62.09%)"
    echo ""

    # Vérifier si les poids existent
    if [ -f "dan_rafdb.pth" ]; then
        print_success "Poids RAF-DB trouvés"
    else
        print_warning "Poids RAF-DB non trouvés"
    fi

    cd "$SCRIPT_DIR"
    print_success "Installation DAN terminée"
}

# -----------------------------------------------------------------------------
# Installation via pip des modèles simples
# -----------------------------------------------------------------------------
setup_pip_models() {
    print_info "Installation des modèles pip (production-ready)"
    echo ""

    pip install --quiet \
        deepface \
        hsemotion \
        hsemotion-onnx \
        fer \
        rmn \
        transformers \
        timm \
        py-feat

    print_success "Modèles pip installés"
}

# -----------------------------------------------------------------------------
# Vérification de l'installation
# -----------------------------------------------------------------------------
verify_installation() {
    print_info "Vérification de l'installation..."
    echo ""

    cd "$SCRIPT_DIR"
    python -c "
from classifiers import list_classifiers

print('Classifieurs disponibles:')
print('=' * 60)

for name, info in list_classifiers().items():
    status = '[OK]' if info['is_available'] else '[NOT AVAILABLE]'
    desc = info.get('description', '')[:40]
    print(f'  {name:20s} {status:15s} {desc}')
"
}

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
show_help() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commandes:"
    echo "  poster      Installer POSTER++ (SOTA sur RAF-DB)"
    echo "  dan         Installer DAN"
    echo "  pip         Installer les modèles pip (deepface, rmn, etc.)"
    echo "  all         Installer tous les modèles"
    echo "  verify      Vérifier l'installation"
    echo ""
    echo "Modèles SOTA nécessitent des poids pré-entraînés manuels."
    echo "Suivez les instructions affichées après l'installation."
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
case "${1:-help}" in
    poster)
        setup_poster
        ;;
    dan)
        setup_dan
        ;;
    pip)
        setup_pip_models
        ;;
    all)
        setup_pip_models
        echo ""
        echo "=============================================="
        setup_poster
        echo ""
        echo "=============================================="
        setup_dan
        ;;
    verify)
        verify_installation
        ;;
    *)
        show_help
        ;;
esac

echo ""
print_info "Pour vérifier l'installation: $0 verify"
