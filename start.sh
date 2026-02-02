#!/bin/bash

# Script de lancement ProjectCare
# Lance le backend et le frontend simultanement
# Compatible macOS et Linux

echo "=========================================="
echo "   ProjectCare - Demarrage (Unix)"
echo "=========================================="

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Repertoire du script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detecter l'OS
OS="$(uname -s)"
case "$OS" in
    Linux*)  OPEN_CMD="xdg-open";;
    Darwin*) OPEN_CMD="open";;
    *)       OPEN_CMD="open";;
esac

# Verifier les dependances
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}[ERREUR] $1 n'est pas installe ou pas dans le PATH${NC}"
        exit 1
    fi
}

echo -e "${YELLOW}Verification des dependances...${NC}"
check_command python3 || check_command python
check_command node
check_command npm
# uvicorn sera lance via python -m uvicorn

# Fonction de nettoyage
cleanup() {
    echo ""
    echo -e "${YELLOW}Arret des serveurs...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    # Tuer aussi les processus enfants
    pkill -P $$ 2>/dev/null
    exit 0
}

# Capture Ctrl+C
trap cleanup SIGINT SIGTERM EXIT

# Lancement du Backend
echo -e "${GREEN}[1/2] Demarrage du Backend (port 8000)...${NC}"
cd "$SCRIPT_DIR/backend"
python -m uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# Attendre que le backend demarre
sleep 3

# Verifier que le backend est lance
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}[ERREUR] Le backend n'a pas demarre correctement${NC}"
    exit 1
fi

# Lancement du Frontend
echo -e "${GREEN}[2/2] Demarrage du Frontend (port 3000)...${NC}"
cd "$SCRIPT_DIR/frontend"
npm start &
FRONTEND_PID=$!

# Attendre que le frontend demarre
sleep 5

# Ouvrir les pages dans le navigateur
echo -e "${GREEN}[3/3] Ouverture du navigateur...${NC}"
$OPEN_CMD http://localhost:3000 2>/dev/null || true
$OPEN_CMD http://localhost:8000/docs 2>/dev/null || true

echo ""
echo "=========================================="
echo -e "${GREEN}Serveurs demarres !${NC}"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}Appuyez sur Ctrl+C pour arreter${NC}"
echo "=========================================="

# Attendre les processus
wait
