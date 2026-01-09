#!/bin/bash

# Script de lancement ProjectCare
# Lance le backend et le frontend simultanement

echo "=========================================="
echo "   ProjectCare - Demarrage"
echo "=========================================="

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Repertoire du script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Fonction de nettoyage
cleanup() {
    echo ""
    echo -e "${YELLOW}Arret des serveurs...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Capture Ctrl+C
trap cleanup SIGINT SIGTERM

# Lancement du Backend
echo -e "${GREEN}[1/2] Demarrage du Backend (port 8000)...${NC}"
cd "$SCRIPT_DIR/backend"
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!

# Attendre que le backend demarre
sleep 3

# Lancement du Frontend
echo -e "${GREEN}[2/2] Demarrage du Frontend (port 3000)...${NC}"
cd "$SCRIPT_DIR/frontend"
npm start &
FRONTEND_PID=$!

# Attendre que le frontend demarre
sleep 5

# Ouvrir les pages dans le navigateur
echo -e "${GREEN}[3/3] Ouverture du navigateur...${NC}"
open http://localhost:3000
open http://localhost:8000/docs

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
