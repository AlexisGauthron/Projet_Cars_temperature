@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: Script de lancement ProjectCare pour Windows
:: Lance le backend et le frontend simultanement

echo ==========================================
echo    ProjectCare - Demarrage (Windows)
echo ==========================================

:: Repertoire du script
set SCRIPT_DIR=%~dp0

:: Verifier Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

:: Verifier Node.js
where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Node.js n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)

:: Verifier uvicorn
where uvicorn >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] uvicorn n'est pas installe. Executez: pip install uvicorn
    pause
    exit /b 1
)

:: Lancement du Backend
echo [1/2] Demarrage du Backend (port 8000)...
cd /d "%SCRIPT_DIR%backend"
start "ProjectCare Backend" cmd /c "uvicorn app.main:app --reload --port 8000"

:: Attendre que le backend demarre
timeout /t 3 /nobreak >nul

:: Lancement du Frontend
echo [2/2] Demarrage du Frontend (port 3000)...
cd /d "%SCRIPT_DIR%frontend"
start "ProjectCare Frontend" cmd /c "npm start"

:: Attendre que le frontend demarre
timeout /t 5 /nobreak >nul

:: Ouvrir les pages dans le navigateur
echo [3/3] Ouverture du navigateur...
start http://localhost:3000
start http://localhost:8000/docs

echo.
echo ==========================================
echo Serveurs demarres !
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo   API Docs: http://localhost:8000/docs
echo.
echo Fermez cette fenetre pour arreter les serveurs
echo ==========================================
echo.

:: Garder la fenetre ouverte
pause
