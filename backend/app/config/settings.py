# -*- coding: utf-8 -*-
"""
Configuration générale de l'application.
Paramètres serveur, CORS, et variables d'environnement.
"""
import os
from typing import List


class Settings:
    """Configuration générale du serveur et de l'API."""

    # --- API ---
    API_VERSION: str = "1.0.0"
    API_TITLE: str = "ProjectCare API"
    API_DESCRIPTION: str = "Système de détection d'émotions et contrôle de température"

    # --- Serveur ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # --- CORS ---
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # --- Qualité d'image ---
    JPEG_QUALITY: int = 80  # Qualité de compression JPEG (0-100)


# Instance singleton
settings = Settings()
