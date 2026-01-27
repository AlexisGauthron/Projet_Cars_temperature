# -*- coding: utf-8 -*-
"""
Module de logging centralisé pour le benchmark de détection de visages.

Usage:
    from core.logger import get_logger
    logger = get_logger(__name__)

    logger.debug("Message de debug")
    logger.info("Information")
    logger.warning("Avertissement")
    logger.error("Erreur")
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Répertoire des logs
LOGS_DIR = Path(__file__).parent.parent / "logs"

# Format des messages
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FORMAT_SIMPLE = "%(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Niveau par défaut
DEFAULT_LEVEL = logging.INFO

# Flag pour éviter la configuration multiple
_configured = False


def setup_logging(
    level: int = DEFAULT_LEVEL,
    log_file: Optional[str] = None,
    verbose: bool = False
) -> None:
    """
    Configure le logging pour l'application.

    Args:
        level: Niveau de logging (logging.DEBUG, INFO, WARNING, ERROR)
        log_file: Chemin vers un fichier de log (optionnel)
        verbose: Si True, affiche aussi les messages DEBUG
    """
    global _configured

    if _configured:
        return

    # Niveau effectif
    effective_level = logging.DEBUG if verbose else level

    # Configurer le logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    # Handler console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(effective_level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT_SIMPLE))
    root_logger.addHandler(console_handler)

    # Handler fichier (optionnel)
    if log_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Toujours verbose dans le fichier
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root_logger.addHandler(file_handler)

    # Réduire le bruit des bibliothèques tierces
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger pour le module spécifié.

    Args:
        name: Nom du module (généralement __name__)

    Returns:
        Logger configuré
    """
    # S'assurer que le logging est configuré
    if not _configured:
        setup_logging()

    # Simplifier le nom pour l'affichage
    if name.startswith("benchmarks.face_detection."):
        name = name.replace("benchmarks.face_detection.", "")

    return logging.getLogger(name)


class SilentLogger:
    """Logger silencieux pour les cas où on veut désactiver les logs."""

    def debug(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def exception(self, *args, **kwargs): pass


def get_silent_logger() -> SilentLogger:
    """Retourne un logger silencieux."""
    return SilentLogger()
