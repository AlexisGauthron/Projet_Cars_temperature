# -*- coding: utf-8 -*-
"""
Classe de base pour les détecteurs de visage.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structures import BBox
from core.logger import get_logger

# Logger pour ce module
logger = get_logger(__name__)


class BaseDetector(ABC):
    """Interface commune pour tous les détecteurs de visage."""

    name: str = "BaseDetector"

    def __init__(self):
        """Initialise le détecteur avec logging."""
        self._logger = get_logger(f"detectors.{self.name}")

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[BBox]:
        """
        Détecte les visages dans une image.

        Args:
            image: Image BGR (numpy array)

        Returns:
            Liste de BBox avec les visages détectés
        """
        pass

    def is_available(self) -> bool:
        """
        Vérifie si le détecteur est disponible (modèle chargé, dépendances OK).

        Returns:
            True si le détecteur est prêt à l'emploi
        """
        return True

    def _log_init_success(self):
        """Log le succès de l'initialisation."""
        self._logger.debug(f"{self.name} initialized successfully")

    def _log_init_error(self, error: Exception):
        """Log une erreur d'initialisation."""
        self._logger.warning(f"{self.name} initialization failed: {error}")

    def _log_detection_error(self, error: Exception):
        """Log une erreur de détection."""
        self._logger.debug(f"{self.name} detection error: {error}")

    def __repr__(self) -> str:
        status = "✓" if self.is_available() else "✗"
        return f"{self.__class__.__name__}(name='{self.name}', available={status})"
