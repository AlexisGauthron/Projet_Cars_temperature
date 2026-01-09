# -*- coding: utf-8 -*-
"""
Configuration du contrôle de température.
Plages, pas d'ajustement, et valeurs par défaut.
"""


class TemperatureConfig:
    """Paramètres pour le contrôle de température du véhicule."""

    # --- Plage de température ---
    MIN_TEMP: float = 16.0  # Température minimale (°C)
    MAX_TEMP: float = 28.0  # Température maximale (°C)
    DEFAULT_TEMP: float = 22.0  # Température par défaut (°C)

    # --- Ajustements ---
    ADJUSTMENT_STEP: float = 1.5  # Pas d'ajustement par action (°C)

    # --- Affichage ---
    TEMP_UNIT: str = "°C"
    DECIMAL_PLACES: int = 1  # Nombre de décimales affichées

    @classmethod
    def clamp(cls, temperature: float) -> float:
        """Limite la température aux bornes autorisées."""
        return max(cls.MIN_TEMP, min(cls.MAX_TEMP, temperature))

    @classmethod
    def format(cls, temperature: float) -> str:
        """Formate la température pour l'affichage."""
        return f"{temperature:.{cls.DECIMAL_PLACES}f}{cls.TEMP_UNIT}"
