# -*- coding: utf-8 -*-
"""
Configuration du mode benchmark STRICT.

Le mode strict garantit des mesures reproductibles et comparables :
- Warmup pour éliminer les effets de cache/JIT
- Multi-passes pour des statistiques fiables
- Libération mémoire GPU entre les modèles
- Timer haute précision (time.perf_counter)

Ces paramètres s'appliquent à tous les benchmarks (face, emotion, etc.)

Usage:
    from benchmarks.config import get_profile, WARMUP_IMAGES, NUM_PASSES

    # Utiliser les valeurs par défaut
    print(f"Warmup: {WARMUP_IMAGES}, Passes: {NUM_PASSES}")

    # Utiliser un profil spécifique
    profile = get_profile("publication")
    print(f"Warmup: {profile['warmup_images']}")
"""

from typing import Dict, Any

# =============================================================================
# PROFILS PRÉDÉFINIS
# =============================================================================

PROFILES: Dict[str, Dict[str, Any]] = {
    # Test rapide pendant le développement
    "quick": {
        "warmup_images": 3,
        "num_passes": 1,
        "clear_gpu_memory": False,
        "gc_between_passes": False,
        "gc_between_models": False,
        "description": "Test rapide (dev uniquement)",
    },

    # Benchmark standard (défaut)
    "standard": {
        "warmup_images": 10,
        "num_passes": 3,
        "clear_gpu_memory": True,
        "gc_between_passes": False,
        "gc_between_models": True,
        "description": "Benchmark standard",
    },

    # Benchmark rigoureux pour publication
    "publication": {
        "warmup_images": 20,
        "num_passes": 5,
        "clear_gpu_memory": True,
        "gc_between_passes": True,
        "gc_between_models": True,
        "description": "Benchmark pour publication (plus lent)",
    },
}

# =============================================================================
# PROFIL PAR DÉFAUT
# =============================================================================

DEFAULT_PROFILE = "standard"

# =============================================================================
# PARAMÈTRES PAR DÉFAUT (extraits du profil par défaut)
# =============================================================================

# Nombre d'images de warmup
# - Élimine les temps d'initialisation (JIT, chargement cache, etc.)
# - Ces images ne sont PAS comptées dans les métriques
WARMUP_IMAGES: int = PROFILES[DEFAULT_PROFILE]["warmup_images"]

# Nombre de passes par image
# - Chaque image est traitée N fois
# - Le temps retenu = moyenne des N passes
# - Plus de passes = statistiques plus fiables
NUM_PASSES: int = PROFILES[DEFAULT_PROFILE]["num_passes"]

# Libérer la mémoire GPU entre chaque modèle
# - Garantit des conditions identiques
# - Évite les interférences entre modèles
CLEAR_GPU_MEMORY: bool = PROFILES[DEFAULT_PROFILE]["clear_gpu_memory"]

# Garbage collection entre les passes
GC_BETWEEN_PASSES: bool = PROFILES[DEFAULT_PROFILE]["gc_between_passes"]

# Garbage collection entre les modèles
GC_BETWEEN_MODELS: bool = PROFILES[DEFAULT_PROFILE]["gc_between_models"]

# =============================================================================
# PARAMÈTRES AVANCÉS
# =============================================================================

# Timer à utiliser
# - "perf_counter" : Haute précision (recommandé)
# - "time" : Standard
TIMER_FUNCTION: str = "perf_counter"

# Attendre la synchronisation GPU avant de mesurer
# - True = plus précis mais plus lent
GPU_SYNC_BEFORE_TIMING: bool = True


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_profile(name: str) -> Dict[str, Any]:
    """
    Récupère un profil de configuration par son nom.

    Args:
        name: Nom du profil ("quick", "standard", "publication")

    Returns:
        Dictionnaire avec les paramètres du profil

    Raises:
        ValueError: Si le profil n'existe pas
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Profil inconnu: '{name}'. Disponibles: {available}")

    return PROFILES[name].copy()


def list_profiles() -> Dict[str, Dict[str, Any]]:
    """
    Retourne tous les profils disponibles.

    Returns:
        Dictionnaire {nom: configuration}
    """
    return PROFILES.copy()


def print_profiles():
    """Affiche les profils disponibles de manière formatée."""
    print("\n" + "=" * 70)
    print("PROFILS MODE STRICT")
    print("=" * 70)

    for name, config in PROFILES.items():
        is_default = " [DÉFAUT]" if name == DEFAULT_PROFILE else ""
        print(f"\n  {name}{is_default}")
        print(f"    {config['description']}")
        print(f"    warmup={config['warmup_images']}, "
              f"passes={config['num_passes']}, "
              f"clear_gpu={config['clear_gpu_memory']}")

    print("\n" + "-" * 70)
    print("Usage:")
    print("  python benchmark.py --strict --profile publication")
    print("  python benchmark.py --strict --warmup 20 --passes 5")
    print("=" * 70 + "\n")
