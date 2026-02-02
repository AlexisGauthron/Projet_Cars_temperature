# -*- coding: utf-8 -*-
"""
Service de Photopléthysmographie (PPG) pour la détection du confort thermique.

Basé sur l'étude : "Vision-based Thermal Comfort Quantification for HVAC Control"

Principe :
- Vasodilatation (chaleur) → Amplitude PPG élevée → Intensité Pulsatile haute
- Vasoconstriction (froid) → Amplitude PPG faible → Intensité Pulsatile basse

Pipeline :
1. Extraction de la ROI (joue) depuis le visage détecté
2. Extraction des signaux RGB moyens par frame
3. ICA pour estimer le signal d'artefact de mouvement
4. Filtrage adaptatif LMS pour épurer le signal PPG
5. Calcul de l'Intensité Pulsatile (variance du signal)
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, List
import cv2

# =============================================================================
# CONFIGURATION
# =============================================================================

class PPGConfig:
    """Configuration du service PPG."""

    # Buffer de frames (30 fps * 10 secondes = 300 frames)
    BUFFER_SIZE = 300
    MIN_FRAMES_FOR_ANALYSIS = 90  # Minimum 3 secondes à 30 fps

    # Fréquence cardiaque attendue (filtrage)
    HR_MIN = 0.7  # Hz (~42 bpm)
    HR_MAX = 3.5  # Hz (~210 bpm)

    # Filtrage adaptatif LMS
    LMS_MU = 0.01  # Taux d'apprentissage
    LMS_ORDER = 32  # Ordre du filtre

    # Seuils d'Intensité Pulsatile pour le confort
    IP_COLD_THRESHOLD = 0.3      # En dessous = froid
    IP_NEUTRAL_LOW = 0.5         # Zone neutre basse
    IP_NEUTRAL_HIGH = 0.7        # Zone neutre haute
    IP_HOT_THRESHOLD = 0.8       # Au dessus = chaud

    # ROI de la joue (proportions relatives au visage)
    CHEEK_X_START = 0.1   # Début X (10% depuis la gauche)
    CHEEK_X_END = 0.4     # Fin X (40%)
    CHEEK_Y_START = 0.4   # Début Y (40% depuis le haut)
    CHEEK_Y_END = 0.7     # Fin Y (70%)


# =============================================================================
# EXTRACTION DE LA ROI (JOUE)
# =============================================================================

def extract_cheek_roi(image: np.ndarray, face_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Extrait la région de la joue droite depuis un visage détecté.

    Args:
        image: Image BGR
        face_box: (x, y, w, h) du visage

    Returns:
        ROI de la joue (BGR) ou None si échec
    """
    x, y, w, h = face_box

    # Calculer les coordonnées de la joue droite
    cheek_x1 = int(x + w * PPGConfig.CHEEK_X_START)
    cheek_x2 = int(x + w * PPGConfig.CHEEK_X_END)
    cheek_y1 = int(y + h * PPGConfig.CHEEK_Y_START)
    cheek_y2 = int(y + h * PPGConfig.CHEEK_Y_END)

    # Vérifier les limites
    cheek_x1 = max(0, cheek_x1)
    cheek_y1 = max(0, cheek_y1)
    cheek_x2 = min(image.shape[1], cheek_x2)
    cheek_y2 = min(image.shape[0], cheek_y2)

    if cheek_x2 <= cheek_x1 or cheek_y2 <= cheek_y1:
        return None

    return image[cheek_y1:cheek_y2, cheek_x1:cheek_x2]


def extract_rgb_signal(roi: np.ndarray) -> Tuple[float, float, float]:
    """
    Extrait les moyennes RGB d'une ROI.

    Args:
        roi: Image BGR de la ROI

    Returns:
        (R_mean, G_mean, B_mean)
    """
    # OpenCV utilise BGR, on convertit
    b_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    r_mean = np.mean(roi[:, :, 2])

    return (r_mean, g_mean, b_mean)


# =============================================================================
# ANALYSE EN COMPOSANTES INDÉPENDANTES (ICA)
# =============================================================================

def fast_ica(signals: np.ndarray, n_components: int = 3, max_iter: int = 200) -> np.ndarray:
    """
    Implémentation simplifiée de FastICA.

    Args:
        signals: Matrice (n_samples, n_features) des signaux RGB
        n_components: Nombre de composantes à extraire
        max_iter: Nombre maximum d'itérations

    Returns:
        Composantes indépendantes (n_components, n_samples)
    """
    # Centrer les données
    signals = signals - np.mean(signals, axis=0)

    # Blanchiment (whitening)
    cov = np.cov(signals.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Éviter division par zéro
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    signals_white = (whitening @ signals.T).T

    n_samples = signals_white.shape[0]

    # Initialisation aléatoire de la matrice de démixage
    np.random.seed(42)
    W = np.random.randn(n_components, n_components)

    # Orthogonalisation
    W = W @ np.linalg.inv(np.sqrt(W @ W.T))

    # Itérations FastICA
    for _ in range(max_iter):
        W_old = W.copy()

        for i in range(n_components):
            w = W[i, :]

            # Projection
            wx = signals_white @ w

            # Non-linéarité (tanh)
            g = np.tanh(wx)
            g_prime = 1 - g ** 2

            # Mise à jour
            w_new = np.mean(signals_white.T * g, axis=1) - np.mean(g_prime) * w

            # Décorrélation
            for j in range(i):
                w_new -= np.dot(w_new, W[j]) * W[j]

            # Normalisation
            w_new /= np.linalg.norm(w_new) + 1e-10
            W[i, :] = w_new

        # Vérifier convergence
        if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1)) < 1e-6:
            break

    # Calculer les composantes indépendantes
    sources = W @ signals_white.T

    return sources


def estimate_motion_artifact(rgb_signals: np.ndarray) -> np.ndarray:
    """
    Estime le signal d'artefact de mouvement via ICA.

    L'artefact de mouvement est identifié comme la composante ICA
    la plus corrélée avec le canal vert (qui contient le signal PPG).

    Args:
        rgb_signals: Array (n_frames, 3) des signaux RGB

    Returns:
        Signal d'artefact de mouvement estimé
    """
    # Appliquer ICA
    sources = fast_ica(rgb_signals, n_components=3)

    # Le canal vert (index 1) contient le signal PPG le plus fort
    green_signal = rgb_signals[:, 1]
    green_signal = (green_signal - np.mean(green_signal)) / (np.std(green_signal) + 1e-10)

    # Trouver la composante la plus corrélée avec le vert
    max_corr = -1
    motion_signal = sources[0]

    for i in range(sources.shape[0]):
        source = sources[i]
        source = (source - np.mean(source)) / (np.std(source) + 1e-10)

        corr = np.abs(np.corrcoef(green_signal, source)[0, 1])
        if corr > max_corr:
            max_corr = corr
            motion_signal = source

    return motion_signal


# =============================================================================
# FILTRAGE ADAPTATIF LMS
# =============================================================================

def lms_filter(signal: np.ndarray, reference: np.ndarray,
               mu: float = PPGConfig.LMS_MU,
               order: int = PPGConfig.LMS_ORDER) -> np.ndarray:
    """
    Filtre adaptatif LMS pour supprimer les artefacts de mouvement.

    Args:
        signal: Signal d'entrée (canal vert original)
        reference: Signal de référence (artefact de mouvement)
        mu: Taux d'apprentissage
        order: Ordre du filtre

    Returns:
        Signal épuré
    """
    n = len(signal)

    # Normaliser les signaux
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    reference = (reference - np.mean(reference)) / (np.std(reference) + 1e-10)

    # Initialisation
    w = np.zeros(order)
    output = np.zeros(n)

    # Padding du signal de référence
    reference_padded = np.concatenate([np.zeros(order - 1), reference])

    for i in range(n):
        # Vecteur d'entrée
        x = reference_padded[i:i + order][::-1]

        # Estimation de l'artefact
        artifact_estimate = np.dot(w, x)

        # Signal épuré (erreur)
        output[i] = signal[i] - artifact_estimate

        # Mise à jour des poids
        w = w + mu * output[i] * x

    return output


# =============================================================================
# CALCUL DE L'INTENSITÉ PULSATILE
# =============================================================================

def bandpass_filter(signal: np.ndarray, fps: float = 30.0,
                    low: float = PPGConfig.HR_MIN,
                    high: float = PPGConfig.HR_MAX) -> np.ndarray:
    """
    Filtre passe-bande simple pour isoler les fréquences cardiaques.

    Args:
        signal: Signal à filtrer
        fps: Fréquence d'échantillonnage
        low: Fréquence de coupure basse (Hz)
        high: Fréquence de coupure haute (Hz)

    Returns:
        Signal filtré
    """
    # FFT
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1.0 / fps)

    # Masque passe-bande
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)

    # Appliquer le masque
    fft_filtered = fft * mask

    # IFFT
    return np.real(np.fft.ifft(fft_filtered))


def compute_pulsatile_intensity(ppg_signal: np.ndarray, fps: float = 30.0) -> float:
    """
    Calcule l'Intensité Pulsatile (IP) à partir du signal PPG.

    L'IP est définie comme la variance du signal PPG dans la bande
    de fréquences cardiaques.

    Args:
        ppg_signal: Signal PPG épuré
        fps: Fréquence d'échantillonnage

    Returns:
        Intensité Pulsatile (valeur normalisée 0-1)
    """
    # Filtrer dans la bande cardiaque
    filtered = bandpass_filter(ppg_signal, fps)

    # Calculer la variance
    variance = np.var(filtered)

    # DEBUG: Afficher la variance pour calibrer
    print(f"[PPG DEBUG] Variance brute: {variance:.6f}")

    # Normalisation linéaire avec clipping
    # Basé sur des valeurs empiriques typiques (variance entre 0.001 et 0.5)
    # On utilise une échelle log pour mieux distribuer les valeurs
    if variance < 1e-10:
        normalized = 0.0
    else:
        # Échelle logarithmique pour mieux capturer la dynamique
        log_var = np.log10(variance + 1e-10)
        # Mapper [-3, 0] (variance 0.001 à 1) vers [0, 1]
        normalized = (log_var + 3) / 3
        normalized = np.clip(normalized, 0.0, 1.0)

    print(f"[PPG DEBUG] IP normalisée: {normalized:.4f}")

    return float(normalized)


# =============================================================================
# SERVICE PPG PRINCIPAL
# =============================================================================

class PPGService:
    """
    Service de mesure PPG pour le confort thermique.

    Utilisation :
        ppg = PPGService()

        # À chaque frame avec un visage détecté
        ppg.add_frame(image, face_box)

        # Obtenir l'état de confort
        result = ppg.get_thermal_comfort()
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialise le service PPG.

        Args:
            fps: Fréquence d'images attendue
        """
        self.fps = fps

        # Buffers pour les signaux RGB
        self.r_buffer = deque(maxlen=PPGConfig.BUFFER_SIZE)
        self.g_buffer = deque(maxlen=PPGConfig.BUFFER_SIZE)
        self.b_buffer = deque(maxlen=PPGConfig.BUFFER_SIZE)

        # Cache pour les résultats
        self._last_result: Optional[Dict] = None
        self._frames_since_update = 0
        self._update_interval = 30  # Recalculer toutes les 30 frames (1 sec)

    def add_frame(self, image: np.ndarray, face_box: Tuple[int, int, int, int]) -> bool:
        """
        Ajoute une frame au buffer PPG.

        Args:
            image: Image BGR
            face_box: (x, y, w, h) du visage détecté

        Returns:
            True si la frame a été ajoutée avec succès
        """
        # Extraire la ROI de la joue
        roi = extract_cheek_roi(image, face_box)
        if roi is None or roi.size == 0:
            return False

        # Extraire les signaux RGB
        r, g, b = extract_rgb_signal(roi)

        # Ajouter aux buffers
        self.r_buffer.append(r)
        self.g_buffer.append(g)
        self.b_buffer.append(b)

        self._frames_since_update += 1

        return True

    def get_thermal_comfort(self, force_update: bool = False) -> Optional[Dict]:
        """
        Calcule et retourne l'état de confort thermique.

        Args:
            force_update: Forcer le recalcul même si le cache est récent

        Returns:
            Dict avec :
                - pulsatile_intensity: Valeur IP (0-1)
                - thermal_state: "cold", "cool", "neutral", "warm", "hot"
                - confidence: Confiance de la mesure (0-1)
                - buffer_fill: Pourcentage de remplissage du buffer
        """
        n_frames = len(self.g_buffer)

        # Vérifier si assez de frames
        if n_frames < PPGConfig.MIN_FRAMES_FOR_ANALYSIS:
            return {
                "pulsatile_intensity": 0.5,
                "thermal_state": "unknown",
                "confidence": 0.0,
                "buffer_fill": n_frames / PPGConfig.BUFFER_SIZE
            }

        # Utiliser le cache si récent
        if not force_update and self._frames_since_update < self._update_interval:
            if self._last_result is not None:
                self._last_result["buffer_fill"] = n_frames / PPGConfig.BUFFER_SIZE
                return self._last_result

        # Construire la matrice des signaux RGB
        rgb_signals = np.array([
            list(self.r_buffer),
            list(self.g_buffer),
            list(self.b_buffer)
        ]).T  # Shape: (n_frames, 3)

        try:
            # 1. Estimer l'artefact de mouvement via ICA
            motion_artifact = estimate_motion_artifact(rgb_signals)

            # 2. Filtrage adaptatif LMS sur le canal vert
            green_signal = rgb_signals[:, 1]
            ppg_clean = lms_filter(green_signal, motion_artifact)

            # 3. Calculer l'Intensité Pulsatile
            ip = compute_pulsatile_intensity(ppg_clean, self.fps)

            # 4. Déterminer l'état thermique
            thermal_state = self._classify_thermal_state(ip)

            # 5. Calculer la confiance (basée sur le remplissage du buffer)
            confidence = min(1.0, n_frames / PPGConfig.BUFFER_SIZE)

            result = {
                "pulsatile_intensity": round(ip, 4),
                "thermal_state": thermal_state,
                "confidence": round(confidence, 2),
                "buffer_fill": round(n_frames / PPGConfig.BUFFER_SIZE, 2)
            }

            self._last_result = result
            self._frames_since_update = 0

            return result

        except Exception as e:
            print(f"[PPG ERROR] {e}")
            return {
                "pulsatile_intensity": 0.5,
                "thermal_state": "error",
                "confidence": 0.0,
                "buffer_fill": n_frames / PPGConfig.BUFFER_SIZE
            }

    def _classify_thermal_state(self, ip: float) -> str:
        """
        Classifie l'état thermique basé sur l'Intensité Pulsatile.

        Args:
            ip: Intensité Pulsatile (0-1)

        Returns:
            État thermique: "cold", "cool", "neutral", "warm", "hot"
        """
        if ip < PPGConfig.IP_COLD_THRESHOLD:
            return "cold"
        elif ip < PPGConfig.IP_NEUTRAL_LOW:
            return "cool"
        elif ip < PPGConfig.IP_NEUTRAL_HIGH:
            return "neutral"
        elif ip < PPGConfig.IP_HOT_THRESHOLD:
            return "warm"
        else:
            return "hot"

    def reset(self):
        """Réinitialise les buffers."""
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self._last_result = None
        self._frames_since_update = 0

    def get_debug_info(self) -> Dict:
        """
        Retourne des informations de debug.

        Returns:
            Dict avec statistiques des buffers
        """
        n = len(self.g_buffer)

        if n < 10:
            return {"buffer_size": n, "status": "collecting"}

        return {
            "buffer_size": n,
            "r_mean": round(np.mean(list(self.r_buffer)), 2),
            "g_mean": round(np.mean(list(self.g_buffer)), 2),
            "b_mean": round(np.mean(list(self.b_buffer)), 2),
            "r_std": round(np.std(list(self.r_buffer)), 4),
            "g_std": round(np.std(list(self.g_buffer)), 4),
            "b_std": round(np.std(list(self.b_buffer)), 4),
            "status": "ready" if n >= PPGConfig.MIN_FRAMES_FOR_ANALYSIS else "collecting"
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

ppg_service = PPGService()
