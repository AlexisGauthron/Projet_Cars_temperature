/**
 * Configuration des temporisations de l'application.
 * Contrôle les intervalles de capture, vérification VLM, et cooldowns.
 */

export const TIMING_CONFIG = {
  // --- Capture des frames ---
  // Intervalle de capture (en ms)
  // 200ms = 5 FPS (suffisant pour la détection d'émotions)
  FRAME_INTERVAL_MS: 200,

  // --- Vérification VLM ---
  // Intervalle de vérification (en ms)
  // 2000ms = vérification toutes les 2 secondes
  VLM_CHECK_INTERVAL_MS: 2000,

  // Cooldown après une question VLM (en ms)
  // 15000ms = 15 secondes avant de pouvoir reposer une question
  VLM_COOLDOWN_MS: 15000,
};

// Valeurs calculées pour référence (lecture seule)
export const TIMING_INFO = {
  frameRate: `${1000 / TIMING_CONFIG.FRAME_INTERVAL_MS} FPS`,
  vlmCheckRate: `toutes les ${TIMING_CONFIG.VLM_CHECK_INTERVAL_MS / 1000}s`,
  vlmCooldown: `${TIMING_CONFIG.VLM_COOLDOWN_MS / 1000}s`,
};
