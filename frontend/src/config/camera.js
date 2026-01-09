/**
 * Configuration de la caméra et capture vidéo.
 */

export const CAMERA_CONFIG = {
  // --- Résolution vidéo ---
  WIDTH: 640,
  HEIGHT: 480,

  // --- Qualité d'image ---
  // Qualité JPEG pour la capture (0.0 - 1.0)
  JPEG_QUALITY: 0.8,

  // --- Contraintes MediaDevices ---
  getConstraints() {
    return {
      video: {
        width: this.WIDTH,
        height: this.HEIGHT,
        facingMode: 'user', // Caméra frontale
      },
      audio: false,
    };
  },
};
