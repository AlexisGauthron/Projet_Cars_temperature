/**
 * Configuration du contrôle de température.
 */

export const TEMPERATURE_CONFIG = {
  // --- Valeurs ---
  DEFAULT: 20,      // Température initiale (°C)
  MIN: 16,          // Température minimale (°C)
  MAX: 28,          // Température maximale (°C)
  STEP: 0.5,        // Pas du slider (°C)

  // --- Affichage ---
  UNIT: '°C',
  DECIMALS: 1,      // Nombre de décimales affichées

  // --- Jauge visuelle ---
  GAUGE_MAX: 50,    // Valeur max pour le calcul de la jauge

  // Méthodes utilitaires
  format(temp) {
    return `${temp.toFixed(this.DECIMALS)}${this.UNIT}`;
  },

  clamp(temp) {
    return Math.max(this.MIN, Math.min(this.MAX, temp));
  },

  toGaugePercent(temp) {
    return (temp / this.GAUGE_MAX) * 100;
  },
};
