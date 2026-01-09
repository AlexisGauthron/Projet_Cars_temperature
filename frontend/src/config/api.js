/**
 * Configuration de l'API backend.
 */

// Utiliser les variables d'environnement si disponibles
const API_HOST = process.env.REACT_APP_API_HOST || 'localhost';
const API_PORT = process.env.REACT_APP_API_PORT || '8000';

export const API_CONFIG = {
  // URL de base de l'API
  BASE_URL: `http://${API_HOST}:${API_PORT}/api`,

  // Endpoints
  ENDPOINTS: {
    FRAME: '/frame',
    VLM_CHECK: '/vlm-check',
    VLM_RESPONSE: '/vlm-response',
  },

  // Timeouts (en ms)
  TIMEOUT: {
    DEFAULT: 10000,
    FRAME: 5000,
    VLM: 3000,
  },
};
