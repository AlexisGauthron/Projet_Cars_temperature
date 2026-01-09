import axios from 'axios';
import { API_CONFIG } from '../config/api';

// Instance Axios avec configuration par défaut
const apiClient = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: API_CONFIG.TIMEOUT.DEFAULT,
});

export const api = {
  /**
   * Envoie une frame pour analyse d'émotions.
   */
  async sendFrame(imageBase64, temperature, mode = 'single') {
    try {
      const response = await apiClient.post(
        API_CONFIG.ENDPOINTS.FRAME,
        {
          image: imageBase64,
          temperature: temperature,
          mode: mode,
        },
        { timeout: API_CONFIG.TIMEOUT.FRAME }
      );
      return response.data;
    } catch (error) {
      console.error('Erreur envoi frame:', error);
      throw error;
    }
  },

  /**
   * Vérifie s'il y a une question VLM à afficher.
   */
  async checkVLM() {
    try {
      const response = await apiClient.get(
        API_CONFIG.ENDPOINTS.VLM_CHECK,
        { timeout: API_CONFIG.TIMEOUT.VLM }
      );
      return response.data;
    } catch (error) {
      console.error('Erreur VLM check:', error);
      throw error;
    }
  },

  /**
   * Envoie la réponse utilisateur à une question VLM.
   */
  async sendVLMResponse(userResponse) {
    try {
      const response = await apiClient.post(
        API_CONFIG.ENDPOINTS.VLM_RESPONSE,
        { response: userResponse },
        { timeout: API_CONFIG.TIMEOUT.VLM }
      );
      return response.data;
    } catch (error) {
      console.error('Erreur VLM response:', error);
      throw error;
    }
  },
};
