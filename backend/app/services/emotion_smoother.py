# -*- coding: utf-8 -*-
"""
Service de lissage temporel des émotions.
Évite les faux positifs causés par des mouvements rapides ou clignements.
"""
from collections import Counter, deque
from typing import Dict, List, Optional

from app.config.emotion import EmotionConfig


class EmotionSmoother:
    """
    Lisse les émotions détectées sur plusieurs frames pour plus de stabilité.
    Utilise un vote majoritaire sur une fenêtre glissante.
    """

    def __init__(self):
        # Buffer circulaire pour chaque visage (clé = index du visage)
        self._buffers: Dict[int, deque] = {}

    def _get_buffer(self, face_id: int) -> deque:
        """Récupère ou crée le buffer pour un visage donné."""
        if face_id not in self._buffers:
            self._buffers[face_id] = deque(maxlen=EmotionConfig.SMOOTHING_BUFFER_SIZE)
        return self._buffers[face_id]

    def add_emotion(self, face_id: int, emotion: str, confidence: float) -> str:
        """
        Ajoute une émotion au buffer et retourne l'émotion lissée.

        Args:
            face_id: Identifiant du visage
            emotion: Émotion détectée
            confidence: Score de confiance (0.0 - 1.0)

        Returns:
            Émotion lissée (majoritaire dans le buffer)
        """
        buffer = self._get_buffer(face_id)

        # Ne pas ajouter les émotions avec une confiance trop faible
        if confidence >= EmotionConfig.MIN_CONFIDENCE_THRESHOLD:
            buffer.append(emotion)

        return self.get_smoothed_emotion(face_id)

    def get_smoothed_emotion(self, face_id: int) -> str:
        """
        Retourne l'émotion majoritaire pour un visage.

        Args:
            face_id: Identifiant du visage

        Returns:
            Émotion majoritaire ou "neutral" si buffer vide
        """
        buffer = self._get_buffer(face_id)

        if not buffer:
            return "neutral"

        # Vote majoritaire
        counter = Counter(buffer)
        return counter.most_common(1)[0][0]

    def get_smoothed_emotions_all(self) -> Dict[int, str]:
        """Retourne les émotions lissées pour tous les visages connus."""
        return {
            face_id: self.get_smoothed_emotion(face_id)
            for face_id in self._buffers
        }

    def clear(self, face_id: Optional[int] = None):
        """
        Efface le buffer d'un visage ou de tous les visages.

        Args:
            face_id: ID du visage à effacer, ou None pour tout effacer
        """
        if face_id is not None:
            if face_id in self._buffers:
                self._buffers[face_id].clear()
        else:
            self._buffers.clear()

    def cleanup_stale_faces(self, active_face_ids: List[int]):
        """
        Supprime les buffers des visages qui ne sont plus détectés.

        Args:
            active_face_ids: Liste des IDs de visages actuellement visibles
        """
        stale_ids = [fid for fid in self._buffers if fid not in active_face_ids]
        for fid in stale_ids:
            del self._buffers[fid]


# Instance singleton
emotion_smoother = EmotionSmoother()
