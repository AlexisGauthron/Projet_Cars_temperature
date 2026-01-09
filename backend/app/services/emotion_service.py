# -*- coding: utf-8 -*-
import cv2
import numpy as np
from fer import FER
from typing import Dict, Tuple, Optional, List
from collections import Counter
import base64

from app.config import EmotionConfig, VLMConfig, Settings
from app.services.emotion_smoother import emotion_smoother


class FaceEmotion:
    """Represents a detected face with its emotion."""

    def __init__(
        self,
        box: Tuple[int, int, int, int],
        emotions: Dict[str, float],
        dominant: str,
        smoothed_dominant: Optional[str] = None
    ):
        self.box = box
        self.emotions = emotions
        self.dominant = dominant
        # Émotion lissée (plus stable)
        self.smoothed_dominant = smoothed_dominant or dominant


class EmotionDetector:
    """
    Détecteur d'émotions faciales utilisant FER (Facial Emotion Recognition).
    Gère l'historique des émotions et les questions VLM.
    Intègre le lissage temporel pour éviter les faux positifs.
    """

    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.emotion_history: List[str] = []

        # Utiliser les catégories depuis la config
        self.comfort_emotions = EmotionConfig.COMFORT_EMOTIONS
        self.discomfort_emotions = EmotionConfig.DISCOMFORT_EMOTIONS

    def decode_base64_image(self, base64_string: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def encode_image_base64(self, image: np.ndarray) -> str:
        """Encode numpy array to base64 string."""
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, Settings.JPEG_QUALITY])
        return base64.b64encode(buffer).decode('utf-8')

    def detect_all_emotions(self, image: np.ndarray) -> List[FaceEmotion]:
        """
        Detect emotions from ALL faces in the image.
        Applique le lissage temporel pour éviter les faux positifs.

        Returns:
            List of FaceEmotion objects with smoothed emotions
        """
        try:
            results = self.detector.detect_emotions(image)

            if not results:
                # Nettoyer les buffers de lissage si aucun visage
                emotion_smoother.clear()
                return []

            faces = []
            active_face_ids = []

            for i, face_data in enumerate(results):
                box = tuple(face_data['box'])
                emotions = face_data['emotions']
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]

                # Appliquer le lissage temporel
                smoothed = emotion_smoother.add_emotion(i, dominant, confidence)

                faces.append(FaceEmotion(
                    box=box,
                    emotions=emotions,
                    dominant=dominant,
                    smoothed_dominant=smoothed
                ))
                active_face_ids.append(i)

            # Nettoyer les buffers des visages qui ont disparu
            emotion_smoother.cleanup_stale_faces(active_face_ids)

            # Update history with smoothed global emotion
            if faces:
                global_emotion = self._calculate_global_emotion(faces, use_smoothed=True)
                self._update_history(global_emotion)

            return faces

        except Exception as e:
            print(f"Error detecting emotions: {e}")
            return []

    def _calculate_global_emotion(self, faces: List[FaceEmotion], use_smoothed: bool = False) -> str:
        """
        Calculate the global emotion from all detected faces.

        Args:
            faces: Liste des visages détectés
            use_smoothed: Utiliser les émotions lissées (True) ou brutes (False)

        Returns:
            Émotion globale dominante
        """
        if not faces:
            return "neutral"

        if use_smoothed:
            # Vote majoritaire sur les émotions lissées
            emotions = [f.smoothed_dominant for f in faces]
            counter = Counter(emotions)
            return counter.most_common(1)[0][0]

        # Méthode originale: moyenne des scores
        emotion_totals = {
            "angry": 0, "disgust": 0, "fear": 0,
            "happy": 0, "sad": 0, "surprise": 0, "neutral": 0
        }

        for face in faces:
            for emotion, score in face.emotions.items():
                emotion_totals[emotion] += score

        num_faces = len(faces)
        emotion_averages = {e: s / num_faces for e, s in emotion_totals.items()}

        return max(emotion_averages, key=emotion_averages.get)

    def get_global_primary_emotion(self, faces: List[FaceEmotion]) -> str:
        """
        Determine global comfort status based on all faces.
        Utilise les émotions lissées pour plus de stabilité.

        Returns:
            "confortable" ou "inconfortable"
        """
        if not faces:
            return "confortable"

        discomfort_count = 0
        for face in faces:
            # Utiliser l'émotion lissée pour plus de stabilité
            if face.smoothed_dominant in self.discomfort_emotions:
                discomfort_count += 1

        # Vérifier si le seuil de majorité est dépassé
        if discomfort_count / len(faces) > EmotionConfig.COMFORT_MAJORITY_THRESHOLD:
            return "inconfortable"
        return "confortable"

    def get_faces_summary(self, faces: List[FaceEmotion]) -> Dict:
        """
        Get a summary of all detected faces.
        Utilise les émotions lissées pour le statut de confort.
        """
        if not faces:
            return {
                "total_faces": 0,
                "comfortable_count": 0,
                "uncomfortable_count": 0,
                "faces": []
            }

        comfortable = 0
        uncomfortable = 0
        faces_data = []

        for i, face in enumerate(faces):
            # Utiliser l'émotion lissée pour le statut
            is_comfortable = face.smoothed_dominant in self.comfort_emotions

            if is_comfortable:
                comfortable += 1
            else:
                uncomfortable += 1

            faces_data.append({
                "id": i + 1,
                "emotion": face.smoothed_dominant,  # Émotion lissée
                "raw_emotion": face.dominant,  # Émotion brute pour debug
                "confidence": round(face.emotions[face.dominant] * 100, 1),
                "status": "confortable" if is_comfortable else "inconfortable"
            })

        return {
            "total_faces": len(faces),
            "comfortable_count": comfortable,
            "uncomfortable_count": uncomfortable,
            "faces": faces_data
        }

    def _update_history(self, emotion: str):
        """Update emotion history for VLM questions."""
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > EmotionConfig.HISTORY_MAX_SIZE:
            self.emotion_history.pop(0)

    def should_ask_vlm_question(self) -> bool:
        """
        Vérifie si une question VLM doit être posée basée sur l'historique.

        Conditions:
        - Historique minimum atteint (HISTORY_MIN_SIZE)
        - Seuil d'inconfort dépassé (VLM_DISCOMFORT_THRESHOLD sur VLM_ANALYSIS_WINDOW)
        """
        if len(self.emotion_history) < EmotionConfig.HISTORY_MIN_SIZE:
            return False

        # Analyser la fenêtre d'émotions récentes
        recent = self.emotion_history[-VLMConfig.ANALYSIS_WINDOW:]
        discomfort_count = sum(1 for e in recent if e in self.discomfort_emotions)

        return discomfort_count >= VLMConfig.DISCOMFORT_THRESHOLD

    def get_vlm_question(self) -> Optional[Dict]:
        """
        Génère une question VLM contextuelle basée sur l'historique des émotions.

        Analyse les patterns d'émotions pour poser la question la plus pertinente.

        Returns:
            Dict avec question, options et type, ou None si pas de question
        """
        if not self.should_ask_vlm_question():
            return None

        # Analyser la fenêtre récente d'émotions
        recent = self.emotion_history[-VLMConfig.ANALYSIS_WINDOW:]

        # Compter les types d'émotions d'inconfort
        emotion_counts = {e: recent.count(e) for e in self.discomfort_emotions}

        # Trouver l'émotion d'inconfort dominante
        dominant_discomfort = max(emotion_counts, key=emotion_counts.get)
        dominant_count = emotion_counts[dominant_discomfort]

        # Récupérer la question depuis la config
        question_data = VLMConfig.get_question(dominant_discomfort)

        return {
            "question": question_data["question"],
            "options": question_data["options"],
            "question_type": "temperature",
            "context": {
                "dominant_emotion": dominant_discomfort,
                "intensity": round(dominant_count / len(recent) * 100, 1)
            }
        }

    def clear_history(self):
        """Clear emotion history after VLM response."""
        self.emotion_history = []


# Singleton instance
emotion_detector = EmotionDetector()
