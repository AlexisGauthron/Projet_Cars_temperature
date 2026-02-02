# -*- coding: utf-8 -*-
import cv2
import numpy as np
from fer import FER
from typing import Dict, Tuple, Optional, List
from collections import Counter
import base64

# Patch global pour PyTorch 2.6+ (weights_only=True par défaut)
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from app.config import EmotionConfig, VLMConfig, Settings
from app.services.emotion_smoother import emotion_smoother
from app.services.face_detector import yunet_detector


# =============================================================================
# MAPPINGS POUR LES MODELES
# =============================================================================

# Mapping HSEmotion -> FER2013
HSEMOTION_TO_FER = {
    "anger": "angry",
    "contempt": "disgust",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise"
}

# Mapping DeepFace -> FER2013 (deja compatible)
DEEPFACE_TO_FER = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral"
}


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


# =============================================================================
# MODELES ADDITIONNELS
# =============================================================================

class HSEmotionDetector:
    """Wrapper pour HSEmotion."""

    def __init__(self):
        import torch
        from hsemotion.facial_emotions import HSEmotionRecognizer

        # Utiliser MPS (Apple Silicon) si disponible, sinon CPU
        if torch.backends.mps.is_available():
            device = 'mps'
            print("[INFO] HSEmotion: Using MPS (Apple Silicon GPU)")
        else:
            device = 'cpu'
            print("[INFO] HSEmotion: Using CPU")

        self.model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device=device)
        print("[INFO] HSEmotion model loaded successfully")

    def detect_emotions(self, image: np.ndarray) -> List[Dict]:
        """Détecte les émotions avec HSEmotion."""
        results = []

        # Détecter les visages avec FER/MTCNN
        faces = self.face_detector.detect_emotions(image)
        if not faces:
            return []

        for face_data in faces:
            box = face_data['box']
            x, y, w, h = box

            # Extraire le visage avec marge
            margin = int(0.1 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            face_crop = image[y1:y2, x1:x2]

            # Convertir en RGB
            if len(face_crop.shape) == 2:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2RGB)
            else:
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

            try:
                emotion, scores = self.model.predict_emotions(face_rgb, logits=False)

                # Mapper vers classes FER
                fer_emotion = HSEMOTION_TO_FER.get(emotion, emotion)

                # Construire dict d'émotions compatible
                hsemotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
                emotions_dict = {}
                for i, cls in enumerate(hsemotion_classes):
                    fer_cls = HSEMOTION_TO_FER.get(cls, cls)
                    score = float(scores[i]) if i < len(scores) else 0.0
                    if fer_cls in emotions_dict:
                        emotions_dict[fer_cls] = max(emotions_dict[fer_cls], score)
                    else:
                        emotions_dict[fer_cls] = score

                results.append({
                    'box': box,
                    'emotions': emotions_dict
                })
            except Exception as e:
                print(f"[HSEmotion ERROR] {e}")

        return results


class DeepFaceDetector:
    """Wrapper pour DeepFace."""

    def __init__(self):
        from deepface import DeepFace
        self.DeepFace = DeepFace
        # FER pour détection de visages
        self.face_detector = FER(mtcnn=True)

    def detect_emotions(self, image: np.ndarray) -> List[Dict]:
        """Détecte les émotions avec DeepFace."""
        results = []

        # Détecter les visages avec FER/MTCNN pour les boxes
        faces = self.face_detector.detect_emotions(image)

        try:
            # Sauvegarder temporairement
            temp_path = "/tmp/deepface_temp.jpg"
            cv2.imwrite(temp_path, image)

            df_results = self.DeepFace.analyze(
                img_path=temp_path,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )

            if not isinstance(df_results, list):
                df_results = [df_results]

            for i, df_data in enumerate(df_results):
                # Utiliser la box de FER si disponible
                if i < len(faces):
                    box = faces[i]['box']
                else:
                    # Fallback: utiliser la région de DeepFace
                    region = df_data.get('region', {})
                    box = (
                        region.get('x', 0),
                        region.get('y', 0),
                        region.get('w', 100),
                        region.get('h', 100)
                    )

                # Normaliser les scores (DeepFace donne 0-100)
                emotions_raw = df_data.get('emotion', {})
                emotions_dict = {}
                for emotion, score in emotions_raw.items():
                    fer_emotion = DEEPFACE_TO_FER.get(emotion, emotion)
                    emotions_dict[fer_emotion] = score / 100.0

                results.append({
                    'box': box,
                    'emotions': emotions_dict
                })

        except Exception as e:
            print(f"[DeepFace ERROR] {e}")

        return results


class EmotionDetector:
    """
    Détecteur d'émotions faciales multi-modèle.
    Supporte: FER, HSEmotion, DeepFace.
    Utilise YuNet pour la détection de visages (plus robuste aux occlusions).
    Gère l'historique des émotions et les questions VLM.
    Intègre le lissage temporel pour éviter les faux positifs.
    """

    def __init__(self):
        # Modèle FER pour classification d'émotions (sans MTCNN, on utilise YuNet)
        self.fer_detector = FER(mtcnn=False)  # Désactiver MTCNN, on utilise YuNet
        self.current_model = "fer"

        # Détecteur de visages YuNet (plus robuste aux occlusions)
        self.face_detector = yunet_detector

        # Modèles additionnels (chargés à la demande)
        self._hsemotion_detector = None
        self._deepface_detector = None

        self.emotion_history: List[str] = []

        # Utiliser les catégories depuis la config
        self.comfort_emotions = EmotionConfig.COMFORT_EMOTIONS
        self.discomfort_emotions = EmotionConfig.DISCOMFORT_EMOTIONS

    def _get_detector(self, model: str):
        """Retourne le détecteur approprié (lazy loading)."""
        if model == "fer":
            return self.fer_detector
        elif model == "hsemotion":
            if self._hsemotion_detector is None:
                print("[INFO] Loading HSEmotion model...")
                self._hsemotion_detector = HSEmotionDetector()
            return self._hsemotion_detector
        elif model == "deepface":
            if self._deepface_detector is None:
                print("[INFO] Loading DeepFace model...")
                self._deepface_detector = DeepFaceDetector()
            return self._deepface_detector
        else:
            return self.fer_detector

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

    def detect_all_emotions(self, image: np.ndarray, smoothing: bool = True, model: str = "fer") -> List[FaceEmotion]:
        """
        Detect emotions from ALL faces in the image.

        Utilise YuNet pour la détection de visages (robuste aux occlusions),
        puis le modèle choisi pour la classification des émotions.

        Args:
            image: Image numpy array
            smoothing: True = lissage temporel, False = émotions brutes
            model: "fer", "hsemotion", "deepface"

        Returns:
            List of FaceEmotion objects
        """
        try:
            self.current_model = model

            # 1. Détecter les visages avec YuNet (plus robuste)
            face_boxes = self.face_detector.detect(image)

            if not face_boxes:
                # Nettoyer les buffers de lissage si aucun visage
                emotion_smoother.clear()
                return []

            # 2. Classifier les émotions pour chaque visage détecté
            results = []

            for box in face_boxes:
                x, y, w, h = box

                # Extraire le visage avec une petite marge
                margin = int(0.1 * max(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image.shape[1], x + w + margin)
                y2 = min(image.shape[0], y + h + margin)
                face_crop = image[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                # Classifier l'émotion selon le modèle choisi
                emotions = self._classify_emotion(face_crop, model)

                if emotions:
                    results.append({
                        'box': box,
                        'emotions': emotions
                    })

            if not results:
                emotion_smoother.clear()
                return []

            faces = []
            active_face_ids = []

            for i, face_data in enumerate(results):
                box = tuple(face_data['box'])
                emotions = face_data['emotions']
                dominant = max(emotions, key=emotions.get)
                confidence = emotions[dominant]

                # DEBUG: Afficher les scores bruts
                print(f"[{model.upper()} DEBUG] Face {i}: {dominant} ({confidence:.2f}) | All: {', '.join(f'{k}:{v:.2f}' for k,v in sorted(emotions.items(), key=lambda x:-x[1]))}")

                if smoothing:
                    # Appliquer le lissage temporel
                    smoothed = emotion_smoother.add_emotion(i, dominant, confidence)
                else:
                    # Mode RAW: pas de lissage, émotion brute directe
                    smoothed = dominant

                faces.append(FaceEmotion(
                    box=box,
                    emotions=emotions,
                    dominant=dominant,
                    smoothed_dominant=smoothed
                ))
                active_face_ids.append(i)

            # Nettoyer les buffers des visages qui ont disparu
            if smoothing:
                emotion_smoother.cleanup_stale_faces(active_face_ids)

            # Update history with smoothed global emotion
            if faces:
                global_emotion = self._calculate_global_emotion(faces, use_smoothed=True)
                self._update_history(global_emotion)

            return faces

        except Exception as e:
            print(f"Error detecting emotions: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _classify_emotion(self, face_image: np.ndarray, model: str) -> Optional[Dict[str, float]]:
        """
        Classifie l'émotion d'un visage déjà extrait.

        Args:
            face_image: Image du visage (BGR)
            model: "fer", "hsemotion", "deepface"

        Returns:
            Dict des scores d'émotions ou None
        """
        try:
            if model == "fer":
                # FER attend une image complète, on lui passe le crop
                # Il va re-détecter le visage dedans
                result = self.fer_detector.detect_emotions(face_image)
                if result and len(result) > 0:
                    return result[0]['emotions']

                # Fallback: utiliser top_emotion qui est plus permissif
                emotion, score = self.fer_detector.top_emotion(face_image)
                if emotion:
                    # Créer un dict avec l'émotion dominante
                    emotions = {
                        "angry": 0.0, "disgust": 0.0, "fear": 0.0,
                        "happy": 0.0, "sad": 0.0, "surprise": 0.0, "neutral": 0.0
                    }
                    emotions[emotion] = score if score else 0.5
                    return emotions

            elif model == "hsemotion":
                detector = self._get_detector("hsemotion")
                # HSEmotion sur le crop directement
                if len(face_image.shape) == 2:
                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
                else:
                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                emotion, scores = detector.model.predict_emotions(face_rgb, logits=False)
                fer_emotion = HSEMOTION_TO_FER.get(emotion, emotion)

                hsemotion_classes = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
                emotions_dict = {}
                for idx, cls in enumerate(hsemotion_classes):
                    fer_cls = HSEMOTION_TO_FER.get(cls, cls)
                    score = float(scores[idx]) if idx < len(scores) else 0.0
                    if fer_cls in emotions_dict:
                        emotions_dict[fer_cls] = max(emotions_dict[fer_cls], score)
                    else:
                        emotions_dict[fer_cls] = score
                return emotions_dict

            elif model == "deepface":
                detector = self._get_detector("deepface")
                # Sauvegarder temporairement
                temp_path = "/tmp/deepface_crop.jpg"
                cv2.imwrite(temp_path, face_image)

                df_result = detector.DeepFace.analyze(
                    img_path=temp_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(df_result, list):
                    df_result = df_result[0]

                emotions_raw = df_result.get('emotion', {})
                emotions_dict = {}
                for emotion, score in emotions_raw.items():
                    fer_emotion = DEEPFACE_TO_FER.get(emotion, emotion)
                    emotions_dict[fer_emotion] = score / 100.0
                return emotions_dict

        except Exception as e:
            print(f"[EMOTION CLASSIFY ERROR] {model}: {e}")

        return None

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
