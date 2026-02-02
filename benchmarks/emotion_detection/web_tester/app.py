# -*- coding: utf-8 -*-
"""
Web Tester pour les classifieurs d'émotions.

Mini application FastAPI pour tester visuellement les différents
modèles de détection d'émotions avec une webcam.

Usage:
    cd benchmarks/emotion_detection/web_tester
    python app.py
    # Ouvrir http://localhost:8002
"""

import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import base64
import time
from typing import Optional, List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from classifiers import CLASSIFIER_REGISTRY, get_classifier, list_classifiers

# Configuration
HOST = "0.0.0.0"
PORT = 8002

app = FastAPI(
    title="Emotion Detection Benchmark - Web Tester",
    description="Test visuel des classifieurs d'émotions",
    version="1.0.0"
)

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Cache des classifieurs
classifier_cache = {}

# Détecteur de visages (OpenCV DNN - rapide et fiable)
face_detector = None
FACE_PROTO = None
FACE_MODEL = None


def init_face_detector():
    """Initialise le détecteur de visages OpenCV DNN."""
    global face_detector, FACE_PROTO, FACE_MODEL

    if face_detector is not None:
        return face_detector

    try:
        # Chemins des modèles
        base_dir = Path(__file__).parent.parent
        proto_path = base_dir / "deploy.prototxt.txt"
        model_path = base_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if proto_path.exists() and model_path.exists():
            face_detector = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            print(f"[FaceDetector] OpenCV DNN chargé")
        else:
            # Fallback: Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_detector = cv2.CascadeClassifier(cascade_path)
            print(f"[FaceDetector] Haar Cascade chargé (fallback)")

    except Exception as e:
        print(f"[FaceDetector ERROR] {e}")

    return face_detector


def detect_faces(image: np.ndarray) -> List[tuple]:
    """Détecte les visages dans une image."""
    detector = init_face_detector()

    if detector is None:
        return []

    h, w = image.shape[:2]

    # OpenCV DNN
    if hasattr(detector, 'setInput'):
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        detector.setInput(blob)
        detections = detector.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((max(0, x1), max(0, y1), x2 - x1, y2 - y1, float(confidence)))
        return faces

    # Haar Cascade fallback
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rect = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return [(x, y, w, h, 1.0) for (x, y, w, h) in faces_rect]


def get_cached_classifier(name: str):
    """Récupère un classifieur du cache ou le charge."""
    if name not in classifier_cache:
        try:
            classifier_cache[name] = get_classifier(name)
        except Exception as e:
            print(f"[ERROR] Impossible de charger {name}: {e}")
            return None
    return classifier_cache[name]


class DetectRequest(BaseModel):
    """Requête de détection."""
    image: str  # Base64 encoded image
    classifiers: List[str] = ["hsemotion"]


class EmotionResult(BaseModel):
    """Résultat pour un classifieur."""
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    time_ms: float
    color: str


class DetectResponse(BaseModel):
    """Réponse de détection."""
    success: bool
    faces: List[Dict]  # Liste des visages avec leurs émotions
    results_by_classifier: Dict[str, EmotionResult]
    error: Optional[str] = None


# Couleurs pour les classifieurs
CLASSIFIER_COLORS = {
    "deepface": "#ff6b6b",
    "hsemotion": "#4ecdc4",
    "hsemotion_onnx": "#45b7d1",
    "fer_pytorch": "#96ceb4",
    "rmn": "#ffeaa7",
    "vit": "#dfe6e9",
    "deit": "#a29bfe",
    "pyfeat": "#fd79a8",
    "pyfeat_svm": "#e84393",
    "efficientnet": "#00b894",
    "efficientnet_b2": "#00cec9",
    "efficientnet_v2": "#0984e3",
    "poster": "#e17055",
    "poster_affectnet": "#d63031",
    "dan": "#6c5ce7",
    "dan_affectnet": "#a855f7",
}

# Couleurs pour les émotions
EMOTION_COLORS = {
    "angry": "#e74c3c",
    "disgust": "#9b59b6",
    "fear": "#8e44ad",
    "happy": "#f1c40f",
    "sad": "#3498db",
    "surprise": "#e67e22",
    "neutral": "#95a5a6",
    "contempt": "#7f8c8d",
}


def get_classifier_color(name: str) -> str:
    """Retourne la couleur hex pour un classifieur."""
    return CLASSIFIER_COLORS.get(name, "#888888")


def get_emotion_color(emotion: str) -> str:
    """Retourne la couleur hex pour une émotion."""
    return EMOTION_COLORS.get(emotion.lower(), "#888888")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Page principale avec l'interface de test."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/classifiers")
async def get_classifiers():
    """Liste tous les classifieurs disponibles."""
    classifiers_info = {}

    for name, cls in CLASSIFIER_REGISTRY.items():
        try:
            instance = get_cached_classifier(name)
            if instance:
                classifiers_info[name] = {
                    "name": instance.name,
                    "description": instance.description,
                    "available": instance.is_available(),
                    "color": get_classifier_color(name),
                }
            else:
                classifiers_info[name] = {
                    "name": name,
                    "description": "Error loading",
                    "available": False,
                    "color": "#888888",
                }
        except Exception as e:
            classifiers_info[name] = {
                "name": name,
                "description": str(e),
                "available": False,
                "color": "#888888",
            }

    # Trier: disponibles d'abord
    sorted_classifiers = dict(
        sorted(classifiers_info.items(), key=lambda x: (not x[1]["available"], x[0]))
    )

    return JSONResponse(content=sorted_classifiers)


@app.post("/api/detect")
async def detect_emotions(request: DetectRequest):
    """Détecte les émotions dans une image."""
    try:
        # Décoder l'image base64
        if "," in request.image:
            image_data = request.image.split(",")[1]
        else:
            image_data = request.image

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return DetectResponse(
                success=False,
                faces=[],
                results_by_classifier={},
                error="Impossible de décoder l'image"
            )

        # Détecter les visages
        faces = detect_faces(image)

        if not faces:
            return DetectResponse(
                success=True,
                faces=[],
                results_by_classifier={},
            )

        # Résultats par classifieur (agrégés sur tous les visages)
        results_by_classifier = {}

        # Résultats par visage
        faces_results = []

        for face_idx, (x, y, w, h, face_conf) in enumerate(faces):
            # Extraire le visage avec marge
            margin = int(0.15 * max(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image.shape[1], x + w + margin)
            y2 = min(image.shape[0], y + h + margin)
            face_crop = image[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            face_data = {
                "box": [x, y, w, h],
                "face_confidence": face_conf,
                "emotions": {}
            }

            # Classifier avec chaque modèle demandé
            for classifier_name in request.classifiers:
                classifier = get_cached_classifier(classifier_name)

                if classifier is None or not classifier.is_available():
                    continue

                try:
                    start_time = time.perf_counter()
                    prediction = classifier.predict(face_crop)
                    elapsed_ms = (time.perf_counter() - start_time) * 1000

                    emotion = prediction.label.value if hasattr(prediction.label, 'value') else str(prediction.label)
                    confidence = prediction.confidence

                    # Convertir probabilities
                    probs = {}
                    for label, prob in prediction.probabilities.items():
                        label_str = label.value if hasattr(label, 'value') else str(label)
                        probs[label_str] = float(prob)

                    face_data["emotions"][classifier_name] = {
                        "emotion": emotion,
                        "confidence": round(confidence, 3),
                        "probabilities": probs,
                        "time_ms": round(elapsed_ms, 2),
                    }

                    # Agréger dans results_by_classifier (prendre le premier visage pour l'instant)
                    if face_idx == 0 and classifier_name not in results_by_classifier:
                        results_by_classifier[classifier_name] = EmotionResult(
                            emotion=emotion,
                            confidence=round(confidence, 3),
                            probabilities=probs,
                            time_ms=round(elapsed_ms, 2),
                            color=get_classifier_color(classifier_name),
                        )

                except Exception as e:
                    print(f"[{classifier_name} ERROR] {e}")

            faces_results.append(face_data)

        return DetectResponse(
            success=True,
            faces=faces_results,
            results_by_classifier={k: v.dict() for k, v in results_by_classifier.items()},
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return DetectResponse(
            success=False,
            faces=[],
            results_by_classifier={},
            error=str(e)
        )


@app.get("/api/health")
async def health_check():
    """Vérification de santé."""
    return {"status": "ok", "classifiers_loaded": len(classifier_cache)}


if __name__ == "__main__":
    print("=" * 60)
    print("EMOTION DETECTION WEB TESTER")
    print("=" * 60)
    print(f"URL: http://localhost:{PORT}")
    print("Ctrl+C pour quitter")
    print("=" * 60)

    # Initialiser le détecteur de visages
    init_face_detector()

    uvicorn.run(app, host=HOST, port=PORT)
