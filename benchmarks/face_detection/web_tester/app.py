# -*- coding: utf-8 -*-
"""
Web Tester pour les détecteurs de visage.

Mini application FastAPI pour tester visuellement les différents
détecteurs de visage avec une webcam.

Usage:
    cd benchmarks/face_detection/web_tester
    python app.py
    # Ouvrir http://localhost:8001
"""

import sys
from pathlib import Path

# Ajouter le chemin parent pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import base64
import time
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from detectors import DETECTOR_REGISTRY, get_detector, list_detectors

# Configuration
HOST = "0.0.0.0"
PORT = 8001

app = FastAPI(
    title="Face Detection Benchmark - Web Tester",
    description="Test visuel des détecteurs de visage",
    version="1.0.0"
)

# Templates et fichiers statiques
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(templates_dir))

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Cache des détecteurs (évite de recharger à chaque requête)
detector_cache = {}


def get_cached_detector(name: str):
    """Récupère un détecteur du cache ou le charge."""
    if name not in detector_cache:
        try:
            detector_cache[name] = get_detector(name)
        except Exception as e:
            print(f"[ERROR] Impossible de charger {name}: {e}")
            return None
    return detector_cache[name]


class DetectRequest(BaseModel):
    """Requête de détection."""
    image: str  # Base64 encoded image
    detector: str = "YuNet"


class MultiDetectRequest(BaseModel):
    """Requête de détection multi-modèles."""
    image: str  # Base64 encoded image
    detectors: list  # Liste des noms de détecteurs


class DetectResponse(BaseModel):
    """Réponse de détection."""
    success: bool
    detector: str
    faces_count: int
    time_ms: float
    boxes: list  # Liste de [x, y, w, h, confidence]
    annotated_image: Optional[str] = None  # Base64 encoded
    error: Optional[str] = None


class MultiDetectResponse(BaseModel):
    """Réponse de détection multi-modèles."""
    success: bool
    results: dict  # {detector_name: {faces_count, time_ms, boxes, color}}
    annotated_image: Optional[str] = None  # Base64 encoded
    error: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Page principale avec l'interface de test."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/detectors")
async def get_detectors():
    """Liste tous les détecteurs disponibles."""
    detectors_info = {}

    for name, cls in DETECTOR_REGISTRY.items():
        try:
            instance = get_cached_detector(name)
            if instance:
                detectors_info[name] = {
                    "name": instance.name,
                    "available": instance.is_available(),
                }
            else:
                detectors_info[name] = {
                    "name": name,
                    "available": False,
                }
        except Exception as e:
            detectors_info[name] = {
                "name": name,
                "available": False,
                "error": str(e)
            }

    # Trier: disponibles d'abord
    sorted_detectors = dict(
        sorted(detectors_info.items(), key=lambda x: (not x[1]["available"], x[0]))
    )

    return JSONResponse(content=sorted_detectors)


@app.post("/api/detect", response_model=DetectResponse)
async def detect_faces(request: DetectRequest):
    """Détecte les visages dans une image."""
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
                detector=request.detector,
                faces_count=0,
                time_ms=0,
                boxes=[],
                error="Impossible de décoder l'image"
            )

        # Récupérer le détecteur
        detector = get_cached_detector(request.detector)
        if detector is None or not detector.is_available():
            return DetectResponse(
                success=False,
                detector=request.detector,
                faces_count=0,
                time_ms=0,
                boxes=[],
                error=f"Détecteur '{request.detector}' non disponible"
            )

        # Détecter les visages
        start_time = time.perf_counter()
        faces = detector.detect(image)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Convertir les BBox en liste simple
        boxes = []
        for face in faces:
            boxes.append([
                face.x, face.y, face.w, face.h,
                getattr(face, 'confidence', 1.0)
            ])

        # Annoter l'image
        annotated = image.copy()
        for face in faces:
            x, y, w, h = face.x, face.y, face.w, face.h
            conf = getattr(face, 'confidence', 1.0)

            # Rectangle avec couleur basée sur la confiance
            color = (0, int(255 * conf), int(255 * (1 - conf)))  # Vert=haute conf, Rouge=basse
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label avec confiance
            label = f"{conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - 20), (x + label_size[0] + 4, y), color, -1)
            cv2.putText(annotated, label, (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encoder l'image annotée en base64
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return DetectResponse(
            success=True,
            detector=request.detector,
            faces_count=len(faces),
            time_ms=round(elapsed_ms, 2),
            boxes=boxes,
            annotated_image=f"data:image/jpeg;base64,{annotated_b64}"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return DetectResponse(
            success=False,
            detector=request.detector,
            faces_count=0,
            time_ms=0,
            boxes=[],
            error=str(e)
        )


# Couleurs pour les différents détecteurs (BGR pour OpenCV)
DETECTOR_COLORS = {
    "YuNet": (0, 255, 0),        # Vert
    "Haar": (0, 255, 255),       # Jaune
    "MediaPipe": (255, 255, 0),  # Cyan
    "FaceBoxes": (255, 0, 255),  # Magenta
    "CenterFace": (128, 255, 128), # Vert clair
    "OpenCV-DNN": (0, 165, 255), # Orange
    "DLib-HOG": (255, 0, 128),   # Violet
    "DLib-CNN": (128, 0, 255),   # Violet foncé
    "YOLO5Face": (0, 128, 255),  # Orange foncé
    "YOLOv8-face": (255, 128, 0), # Bleu clair
    "YOLOv9-face": (255, 64, 64), # Bleu
    "YOLOv10-face": (64, 255, 64), # Vert
    "YOLOv11-face": (64, 64, 255), # Rouge
    "YOLOv11-face-AdamCodd": (128, 128, 255), # Rouge clair
    "YOLOv12-face": (255, 128, 128), # Bleu clair
    "YOLO26": (128, 255, 255),   # Jaune clair
    "SCRFD": (0, 0, 255),        # Rouge
    "SCRFD_500M": (32, 32, 255), # Rouge
    "SCRFD_2.5G": (64, 64, 255), # Rouge
    "SCRFD_10G": (96, 96, 255),  # Rouge
    "SCRFD_34G": (128, 128, 255),# Rouge clair
    "RetinaFace": (255, 0, 0),   # Bleu
    "MTCNN": (255, 128, 0),      # Bleu clair
    "DSFD": (0, 128, 128),       # Olive
    "TinaFace": (128, 0, 128),   # Pourpre
}


def get_detector_color(name: str) -> tuple:
    """Retourne la couleur BGR pour un détecteur."""
    if name in DETECTOR_COLORS:
        return DETECTOR_COLORS[name]
    # Générer une couleur basée sur le hash du nom
    h = hash(name)
    return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))


def bgr_to_hex(bgr: tuple) -> str:
    """Convertit BGR en hex pour le frontend."""
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


@app.post("/api/detect_multi", response_model=MultiDetectResponse)
async def detect_faces_multi(request: MultiDetectRequest):
    """Détecte les visages avec plusieurs détecteurs simultanément."""
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
            return MultiDetectResponse(
                success=False,
                results={},
                error="Impossible de décoder l'image"
            )

        # Image annotée avec tous les détecteurs
        annotated = image.copy()
        results = {}

        for detector_name in request.detectors:
            detector = get_cached_detector(detector_name)
            if detector is None or not detector.is_available():
                results[detector_name] = {
                    "faces_count": 0,
                    "time_ms": 0,
                    "boxes": [],
                    "color": "#888888",
                    "error": "Non disponible"
                }
                continue

            # Détecter les visages
            start_time = time.perf_counter()
            faces = detector.detect(image)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Couleur pour ce détecteur
            color_bgr = get_detector_color(detector_name)
            color_hex = bgr_to_hex(color_bgr)

            # Convertir les BBox en liste simple
            boxes = []
            for face in faces:
                boxes.append([
                    face.x, face.y, face.w, face.h,
                    getattr(face, 'confidence', 1.0)
                ])

                # Dessiner sur l'image annotée
                x, y, w, h = face.x, face.y, face.w, face.h
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color_bgr, 2)

            # Ajouter le nom du détecteur en légende
            results[detector_name] = {
                "faces_count": len(faces),
                "time_ms": round(elapsed_ms, 2),
                "boxes": boxes,
                "color": color_hex
            }

        # Dessiner la légende
        y_offset = 25
        for detector_name, data in results.items():
            if "error" not in data:
                color_bgr = get_detector_color(detector_name)
                text = f"{detector_name}: {data['faces_count']} ({data['time_ms']:.0f}ms)"

                # Fond noir pour le texte
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (8, y_offset - text_h - 4), (12 + text_w, y_offset + 4), (0, 0, 0), -1)

                cv2.putText(annotated, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                y_offset += 28

        # Encoder l'image annotée en base64
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return MultiDetectResponse(
            success=True,
            results=results,
            annotated_image=f"data:image/jpeg;base64,{annotated_b64}"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return MultiDetectResponse(
            success=False,
            results={},
            error=str(e)
        )


@app.get("/api/health")
async def health_check():
    """Vérification de santé."""
    return {"status": "ok", "detectors_loaded": len(detector_cache)}


if __name__ == "__main__":
    print("=" * 60)
    print("FACE DETECTION WEB TESTER")
    print("=" * 60)
    print(f"URL: http://localhost:{PORT}")
    print("Ctrl+C pour quitter")
    print("=" * 60)

    uvicorn.run(app, host=HOST, port=PORT)
