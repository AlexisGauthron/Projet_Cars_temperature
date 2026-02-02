# -*- coding: utf-8 -*-
"""
Router pour la création de dataset d'émotions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import base64

router = APIRouter(prefix="/dataset", tags=["dataset"])

# Dossier de base pour les datasets
DATA_DIR = Path(__file__).parent.parent.parent / "data"


class CaptureRequest(BaseModel):
    image: str  # Base64 encoded image
    emotion: str
    dataset_name: str = "my_dataset"


class CaptureResponse(BaseModel):
    success: bool
    path: str
    count: int


class CountsResponse(BaseModel):
    counts: dict
    total: int


VALID_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


@router.post("/capture", response_model=CaptureResponse)
async def capture_image(request: CaptureRequest):
    """
    Capture et sauvegarde une image pour le dataset.
    """
    # Valider l'émotion
    if request.emotion not in VALID_EMOTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid emotion. Must be one of: {VALID_EMOTIONS}"
        )

    # Créer le dossier
    dataset_dir = DATA_DIR / request.dataset_name / request.emotion
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Décoder l'image base64
    try:
        image_data = request.image
        if "," in image_data:
            image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Nom de fichier unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{request.emotion}_{timestamp}.jpg"
    filepath = dataset_dir / filename

    # Sauvegarder
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    # Compter les images dans ce dossier
    count = len(list(dataset_dir.glob("*.jpg")))

    print(f"[DATASET] Saved: {filepath} (total {request.emotion}: {count})")

    return CaptureResponse(
        success=True,
        path=str(filepath),
        count=count
    )


@router.get("/counts", response_model=CountsResponse)
async def get_counts(name: str = "my_dataset"):
    """
    Retourne le nombre d'images par émotion dans un dataset.
    """
    dataset_dir = DATA_DIR / name
    counts = {}
    total = 0

    for emotion in VALID_EMOTIONS:
        emotion_dir = dataset_dir / emotion
        if emotion_dir.exists():
            count = len(list(emotion_dir.glob("*.jpg"))) + len(list(emotion_dir.glob("*.png")))
            counts[emotion] = count
            total += count
        else:
            counts[emotion] = 0

    return CountsResponse(counts=counts, total=total)


@router.get("/list")
async def list_datasets():
    """
    Liste tous les datasets disponibles.
    """
    datasets = []

    if DATA_DIR.exists():
        for item in DATA_DIR.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Compter les images
                total = 0
                for emotion in VALID_EMOTIONS:
                    emotion_dir = item / emotion
                    if emotion_dir.exists():
                        total += len(list(emotion_dir.glob("*.jpg")))
                        total += len(list(emotion_dir.glob("*.png")))

                datasets.append({
                    "name": item.name,
                    "total_images": total
                })

    return {"datasets": datasets}
