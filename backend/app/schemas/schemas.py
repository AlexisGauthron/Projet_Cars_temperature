from pydantic import BaseModel
from typing import Optional, List


class FrameRequest(BaseModel):
    image: str  # Base64 encoded image
    temperature: float
    mode: str = "single"  # "single" ou "multi"
    smoothing: bool = True  # True = lissé, False = émotions brutes
    model: str = "fer"  # "fer", "hsemotion", "deepface"


class FaceData(BaseModel):
    id: int
    emotion: str
    confidence: float
    status: str  # "confortable" ou "inconfortable"


class FacesSummary(BaseModel):
    total_faces: int
    comfortable_count: int
    uncomfortable_count: int
    faces: List[FaceData]


class PPGData(BaseModel):
    """Données de Photopléthysmographie pour le confort thermique."""
    pulsatile_intensity: float  # Intensité Pulsatile (0-1)
    thermal_state: str  # "cold", "cool", "neutral", "warm", "hot", "unknown"
    confidence: float  # Confiance de la mesure (0-1)
    buffer_fill: float  # Pourcentage de remplissage du buffer (0-1)


class FrameResponse(BaseModel):
    emotion: str  # Global dominant emotion
    annotated_image: str  # Base64 encoded annotated image
    primary_emotion: str  # "confortable" ou "inconfortable" (global)
    temperature: float
    faces_summary: FacesSummary  # Details of all detected faces
    ppg: Optional[PPGData] = None  # Données PPG (optionnel)


class VLMCheckResponse(BaseModel):
    """Réponse du check VLM avec question et options."""
    question: Optional[str] = None
    options: Optional[List[str]] = None  # Ex: ["Trop chaud", "Trop froid", "Ça va"]
    question_type: Optional[str] = None  # "comfort" ou "temperature"


class VLMResponseRequest(BaseModel):
    """Requête de réponse utilisateur au VLM."""
    response: str  # "chaud", "froid", ou "ok"


class VLMResponseResponse(BaseModel):
    """Réponse après traitement de la réponse utilisateur."""
    success: bool
    new_temperature: Optional[float] = None
    message: Optional[str] = None  # Message de confirmation
