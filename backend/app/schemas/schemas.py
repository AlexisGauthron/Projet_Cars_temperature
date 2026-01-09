from pydantic import BaseModel
from typing import Optional, List


class FrameRequest(BaseModel):
    image: str  # Base64 encoded image
    temperature: float
    mode: str = "single"  # "single" ou "multi"


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


class FrameResponse(BaseModel):
    emotion: str  # Global dominant emotion
    annotated_image: str  # Base64 encoded annotated image
    primary_emotion: str  # "confortable" ou "inconfortable" (global)
    temperature: float
    faces_summary: FacesSummary  # Details of all detected faces


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
