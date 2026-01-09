from fastapi import APIRouter

from app.schemas.schemas import VLMCheckResponse, VLMResponseRequest, VLMResponseResponse
from app.services.emotion_service import emotion_detector
from app.config import TemperatureConfig, VLMConfig

router = APIRouter()


class TemperatureController:
    """Gère l'état et les ajustements de température."""

    def __init__(self):
        self.current_temperature = TemperatureConfig.DEFAULT_TEMP
        self.last_question_asked = False

    def adjust_temperature(self, direction: str) -> tuple[float, str]:
        """
        Ajuste la température selon la direction.

        Args:
            direction: "chaud" (baisse temp), "froid" (augmente temp), ou "ok"

        Returns:
            tuple (nouvelle_temperature, message)
        """
        if direction == "chaud":
            # L'utilisateur a trop chaud -> baisser la température
            new_temp = self.current_temperature - TemperatureConfig.ADJUSTMENT_STEP
            new_temp = TemperatureConfig.clamp(new_temp)
            message = VLMConfig.MESSAGES["temp_decreased"].format(temp=new_temp)
        elif direction == "froid":
            # L'utilisateur a trop froid -> augmenter la température
            new_temp = self.current_temperature + TemperatureConfig.ADJUSTMENT_STEP
            new_temp = TemperatureConfig.clamp(new_temp)
            message = VLMConfig.MESSAGES["temp_increased"].format(temp=new_temp)
        else:
            # L'utilisateur est OK
            return self.current_temperature, VLMConfig.MESSAGES["temp_maintained"]

        self.current_temperature = new_temp
        return new_temp, message


# Instance singleton du contrôleur
temperature_controller = TemperatureController()


@router.get("/vlm-check", response_model=VLMCheckResponse)
async def check_vlm():
    """
    Vérifie s'il y a une question VLM à poser à l'utilisateur.
    Retourne la question avec ses options si applicable.
    """
    # Ne pas reposer si on vient de demander
    if temperature_controller.last_question_asked:
        return VLMCheckResponse(question=None, options=None, question_type=None)

    vlm_data = emotion_detector.get_vlm_question()

    if vlm_data:
        temperature_controller.last_question_asked = True
        return VLMCheckResponse(
            question=vlm_data["question"],
            options=vlm_data["options"],
            question_type=vlm_data["question_type"]
        )

    return VLMCheckResponse(question=None, options=None, question_type=None)


@router.post("/vlm-response", response_model=VLMResponseResponse)
async def handle_vlm_response(request: VLMResponseRequest):
    """
    Traite la réponse de l'utilisateur à la question VLM.

    Réponses attendues: "chaud", "froid", ou "ok"
    """
    # Reset de l'état
    temperature_controller.last_question_asked = False
    emotion_detector.clear_history()

    # Parser la réponse via la config
    direction = VLMConfig.parse_response(request.response)

    # Ajuster la température
    new_temp, message = temperature_controller.adjust_temperature(direction)

    return VLMResponseResponse(
        success=True,
        new_temperature=new_temp if direction != "ok" else None,
        message=message
    )
