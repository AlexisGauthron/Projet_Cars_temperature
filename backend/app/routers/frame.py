from fastapi import APIRouter, HTTPException

from app.schemas.schemas import FrameRequest, FrameResponse, FacesSummary, FaceData, PPGData
from app.services.emotion_service import emotion_detector
from app.services.image_annotator import image_annotator
from app.services.ppg_service import ppg_service
from app.services.face_tracker import face_tracker

router = APIRouter()


@router.post("/frame", response_model=FrameResponse)
async def process_frame(request: FrameRequest):
    """
    Process a video frame and detect emotions.
    Mode: "single" (1 person) or "multi" (all faces)
    """
    try:
        # Decode image
        image = emotion_detector.decode_base64_image(request.image)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Detect emotions from ALL faces
        # smoothing=False → émotions brutes du modèle (pas de lissage)
        # model = "fer", "hsemotion", "deepface"
        faces = emotion_detector.detect_all_emotions(
            image,
            smoothing=request.smoothing,
            model=request.model
        )

        # Variable pour les données PPG
        ppg_data = None

        # Default values if no face detected
        if not faces:
            emotion = "neutral"
            primary_emotion = "confortable"
            annotated_image = emotion_detector.encode_image_base64(image)
            faces_summary = FacesSummary(
                total_faces=0,
                comfortable_count=0,
                uncomfortable_count=0,
                faces=[]
            )
        else:
            # MODE SINGLE: Utiliser le tracker pour suivre le même visage
            if request.mode == "single":
                # Convertir les FaceEmotion en dicts pour le tracker
                faces_dicts = [{'box': f.box, 'emotions': f.emotions, 'face_obj': f} for f in faces]

                # Le tracker trouve le meilleur match avec le visage cible
                tracked = face_tracker.update(faces_dicts)

                if tracked:
                    # Garder uniquement le visage tracké
                    faces = [tracked['face_obj']]
                else:
                    # Fallback: prendre le premier visage
                    faces = [faces[0]]
            else:
                # MODE MULTI: Reset le tracker (pas besoin de suivre)
                if face_tracker.target is not None:
                    face_tracker.reset()

            # Calculate global emotion (average of all faces)
            emotion = emotion_detector._calculate_global_emotion(faces)

            # Get global primary emotion (comfort status based on majority)
            primary_emotion = emotion_detector.get_global_primary_emotion(faces)

            # Annotate faces on the image
            annotated = image_annotator.annotate_all_faces(image, faces)
            annotated_image = emotion_detector.encode_image_base64(annotated)

            # Get summary of faces
            summary_dict = emotion_detector.get_faces_summary(faces)
            faces_summary = FacesSummary(
                total_faces=summary_dict["total_faces"],
                comfortable_count=summary_dict["comfortable_count"],
                uncomfortable_count=summary_dict["uncomfortable_count"],
                faces=[FaceData(**f) for f in summary_dict["faces"]]
            )

            # PPG: Ajouter la frame au buffer et calculer le confort thermique
            # Utiliser la première face détectée pour le PPG
            first_face = faces[0]
            face_box = first_face.box
            ppg_service.add_frame(image, face_box)

            # Obtenir les données PPG
            ppg_result = ppg_service.get_thermal_comfort()
            if ppg_result:
                ppg_data = PPGData(
                    pulsatile_intensity=ppg_result["pulsatile_intensity"],
                    thermal_state=ppg_result["thermal_state"],
                    confidence=ppg_result["confidence"],
                    buffer_fill=ppg_result["buffer_fill"]
                )

        return FrameResponse(
            emotion=emotion,
            annotated_image=f"data:image/jpeg;base64,{annotated_image}",
            primary_emotion=primary_emotion,
            temperature=request.temperature,
            faces_summary=faces_summary,
            ppg=ppg_data
        )

    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))
