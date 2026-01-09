from fastapi import APIRouter, HTTPException

from app.schemas.schemas import FrameRequest, FrameResponse, FacesSummary, FaceData
from app.services.emotion_service import emotion_detector
from app.services.image_annotator import image_annotator

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
        faces = emotion_detector.detect_all_emotions(image)

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
            # MODE SINGLE: Only consider the first/largest face
            if request.mode == "single":
                # Keep only the first face (usually the largest/closest)
                faces = [faces[0]]

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

        return FrameResponse(
            emotion=emotion,
            annotated_image=f"data:image/jpeg;base64,{annotated_image}",
            primary_emotion=primary_emotion,
            temperature=request.temperature,
            faces_summary=faces_summary
        )

    except Exception as e:
        print(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))
