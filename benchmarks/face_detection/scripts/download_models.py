# -*- coding: utf-8 -*-
"""
Script de téléchargement des modèles de détection de visage.

Modèles supportés:
- YuNet (OpenCV) - Robuste aux occlusions
- OpenCV-DNN (SSD ResNet) - Meilleur compromis
- Haar Cascade - Ultra rapide
- RetinaFace - Haute précision
- MTCNN - Standard
- MediaPipe - Google
- DLib - HOG et CNN
- SCRFD (InsightFace) - State-of-the-art (ICLR 2022) ⭐
- YOLOv8-face - Rapide et précis ⭐
- YOLOv11-face - Dernier YOLO (2024) ⭐

Usage (depuis la racine du projet):
    python benchmarks/face_detection/scripts/download_models.py              # Télécharge tous les modèles
    python benchmarks/face_detection/scripts/download_models.py --list       # Lister les modèles
    python benchmarks/face_detection/scripts/download_models.py --model yunet # Un seul modèle
"""

import os
import urllib.request
import sys
from pathlib import Path

# Dossier des modèles
MODELS_DIR = Path(__file__).parent.parent / "models"

# URLs des modèles
MODELS = {
    "yunet": {
        "name": "YuNet (OpenCV)",
        "files": {
            "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        },
        "size": "~230KB",
        "description": "Détecteur léger, robuste aux occlusions partielles"
    },
    "opencv_dnn": {
        "name": "OpenCV DNN (SSD ResNet)",
        "files": {
            "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        },
        "size": "~10MB",
        "description": "Meilleur compromis précision/vitesse (99% détection)"
    },
    "haar": {
        "name": "Haar Cascade",
        "files": {
            "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_alt2.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml"
        },
        "size": "~2MB",
        "description": "Ultra rapide (0.3ms), précision limitée"
    },
    "retinaface": {
        "name": "RetinaFace",
        "files": {
            "retinaface_resnet50.onnx": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-640.onnx"
        },
        "size": "~30MB",
        "pip": "retinaface ou insightface",
        "description": "Haute précision, plus lent, nécessite images HD"
    },
    "mtcnn": {
        "name": "MTCNN",
        "files": {},  # Modèles inclus dans le package pip
        "size": "~2MB (via pip)",
        "pip": "mtcnn ou facenet-pytorch",
        "description": "Standard du domaine, bonne précision"
    },
    "mediapipe": {
        "name": "MediaPipe Face Detection",
        "files": {
            "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
            "blaze_face_full_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face/float16/latest/blaze_face.tflite"
        },
        "size": "~1MB",
        "pip": "mediapipe",
        "description": "Google, très rapide, optimisé mobile"
    },
    "dlib": {
        "name": "DLib",
        "files": {
            "mmod_human_face_detector.dat": "http://dlib.net/files/mmod_human_face_detector.dat.bz2",
            "shape_predictor_68_face_landmarks.dat": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        },
        "size": "~100MB",
        "pip": "dlib",
        "description": "HOG (rapide) et CNN (précis), landmarks 68 points"
    },
    "scrfd": {
        "name": "SCRFD (InsightFace) - Default",
        "files": {},  # Modèles téléchargés automatiquement par InsightFace
        "size": "~5MB",
        "pip": "insightface onnxruntime",
        "description": "State-of-the-art, meilleur compromis précision/vitesse (ICLR 2022)"
    },
    "scrfd_500m": {
        "name": "SCRFD 500M",
        "files": {
            "scrfd_500m_bnkps.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_500m_bnkps.onnx"
        },
        "size": "~2MB",
        "pip": "insightface onnxruntime",
        "description": "Ultra-léger (0.5 GFLOPs) - Easy: 90.57%, Medium: 88.12%, Hard: 68.51%"
    },
    "scrfd_2.5g": {
        "name": "SCRFD 2.5G",
        "files": {
            "scrfd_2.5g_bnkps.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx"
        },
        "size": "~3MB",
        "pip": "insightface onnxruntime",
        "description": "Équilibré (2.5 GFLOPs) - Easy: 93.78%, Medium: 92.16%, Hard: 77.87%"
    },
    "scrfd_10g": {
        "name": "SCRFD 10G",
        "files": {
            "scrfd_10g_bnkps.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_10g_bnkps.onnx"
        },
        "size": "~16MB",
        "pip": "insightface onnxruntime",
        "description": "Haute précision (10 GFLOPs) - Easy: 95.16%, Medium: 93.87%, Hard: 83.05%"
    },
    "scrfd_34g": {
        "name": "SCRFD 34G",
        "files": {
            "scrfd_34g.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_34g.onnx"
        },
        "size": "~68MB",
        "pip": "insightface onnxruntime",
        "description": "State-of-the-art (34 GFLOPs) - Easy: 96.06%, Medium: 94.92%, Hard: 85.29%"
    },
    "yolov8_face": {
        "name": "YOLOv8 Face Detection",
        "files": {},  # Modèle téléchargé via HuggingFace Hub
        "size": "~6MB",
        "pip": "ultralytics huggingface_hub",
        "huggingface": "arnabdhar/YOLOv8-Face-Detection",
        "description": "YOLO v8 optimisé pour la détection de visage"
    },
    "yolov11_face": {
        "name": "YOLOv11 Face Detection",
        "files": {},  # Modèle téléchargé via HuggingFace Hub
        "size": "~6MB",
        "pip": "ultralytics huggingface_hub",
        "huggingface": "AdamCodd/YOLOv11n-face-detection",
        "description": "YOLO 2024, Easy AP: 94.2%, Medium: 92.1%, Hard: 81.0%"
    },
    "yolov10_face": {
        "name": "YOLOv10 Face Detection",
        "files": {},  # Modèle téléchargé via HuggingFace Hub
        "size": "~6MB",
        "pip": "ultralytics huggingface_hub",
        "huggingface": "akanametov/yolov10-face",
        "description": "NeurIPS 2024, NMS-free YOLO, latence réduite"
    },
    "yolov12_face": {
        "name": "YOLOv12 Face Detection",
        "files": {},  # Modèle téléchargé via HuggingFace Hub
        "size": "~6MB",
        "pip": "ultralytics huggingface_hub",
        "huggingface": "akanametov/yolov12-face",
        "description": "Dernier YOLO (2026), attention-centric architecture"
    },
    "yolo5face": {
        "name": "YOLO5Face",
        "files": {},  # Modèle téléchargé via HuggingFace Hub
        "size": "~7MB",
        "pip": "ultralytics huggingface_hub",
        "huggingface": "akanametov/yolo5-face",
        "description": "SOTA WIDER FACE: 96.67% Easy, 95.08% Medium, 86.55% Hard"
    }
}


def download_file(url: str, dest_path: Path, show_progress: bool = True):
    """Télécharge un fichier avec barre de progression."""

    def progress_hook(count, block_size, total_size):
        if show_progress and total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
            sys.stdout.write(f"\r  [{bar}] {percent}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        if show_progress:
            print()  # Nouvelle ligne après la barre
        return True
    except Exception as e:
        print(f"\n  ERREUR: {e}")
        return False


def get_model_dir(model_key: str) -> Path:
    """
    Retourne le dossier de destination pour un modèle.

    IMPORTANT: Les chemins doivent correspondre à ceux attendus par les détecteurs!

    Mapping:
    - scrfd_* → models/scrfd/
    - yolov8_face → models/yolov8/
    - yolov9_face → models/yolov9/
    - yolov10_face → models/yolov10/
    - yolov11_face → models/yolov11/
    - yolov12_face → models/yolov12/
    - yolo5face → models/yolo5face/
    - Autres → models/<model_key>/
    """
    # Regrouper les modèles SCRFD dans models/scrfd/
    if model_key.startswith("scrfd"):
        return MODELS_DIR / "scrfd"

    # YOLO face: yolov8_face → models/yolov8/
    yolo_mapping = {
        "yolov8_face": "yolov8",
        "yolov9_face": "yolov9",
        "yolov10_face": "yolov10",
        "yolov11_face": "yolov11",
        "yolov12_face": "yolov12",
        # yolo5face reste tel quel (le détecteur cherche dans yolo5face/)
    }

    if model_key in yolo_mapping:
        return MODELS_DIR / yolo_mapping[model_key]

    # Par défaut: dossier avec le nom du modèle
    return MODELS_DIR / model_key


def download_model(model_key: str) -> bool:
    """Télécharge un modèle spécifique."""

    if model_key not in MODELS:
        print(f"Modèle inconnu: {model_key}")
        print(f"Modèles disponibles: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_key]
    model_dir = get_model_dir(model_key)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"📥 {model_info['name']}")
    print(f"{'='*60}")
    print(f"  Description: {model_info['description']}")
    print(f"  Taille: {model_info['size']}")

    if "pip" in model_info:
        print(f"  Package pip: {model_info['pip']}")

    if not model_info["files"]:
        print(f"  ℹ️  Modèle chargé via pip, pas de fichier à télécharger")

        # Créer un fichier INFO
        info_path = model_dir / "INFO.txt"
        with open(info_path, "w") as f:
            f.write(f"# {model_info['name']}\n\n")
            f.write(f"Description: {model_info['description']}\n")
            f.write(f"Installation: pip install {model_info['pip']}\n")
        return True

    success = True
    for filename, url in model_info["files"].items():
        dest_path = model_dir / filename

        if dest_path.exists():
            print(f"  ✓ {filename} (déjà présent)")
            continue

        print(f"  ⬇️  Téléchargement {filename}...")

        # Gérer les fichiers compressés
        if url.endswith(".bz2"):
            import bz2
            compressed_path = dest_path.with_suffix(dest_path.suffix + ".bz2")
            if download_file(url, compressed_path):
                print(f"  📦 Décompression...")
                with bz2.open(compressed_path, 'rb') as f_in:
                    with open(dest_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                compressed_path.unlink()  # Supprimer le fichier compressé
                print(f"  ✓ {filename}")
            else:
                success = False
        else:
            if download_file(url, dest_path):
                print(f"  ✓ {filename}")
            else:
                success = False

    return success


def download_all_models():
    """Télécharge tous les modèles."""

    print("\n" + "="*60)
    print("🚀 TÉLÉCHARGEMENT DES MODÈLES DE DÉTECTION DE VISAGE")
    print("="*60)
    print(f"\nDossier de destination: {MODELS_DIR}")
    print(f"Modèles à télécharger: {len(MODELS)}")

    results = {}
    for model_key in MODELS:
        results[model_key] = download_model(model_key)

    # Résumé
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)

    for model_key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {MODELS[model_key]['name']}")

    successful = sum(results.values())
    print(f"\n  Total: {successful}/{len(results)} modèles téléchargés")

    return all(results.values())


def list_models():
    """Liste tous les modèles disponibles."""

    print("\n" + "="*60)
    print("📋 MODÈLES DE DÉTECTION DE VISAGE DISPONIBLES")
    print("="*60)

    for key, info in MODELS.items():
        print(f"\n  [{key}] {info['name']}")
        print(f"      {info['description']}")
        print(f"      Taille: {info['size']}")
        if info["files"]:
            print(f"      Fichiers: {', '.join(info['files'].keys())}")
        if "pip" in info:
            print(f"      Install: pip install {info['pip']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Téléchargement des modèles de détection")
    parser.add_argument("--model", type=str, help="Télécharger un modèle spécifique")
    parser.add_argument("--list", action="store_true", help="Lister les modèles disponibles")
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.model:
        download_model(args.model)
    else:
        download_all_models()
