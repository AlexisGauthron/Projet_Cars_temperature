# -*- coding: utf-8 -*-
"""
Benchmark des détecteurs de visage.

Compare les performances de différents détecteurs:
- YuNet (OpenCV) - Robuste aux occlusions
- MTCNN - Standard du domaine
- RetinaFace - Haute précision
- MediaPipe - Très rapide (Google)
- Haar Cascade - Classique OpenCV
- DLib HOG - Robuste

Métriques:
- Taux de détection (recall)
- Vitesse (ms/image)
- Faux positifs
- Robustesse aux conditions variées

Usage:
    python benchmark_face_detectors.py                          # Test webcam
    python benchmark_face_detectors.py --dataset data/my_dataset
    python benchmark_face_detectors.py --dataset data/fer2013/test --limit 500
    python benchmark_face_detectors.py --image photo.jpg
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Callable
import sys
import os

# Ajouter le chemin pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# DÉTECTEURS DE VISAGE
# =============================================================================

class BaseDetector:
    """Classe de base pour les détecteurs."""
    name: str = "Base"

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Retourne liste de (x, y, w, h)."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Vérifie si le détecteur est disponible."""
        return True


class YuNetDetector(BaseDetector):
    """YuNet - OpenCV intégré, robuste aux occlusions."""
    name = "YuNet"

    def __init__(self):
        self.detector = None
        self._load()

    def _load(self):
        try:
            from app.services.face_detector import yunet_detector
            self.detector = yunet_detector
        except Exception as e:
            print(f"[YuNet] Erreur chargement: {e}")
            self.detector = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector is None:
            return []
        return self.detector.detect(image)

    def is_available(self) -> bool:
        return self.detector is not None


class MTCNNDetector(BaseDetector):
    """MTCNN - Multi-task Cascaded CNN."""
    name = "MTCNN"

    def __init__(self):
        self.detector = None
        self._load()

    def _load(self):
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            print(f"[MTCNN] Chargé avec succès")
        except ImportError:
            try:
                # Alternative: via facenet-pytorch
                from facenet_pytorch import MTCNN as MTCNN_PT
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.detector = MTCNN_PT(keep_all=True, device=device)
                self._is_pytorch = True
                print(f"[MTCNN] Chargé via facenet-pytorch ({device})")
            except ImportError:
                print(f"[MTCNN] Non disponible. Installer: pip install mtcnn ou pip install facenet-pytorch")
                self.detector = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector is None:
            return []

        try:
            # Convertir BGR -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if hasattr(self, '_is_pytorch') and self._is_pytorch:
                # facenet-pytorch MTCNN
                boxes, _ = self.detector.detect(rgb)
                if boxes is None:
                    return []
                result = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    result.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))
                return result
            else:
                # mtcnn standard
                faces = self.detector.detect_faces(rgb)
                return [tuple(f['box']) for f in faces]
        except Exception as e:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class RetinaFaceDetector(BaseDetector):
    """RetinaFace - Haute précision."""
    name = "RetinaFace"

    def __init__(self):
        self.detector = None
        self._load()

    def _load(self):
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
            print(f"[RetinaFace] Chargé avec succès")
        except ImportError:
            try:
                # Alternative: insightface
                import insightface
                self.detector = insightface.app.FaceAnalysis(allowed_modules=['detection'])
                self.detector.prepare(ctx_id=0, det_size=(640, 640))
                self._is_insightface = True
                print(f"[RetinaFace] Chargé via insightface")
            except ImportError:
                print(f"[RetinaFace] Non disponible. Installer: pip install retinaface ou pip install insightface")
                self.detector = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector is None:
            return []

        try:
            if hasattr(self, '_is_insightface') and self._is_insightface:
                faces = self.detector.get(image)
                result = []
                for face in faces:
                    box = face.bbox.astype(int)
                    x1, y1, x2, y2 = box
                    result.append((x1, y1, x2-x1, y2-y1))
                return result
            else:
                # retinaface standard
                faces = self.detector.detect_faces(image)
                result = []
                for key, face in faces.items():
                    area = face['facial_area']
                    x1, y1, x2, y2 = area
                    result.append((x1, y1, x2-x1, y2-y1))
                return result
        except Exception as e:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class MediaPipeDetector(BaseDetector):
    """MediaPipe Face Detection - Google, très rapide."""
    name = "MediaPipe"

    def __init__(self):
        self.detector = None
        self._load()

    def _load(self):
        try:
            import mediapipe as mp
            # Nouvelle API mediapipe (>= 0.10.0)
            if hasattr(mp, 'solutions'):
                self.mp_face = mp.solutions.face_detection
                self.detector = self.mp_face.FaceDetection(
                    model_selection=1,  # 0=court range, 1=full range
                    min_detection_confidence=0.5
                )
                self._use_new_api = False
            else:
                # Nouvelle API avec tasks
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                base_options = python.BaseOptions(model_asset_path='models/blaze_face_short_range.tflite')
                options = vision.FaceDetectorOptions(base_options=base_options)
                self.detector = vision.FaceDetector.create_from_options(options)
                self._use_new_api = True
            print(f"[MediaPipe] Chargé avec succès")
        except Exception as e:
            print(f"[MediaPipe] Non disponible: {e}")
            self.detector = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector is None:
            return []

        try:
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb)

            if not results.detections:
                return []

            boxes = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                boxes.append((max(0, x), max(0, y), bw, bh))

            return boxes
        except Exception as e:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class HaarCascadeDetector(BaseDetector):
    """Haar Cascade - Classique OpenCV."""
    name = "Haar"

    def __init__(self):
        self.detector = None
        self._load()

    def _load(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                self.detector = None
            else:
                print(f"[Haar] Chargé avec succès")
        except Exception as e:
            print(f"[Haar] Erreur: {e}")
            self.detector = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector is None:
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return [tuple(f) for f in faces]
        except Exception as e:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class DLibHOGDetector(BaseDetector):
    """DLib HOG - Robuste."""
    name = "DLib-HOG"

    def __init__(self):
        self.detector = None
        self._load()

    def _load(self):
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            print(f"[DLib-HOG] Chargé avec succès")
        except ImportError:
            print(f"[DLib-HOG] Non disponible. Installer: pip install dlib")
            self.detector = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector is None:
            return []

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)

            result = []
            for face in faces:
                x = face.left()
                y = face.top()
                w = face.width()
                h = face.height()
                result.append((max(0, x), max(0, y), w, h))

            return result
        except Exception as e:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class DLibCNNDetector(BaseDetector):
    """DLib CNN - Plus précis mais plus lent."""
    name = "DLib-CNN"

    def __init__(self):
        self.detector = None
        self._load()

    def _load(self):
        try:
            import dlib
            # Le modèle CNN doit être téléchargé séparément
            model_path = "models/mmod_human_face_detector.dat"
            if os.path.exists(model_path):
                self.detector = dlib.cnn_face_detection_model_v1(model_path)
                print(f"[DLib-CNN] Chargé avec succès")
            else:
                print(f"[DLib-CNN] Modèle non trouvé: {model_path}")
                self.detector = None
        except ImportError:
            print(f"[DLib-CNN] Non disponible. Installer: pip install dlib")
            self.detector = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.detector is None:
            return []

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb, 1)

            result = []
            for face in faces:
                rect = face.rect
                x = rect.left()
                y = rect.top()
                w = rect.width()
                h = rect.height()
                result.append((max(0, x), max(0, y), w, h))

            return result
        except Exception as e:
            return []

    def is_available(self) -> bool:
        return self.detector is not None


class OpenCVDNNDetector(BaseDetector):
    """OpenCV DNN (SSD ResNet) - Bon équilibre."""
    name = "OpenCV-DNN"

    def __init__(self):
        self.net = None
        self._load()

    def _load(self):
        try:
            # Chemins des modèles (pré-entraînés OpenCV)
            base_path = "models"
            prototxt = os.path.join(base_path, "deploy.prototxt")
            model = os.path.join(base_path, "res10_300x300_ssd_iter_140000.caffemodel")

            if os.path.exists(prototxt) and os.path.exists(model):
                self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
                print(f"[OpenCV-DNN] Chargé avec succès")
            else:
                # Télécharger les modèles
                self._download_models(base_path, prototxt, model)
        except Exception as e:
            print(f"[OpenCV-DNN] Erreur: {e}")
            self.net = None

    def _download_models(self, base_path, prototxt, model):
        """Télécharge les modèles SSD."""
        import urllib.request

        os.makedirs(base_path, exist_ok=True)

        urls = {
            prototxt: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            model: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        }

        try:
            for path, url in urls.items():
                if not os.path.exists(path):
                    print(f"[OpenCV-DNN] Téléchargement {os.path.basename(path)}...")
                    urllib.request.urlretrieve(url, path)

            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
            print(f"[OpenCV-DNN] Modèles téléchargés et chargés")
        except Exception as e:
            print(f"[OpenCV-DNN] Échec téléchargement: {e}")
            self.net = None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.net is None:
            return []

        try:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )

            self.net.setInput(blob)
            detections = self.net.forward()

            boxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    boxes.append((max(0, x1), max(0, y1), x2-x1, y2-y1))

            return boxes
        except Exception as e:
            return []

    def is_available(self) -> bool:
        return self.net is not None


# =============================================================================
# BENCHMARK
# =============================================================================

def get_all_detectors() -> List[BaseDetector]:
    """Retourne tous les détecteurs disponibles."""
    detectors = [
        YuNetDetector(),
        MTCNNDetector(),
        RetinaFaceDetector(),
        MediaPipeDetector(),
        HaarCascadeDetector(),
        DLibHOGDetector(),
        # DLibCNNDetector(),  # Désactivé par défaut (lent)
        OpenCVDNNDetector(),
    ]

    # Filtrer les détecteurs disponibles
    available = [d for d in detectors if d.is_available()]
    print(f"\nDétecteurs disponibles: {len(available)}/{len(detectors)}")
    for d in available:
        print(f"  ✓ {d.name}")
    for d in detectors:
        if not d.is_available():
            print(f"  ✗ {d.name} (non disponible)")

    return available


def benchmark_single_image(image: np.ndarray, detectors: List[BaseDetector]) -> Dict:
    """Benchmark sur une seule image."""
    results = {}

    for detector in detectors:
        start = time.time()
        faces = detector.detect(image)
        elapsed = (time.time() - start) * 1000

        results[detector.name] = {
            "faces": len(faces),
            "boxes": faces,
            "time_ms": elapsed
        }

    return results


def draw_results(image: np.ndarray, results: Dict) -> np.ndarray:
    """Dessine les résultats sur l'image."""
    output = image.copy()
    colors = {
        "YuNet": (0, 255, 0),      # Vert
        "MTCNN": (255, 0, 0),      # Bleu
        "RetinaFace": (0, 0, 255), # Rouge
        "MediaPipe": (255, 255, 0),# Cyan
        "Haar": (0, 255, 255),     # Jaune
        "DLib-HOG": (255, 0, 255), # Magenta
        "DLib-CNN": (128, 0, 128), # Violet
        "OpenCV-DNN": (0, 128, 255),# Orange
    }

    y_offset = 25
    for name, data in results.items():
        color = colors.get(name, (255, 255, 255))

        # Dessiner les boxes
        for (x, y, w, h) in data["boxes"]:
            cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)

        # Afficher les stats
        text = f"{name}: {data['faces']} faces | {data['time_ms']:.1f}ms"
        cv2.putText(output, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += 25

    return output


def test_webcam(detectors: List[BaseDetector]):
    """Test en temps réel avec la webcam."""
    print("\n" + "="*70)
    print("MODE WEBCAM")
    print("="*70)
    print("Contrôles:")
    print("  q - Quitter")
    print("  1-9 - Sélectionner un détecteur")
    print("  a - Tous les détecteurs")
    print("  s - Screenshot")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la webcam")
        return

    current_idx = 0  # Premier détecteur par défaut
    show_all = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if show_all:
            results = benchmark_single_image(frame, detectors)
        else:
            results = benchmark_single_image(frame, [detectors[current_idx]])

        display = draw_results(frame, results)

        # Instructions
        mode = "ALL" if show_all else detectors[current_idx].name
        cv2.putText(display, f"Mode: {mode} | q:quit a:all 1-{len(detectors)}:select s:save",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Face Detection Benchmark", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            show_all = not show_all
        elif key == ord('s'):
            filename = f"benchmark_{int(time.time())}.jpg"
            cv2.imwrite(filename, display)
            print(f"Sauvegardé: {filename}")
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(detectors):
                current_idx = idx
                show_all = False
                print(f"Sélectionné: {detectors[current_idx].name}")

    cap.release()
    cv2.destroyAllWindows()


def test_image(image_path: str, detectors: List[BaseDetector]):
    """Test sur une image."""
    print(f"\nChargement: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erreur: impossible de charger l'image")
        return

    print(f"Taille: {image.shape[1]}x{image.shape[0]}")

    results = benchmark_single_image(image, detectors)

    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)

    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Visages détectés: {data['faces']}")
        print(f"  Temps: {data['time_ms']:.1f}ms")

    display = draw_results(image, results)
    cv2.imshow("Results", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_dataset(dataset_path: str, detectors: List[BaseDetector], limit: Optional[int] = None):
    """
    Benchmark complet sur un dataset.

    Pour les datasets d'émotions (où chaque image = 1 visage attendu),
    on mesure le taux de détection.
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        print(f"Erreur: dataset non trouvé: {dataset_dir}")
        return

    # Collecter toutes les images
    images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        images.extend(dataset_dir.rglob(ext))

    if not images:
        print(f"Erreur: aucune image trouvée dans {dataset_dir}")
        return

    # Appliquer la limite
    if limit and limit < len(images):
        import random
        random.shuffle(images)
        images = images[:limit]

    print("\n" + "="*70)
    print("BENCHMARK DATASET")
    print("="*70)
    print(f"Dataset: {dataset_dir}")
    print(f"Images: {len(images)}")
    print(f"Détecteurs: {', '.join(d.name for d in detectors)}")

    # Résultats par détecteur
    stats = {d.name: {
        "detections": 0,      # Images avec au moins 1 visage
        "total_faces": 0,     # Total de visages détectés
        "multi_faces": 0,     # Images avec >1 visage (possibles faux positifs)
        "no_face": 0,         # Images sans visage (possibles faux négatifs)
        "times": [],
        "errors": 0
    } for d in detectors}

    # Tester chaque image
    for i, img_path in enumerate(images):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"\rProgress: {i+1}/{len(images)}", end="", flush=True)

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        for detector in detectors:
            try:
                start = time.time()
                faces = detector.detect(image)
                elapsed = (time.time() - start) * 1000

                stats[detector.name]["times"].append(elapsed)

                num_faces = len(faces)
                stats[detector.name]["total_faces"] += num_faces

                if num_faces == 0:
                    stats[detector.name]["no_face"] += 1
                elif num_faces == 1:
                    stats[detector.name]["detections"] += 1
                else:
                    stats[detector.name]["detections"] += 1
                    stats[detector.name]["multi_faces"] += 1
            except Exception as e:
                stats[detector.name]["errors"] += 1

    print("\n")

    # Afficher les résultats
    print("\n" + "="*70)
    print("RÉSULTATS")
    print("="*70)

    total_images = len(images)

    # Tableau récapitulatif
    print(f"\n{'Détecteur':<15} {'Détection':>10} {'Multi':>8} {'Aucun':>8} {'Temps':>12} {'Erreurs':>8}")
    print("-" * 70)

    results_for_sort = []

    for name, data in stats.items():
        detection_rate = data["detections"] / total_images * 100 if total_images > 0 else 0
        multi_rate = data["multi_faces"] / total_images * 100 if total_images > 0 else 0
        no_face_rate = data["no_face"] / total_images * 100 if total_images > 0 else 0
        avg_time = np.mean(data["times"]) if data["times"] else 0

        results_for_sort.append((name, detection_rate, avg_time, data))

        print(f"{name:<15} {detection_rate:>9.1f}% {multi_rate:>7.1f}% {no_face_rate:>7.1f}% {avg_time:>10.1f}ms {data['errors']:>8}")

    # Classement
    print("\n" + "="*70)
    print("CLASSEMENT")
    print("="*70)

    # Par taux de détection
    sorted_by_detection = sorted(results_for_sort, key=lambda x: x[1], reverse=True)
    print("\nPar taux de détection:")
    for i, (name, rate, _, _) in enumerate(sorted_by_detection, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal} {i}. {name}: {rate:.1f}%")

    # Par vitesse
    sorted_by_speed = sorted(results_for_sort, key=lambda x: x[2])
    print("\nPar vitesse:")
    for i, (name, _, time_ms, _) in enumerate(sorted_by_speed, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal} {i}. {name}: {time_ms:.1f}ms")

    # Score combiné (détection * 0.7 + vitesse_norm * 0.3)
    max_time = max(x[2] for x in results_for_sort) if results_for_sort else 1
    combined_scores = []
    for name, rate, time_ms, _ in results_for_sort:
        speed_score = (1 - time_ms / max_time) * 100 if max_time > 0 else 100
        combined = rate * 0.7 + speed_score * 0.3
        combined_scores.append((name, combined, rate, time_ms))

    sorted_combined = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    print("\nScore combiné (70% détection + 30% vitesse):")
    for i, (name, score, rate, time_ms) in enumerate(sorted_combined, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"  {medal} {i}. {name}: {score:.1f} (détection: {rate:.1f}%, temps: {time_ms:.1f}ms)")

    # Recommandation
    print("\n" + "="*70)
    print("RECOMMANDATION")
    print("="*70)

    best = sorted_combined[0] if sorted_combined else None
    if best:
        print(f"\n  ➜ Meilleur choix global: {best[0]}")
        print(f"    - Taux de détection: {best[2]:.1f}%")
        print(f"    - Temps moyen: {best[3]:.1f}ms")

    fastest = sorted_by_speed[0] if sorted_by_speed else None
    if fastest and fastest[0] != best[0]:
        print(f"\n  ➜ Plus rapide: {fastest[0]} ({fastest[2]:.1f}ms)")

    most_accurate = sorted_by_detection[0] if sorted_by_detection else None
    if most_accurate and most_accurate[0] != best[0]:
        print(f"\n  ➜ Plus précis: {most_accurate[0]} ({most_accurate[1]:.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark des détecteurs de visage")
    parser.add_argument("--image", type=str, help="Chemin vers une image")
    parser.add_argument("--dataset", type=str, help="Chemin vers un dossier dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limiter le nombre d'images")
    parser.add_argument("--detector", type=str, help="Tester un seul détecteur")
    args = parser.parse_args()

    print("="*70)
    print("BENCHMARK DÉTECTEURS DE VISAGE")
    print("="*70)

    # Charger les détecteurs
    detectors = get_all_detectors()

    if not detectors:
        print("\nAucun détecteur disponible!")
        print("Installer les dépendances:")
        print("  pip install mtcnn mediapipe dlib retinaface")
        sys.exit(1)

    # Filtrer si demandé
    if args.detector:
        detectors = [d for d in detectors if d.name.lower() == args.detector.lower()]
        if not detectors:
            print(f"Détecteur '{args.detector}' non trouvé ou non disponible")
            sys.exit(1)

    # Lancer le bon mode
    if args.dataset:
        test_dataset(args.dataset, detectors, args.limit)
    elif args.image:
        test_image(args.image, detectors)
    else:
        test_webcam(detectors)
