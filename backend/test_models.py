# -*- coding: utf-8 -*-
"""
Script de test pour comparer les modeles de detection d'emotions.
Compare: FER (actuel), HSEmotion, DeepFace

Usage:
    python test_models.py                              # Test avec webcam
    python test_models.py --image photo.jpg            # Test avec une image
    python test_models.py --dataset data/my_dataset    # Test sur un dataset
    python test_models.py --dataset data/my_dataset --model fer  # Un seul modele
    python test_models.py --dataset data/my_dataset --limit 100  # Limiter a 100 images
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# =============================================================================
# TEST FER (modele actuel)
# =============================================================================
def test_fer(image):
    """Test avec le modele FER actuel."""
    try:
        from fer import FER
        detector = FER(mtcnn=True)

        start = time.time()
        results = detector.detect_emotions(image)
        elapsed = (time.time() - start) * 1000

        if results:
            emotions = results[0]['emotions']
            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant]
            return {
                "model": "FER",
                "emotion": dominant,
                "confidence": confidence,
                "all_scores": emotions,
                "time_ms": elapsed,
                "success": True
            }
        return {"model": "FER", "success": False, "error": "No face detected"}
    except Exception as e:
        return {"model": "FER", "success": False, "error": str(e)}


# =============================================================================
# TEST HSEMOTION
# =============================================================================
def test_hsemotion(image, face_box=None):
    """Test avec HSEmotion."""
    try:
        from hsemotion.facial_emotions import HSEmotionRecognizer

        # Initialiser le modele
        model = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')

        # HSEmotion a besoin d'un visage croppe
        # Si pas de box fournie, utiliser FER pour detecter
        if face_box is None:
            from fer import FER
            detector = FER(mtcnn=True)
            faces = detector.detect_emotions(image)
            if not faces:
                return {"model": "HSEmotion", "success": False, "error": "No face detected"}
            face_box = faces[0]['box']

        x, y, w, h = face_box

        # Ajouter une marge
        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        face_crop = image[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        start = time.time()
        emotion, scores = model.predict_emotions(face_rgb, logits=True)
        elapsed = (time.time() - start) * 1000

        # Classes HSEmotion
        classes = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
        scores_dict = {c: float(s) for c, s in zip(classes, scores)}

        return {
            "model": "HSEmotion",
            "emotion": emotion,
            "confidence": max(scores),
            "all_scores": scores_dict,
            "time_ms": elapsed,
            "success": True
        }
    except ImportError:
        return {"model": "HSEmotion", "success": False, "error": "hsemotion not installed. Run: pip install hsemotion"}
    except Exception as e:
        return {"model": "HSEmotion", "success": False, "error": str(e)}


# =============================================================================
# TEST DEEPFACE
# =============================================================================
def test_deepface(image):
    """Test avec DeepFace."""
    try:
        from deepface import DeepFace

        # Sauvegarder temporairement l'image (DeepFace prefere les chemins)
        temp_path = "/tmp/test_emotion.jpg"
        cv2.imwrite(temp_path, image)

        start = time.time()
        result = DeepFace.analyze(
            img_path=temp_path,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )
        elapsed = (time.time() - start) * 1000

        if result:
            data = result[0] if isinstance(result, list) else result
            return {
                "model": "DeepFace",
                "emotion": data['dominant_emotion'],
                "confidence": data['emotion'][data['dominant_emotion']] / 100,
                "all_scores": {k: v/100 for k, v in data['emotion'].items()},
                "time_ms": elapsed,
                "success": True
            }
        return {"model": "DeepFace", "success": False, "error": "No result"}
    except ImportError:
        return {"model": "DeepFace", "success": False, "error": "deepface not installed. Run: pip install deepface"}
    except Exception as e:
        return {"model": "DeepFace", "success": False, "error": str(e)}


# =============================================================================
# AFFICHAGE DES RESULTATS
# =============================================================================
def print_results(results):
    """Affiche les resultats de maniere formatee."""
    print("\n" + "=" * 70)
    print("RESULTATS DE LA COMPARAISON")
    print("=" * 70)

    for r in results:
        print(f"\n--- {r['model']} ---")
        if r['success']:
            print(f"  Emotion:    {r['emotion'].upper()}")
            print(f"  Confidence: {r['confidence']:.2%}")
            print(f"  Temps:      {r['time_ms']:.1f}ms")
            print(f"  Scores:")
            # Trier par score decroissant
            sorted_scores = sorted(r['all_scores'].items(), key=lambda x: -x[1])
            for emotion, score in sorted_scores[:5]:  # Top 5
                bar = "█" * int(score * 20)
                print(f"    {emotion:12} {score:5.2%} {bar}")
        else:
            print(f"  ERREUR: {r['error']}")

    print("\n" + "=" * 70)


def draw_results_on_image(image, results):
    """Dessine les resultats sur l'image."""
    img_display = image.copy()
    y_offset = 30

    for r in results:
        if r['success']:
            color = (0, 255, 0)  # Vert
            text = f"{r['model']}: {r['emotion']} ({r['confidence']:.0%}) - {r['time_ms']:.0f}ms"
        else:
            color = (0, 0, 255)  # Rouge
            text = f"{r['model']}: ERROR"

        cv2.putText(img_display, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 30

    return img_display


# =============================================================================
# MODE WEBCAM
# =============================================================================
def test_webcam():
    """Test en temps reel avec la webcam."""
    print("Demarrage de la webcam...")
    print("Appuyez sur:")
    print("  'q' - Quitter")
    print("  's' - Screenshot + comparaison complete")
    print("  '1' - FER uniquement")
    print("  '2' - HSEmotion uniquement")
    print("  '3' - DeepFace uniquement")
    print("  'a' - Tous les modeles")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la webcam")
        return

    current_model = "fer"  # Modele par defaut
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Tester le modele selectionne
        if current_model == "fer":
            last_result = [test_fer(frame)]
        elif current_model == "hsemotion":
            last_result = [test_hsemotion(frame)]
        elif current_model == "deepface":
            last_result = [test_deepface(frame)]
        elif current_model == "all":
            # Tous les modeles (plus lent)
            fer_result = test_fer(frame)
            face_box = None
            if fer_result['success']:
                from fer import FER
                detector = FER(mtcnn=True)
                faces = detector.detect_emotions(frame)
                if faces:
                    face_box = faces[0]['box']

            last_result = [
                fer_result,
                test_hsemotion(frame, face_box),
                test_deepface(frame)
            ]

        # Afficher les resultats sur l'image
        if last_result:
            frame = draw_results_on_image(frame, last_result)

        # Instructions
        cv2.putText(frame, f"Mode: {current_model.upper()} | q:quit s:screenshot 1:FER 2:HS 3:DF a:all",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Test Models - Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_model = "fer"
            print("\n> Mode: FER")
        elif key == ord('2'):
            current_model = "hsemotion"
            print("\n> Mode: HSEmotion")
        elif key == ord('3'):
            current_model = "deepface"
            print("\n> Mode: DeepFace")
        elif key == ord('a'):
            current_model = "all"
            print("\n> Mode: ALL (plus lent)")
        elif key == ord('s'):
            # Screenshot + comparaison complete
            print("\n> Screenshot - Comparaison complete...")
            results = [
                test_fer(frame),
                test_hsemotion(frame),
                test_deepface(frame)
            ]
            print_results(results)
            cv2.imwrite("screenshot_comparison.jpg", frame)
            print("Image sauvegardee: screenshot_comparison.jpg")

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# MODE IMAGE
# =============================================================================
def test_image(image_path):
    """Test sur une image."""
    print(f"Chargement de l'image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erreur: impossible de charger {image_path}")
        return

    print("Test des modeles...")
    results = [
        test_fer(image),
        test_hsemotion(image),
        test_deepface(image)
    ]

    print_results(results)

    # Afficher l'image avec les resultats
    img_display = draw_results_on_image(image, results)
    cv2.imshow("Results", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# MODE DATASET
# =============================================================================

# Mapping des emotions HSEmotion -> standard
HSEMOTION_MAP = {
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "contempt": "disgust",
}

# Mapping des emotions DeepFace -> standard (deja ok)
DEEPFACE_MAP = {}


def normalize_emotion(emotion, model):
    """Normalise le nom de l'emotion selon le modele."""
    if model == "HSEmotion":
        return HSEMOTION_MAP.get(emotion, emotion)
    return emotion


def print_confusion_matrix(confusion, emotions):
    """Affiche la matrice de confusion."""
    print(f"\n  Matrice de confusion (lignes=vrai, colonnes=predit):")

    # Header
    print(f"  {'':12}", end="")
    for e in emotions:
        print(f"{e[:3]:>5}", end="")
    print()

    # Lignes
    for true_emotion in emotions:
        print(f"  {true_emotion:<12}", end="")
        for pred_emotion in emotions:
            count = confusion[true_emotion][pred_emotion]
            print(f"{count:>5}", end="")
        print()


def test_dataset(dataset_path, model_filter=None, limit=None):
    """
    Test les modeles sur un dataset complet.

    Args:
        dataset_path: Chemin vers le dossier du dataset (ex: data/my_dataset)
        model_filter: "fer", "hsemotion", "deepface" ou None pour tous
        limit: Nombre maximum d'images a tester (None = toutes)
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        print(f"ERREUR: Dataset non trouve: {dataset_dir}")
        return

    # Emotions attendues
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # Charger toutes les images
    samples = []
    for emotion in emotions:
        emotion_dir = dataset_dir / emotion
        if emotion_dir.exists():
            for img_path in emotion_dir.glob("*.jpg"):
                samples.append((str(img_path), emotion))
            for img_path in emotion_dir.glob("*.png"):
                samples.append((str(img_path), emotion))

    if not samples:
        print(f"ERREUR: Aucune image trouvee dans {dataset_dir}")
        return

    # Appliquer la limite si specifiee (echantillonnage equilibre)
    if limit and limit < len(samples):
        samples_by_class = defaultdict(list)
        for s in samples:
            samples_by_class[s[1]].append(s)

        limited_samples = []
        per_class = limit // len(emotions)
        for emotion in emotions:
            limited_samples.extend(samples_by_class[emotion][:per_class])
        samples = limited_samples
        print(f"\n[LIMITE] Echantillonnage a {len(samples)} images ({per_class}/classe)")

    print(f"\nDataset: {dataset_dir}")
    print(f"Total images: {len(samples)}")

    # Distribution
    dist = defaultdict(int)
    for _, emotion in samples:
        dist[emotion] += 1
    print("\nDistribution:")
    for emotion in emotions:
        if dist[emotion] > 0:
            print(f"  {emotion:10}: {dist[emotion]:4} images")

    # Determiner quels modeles tester
    models_to_test = []
    if model_filter is None or model_filter == "fer":
        models_to_test.append(("FER", test_fer))
    if model_filter is None or model_filter == "hsemotion":
        models_to_test.append(("HSEmotion", test_hsemotion))
    if model_filter is None or model_filter == "deepface":
        models_to_test.append(("DeepFace", test_deepface))

    # Resultats par modele
    all_results = {}

    for model_name, test_func in models_to_test:
        print(f"\n{'='*70}")
        print(f"TEST: {model_name}")
        print(f"{'='*70}")

        results = {
            "correct": 0,
            "total": 0,
            "no_detection": 0,
            "per_class": {e: {"correct": 0, "total": 0} for e in emotions},
            "confusion": defaultdict(lambda: defaultdict(int)),
            "times": []
        }

        for i, (img_path, true_emotion) in enumerate(samples):
            # Afficher progression
            if (i + 1) % 10 == 0 or i == 0:
                print(f"\r  Processing: {i+1}/{len(samples)}", end="", flush=True)

            # Charger l'image
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Tester
            start = time.time()
            result = test_func(image)
            elapsed = time.time() - start
            results["times"].append(elapsed)

            results["total"] += 1
            results["per_class"][true_emotion]["total"] += 1

            if not result["success"]:
                results["no_detection"] += 1
                continue

            # Normaliser l'emotion predite
            pred_emotion = normalize_emotion(result["emotion"], model_name)

            # Verifier si correct
            if pred_emotion == true_emotion:
                results["correct"] += 1
                results["per_class"][true_emotion]["correct"] += 1

            # Matrice de confusion
            results["confusion"][true_emotion][pred_emotion] += 1

        print()  # Nouvelle ligne apres progression

        # Afficher resultats
        total = results["total"]
        correct = results["correct"]
        no_det = results["no_detection"]

        accuracy = correct / total * 100 if total > 0 else 0
        detection_rate = (total - no_det) / total * 100 if total > 0 else 0
        avg_time = np.mean(results["times"]) * 1000 if results["times"] else 0

        print(f"\n  Accuracy:       {accuracy:.1f}%")
        print(f"  Detection rate: {detection_rate:.1f}%")
        print(f"  Temps moyen:    {avg_time:.1f}ms/image")

        # Par classe
        print(f"\n  Par emotion:")
        print(f"  {'Emotion':<12} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8}")

        for emotion in emotions:
            data = results["per_class"][emotion]
            if data["total"] > 0:
                acc = data["correct"] / data["total"] * 100
                print(f"  {emotion:<12} {acc:>9.1f}% {data['correct']:>8} {data['total']:>8}")

        # Matrice de confusion
        print_confusion_matrix(results["confusion"], emotions)

        all_results[model_name] = results

    # Comparaison finale si plusieurs modeles
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARAISON FINALE")
        print(f"{'='*70}")

        print(f"\n{'Modele':<12} {'Accuracy':>10} {'Detection':>12} {'Temps':>12}")
        print(f"{'-'*12} {'-'*10} {'-'*12} {'-'*12}")

        for model_name, results in all_results.items():
            total = results["total"]
            correct = results["correct"]
            no_det = results["no_detection"]

            accuracy = correct / total * 100 if total > 0 else 0
            detection = (total - no_det) / total * 100 if total > 0 else 0
            avg_time = np.mean(results["times"]) * 1000 if results["times"] else 0

            print(f"{model_name:<12} {accuracy:>9.1f}% {detection:>11.1f}% {avg_time:>10.1f}ms")

        # Meilleur par emotion
        print(f"\n  Meilleur modele par emotion:")
        for emotion in emotions:
            best_model = None
            best_acc = -1
            for model_name, results in all_results.items():
                data = results["per_class"][emotion]
                if data["total"] > 0:
                    acc = data["correct"] / data["total"] * 100
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model_name
            if best_model:
                print(f"    {emotion:<12} -> {best_model} ({best_acc:.1f}%)")

    print(f"\n{'='*70}")
    print("Test termine!")
    print(f"{'='*70}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test des modeles de detection d'emotions")
    parser.add_argument("--image", type=str, help="Chemin vers une image")
    parser.add_argument("--dataset", type=str, help="Chemin vers un dossier dataset")
    parser.add_argument("--model", type=str, choices=["fer", "hsemotion", "deepface"],
                        help="Tester un seul modele")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter le nombre d'images (echantillonnage equilibre)")
    args = parser.parse_args()

    print("=" * 70)
    print("TEST DES MODELES DE DETECTION D'EMOTIONS")
    print("=" * 70)
    print("\nModeles disponibles:")
    print("  1. FER       - Entraine sur FER2013")
    print("  2. HSEmotion - Entraine sur AffectNet")
    print("  3. DeepFace  - Multi-backend")
    print()

    if args.dataset:
        test_dataset(args.dataset, model_filter=args.model, limit=args.limit)
    elif args.image:
        test_image(args.image)
    else:
        test_webcam()
