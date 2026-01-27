# -*- coding: utf-8 -*-
"""
Script de téléchargement du dataset SVIRO (100 images par véhicule).
SVIRO: Synthetic Vehicle Interior Rear Seat Occupancy Dataset
https://sviro.kl.dfki.de/
"""

import os
import sys
import random
import zipfile
import requests
from pathlib import Path
from typing import Optional
import shutil

# Configuration
SVIRO_DIR = Path(__file__).parent.parent / "datasets" / "sviro"
IMAGES_PER_VEHICLE = 100

# URLs des archives (grayscale - plus légères que RGB)
GRAYSCALE_URLS = {
    "bmw_i3": "https://sviro.kl.dfki.de/download/i3/?wpdmdl=378",
    "bmw_x5": "https://sviro.kl.dfki.de/download/bmw-x5/?wpdmdl=387",
    "ford_escape": "https://sviro.kl.dfki.de/download/escape/?wpdmdl=376",
    "hyundai_tucson": "https://sviro.kl.dfki.de/download/hyundai-tucson/?wpdmdl=380",
    "lexus_gsf": "https://sviro.kl.dfki.de/download/lexus/?wpdmdl=379",
    "mercedes_a": "https://sviro.kl.dfki.de/download/grayscale-tesla/?wpdmdl=368",
    "renault_zoe": "https://sviro.kl.dfki.de/download/renault-zoe/?wpdmdl=382",
    "tesla_model3": "https://sviro.kl.dfki.de/download/tesla-model-3/?wpdmdl=383",
    "toyota_hilux": "https://sviro.kl.dfki.de/download/hilux/?wpdmdl=377",
    "vw_tiguan": "https://sviro.kl.dfki.de/download/vw-tiguan/?wpdmdl=384",
}

# URLs des bounding boxes
BBOX_URLS = {
    "bmw_i3": "https://sviro.kl.dfki.de/download/bmw-i3-5/?wpdmdl=456",
    "bmw_x5": "https://sviro.kl.dfki.de/download/bmw-x5-6/?wpdmdl=457",
    "ford_escape": "https://sviro.kl.dfki.de/download/ford-escape-5/?wpdmdl=459",
    "hyundai_tucson": "https://sviro.kl.dfki.de/download/hyundai-tucson-6/?wpdmdl=460",
    "lexus_gsf": "https://sviro.kl.dfki.de/download/lexus-gs-f-5/?wpdmdl=461",
    "mercedes_a": "https://sviro.kl.dfki.de/download/mercedes-class-a-5/?wpdmdl=462",
    "renault_zoe": "https://sviro.kl.dfki.de/download/renault-zoe-6/?wpdmdl=463",
    "tesla_model3": "https://sviro.kl.dfki.de/download/tesla-model-3-6/?wpdmdl=464",
    "toyota_hilux": "https://sviro.kl.dfki.de/download/toyota-hilux-5/?wpdmdl=465",
    "vw_tiguan": "https://sviro.kl.dfki.de/download/vw-tiguan-6/?wpdmdl=466",
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Télécharge un fichier avec barre de progression."""
    try:
        print(f"  Téléchargement: {desc or url}")

        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()

        # Taille totale
        total_size = int(response.headers.get('content-length', 0))

        dest.parent.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progression: {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

        print()
        return True

    except Exception as e:
        print(f"\n  ❌ Erreur: {e}")
        return False


def extract_random_images(zip_path: Path, dest_dir: Path, num_images: int, vehicle: str) -> list:
    """Extrait N images aléatoires d'une archive ZIP."""
    extracted = []

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Lister toutes les images PNG/JPG dans l'archive
            all_images = [
                f for f in zf.namelist()
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                and not f.startswith('__MACOSX')
            ]

            if not all_images:
                print(f"  ⚠️  Aucune image trouvée dans l'archive")
                return []

            # Sélectionner N images aléatoires
            selected = random.sample(all_images, min(num_images, len(all_images)))

            print(f"  Extraction de {len(selected)} images...")

            for img_path in selected:
                # Nom simplifié: vehicle_originalname.png
                original_name = Path(img_path).name
                new_name = f"{vehicle}_{original_name}"
                dest_path = dest_dir / new_name

                # Extraire l'image
                with zf.open(img_path) as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())

                extracted.append(new_name)

            print(f"  ✅ {len(extracted)} images extraites")

    except Exception as e:
        print(f"  ❌ Erreur extraction: {e}")

    return extracted


def download_bbox_annotations(vehicle: str, dest_dir: Path) -> Optional[Path]:
    """Télécharge et extrait les annotations bounding box."""
    url = BBOX_URLS.get(vehicle)
    if not url:
        return None

    zip_path = dest_dir / f"{vehicle}_bbox.zip"

    if not download_file(url, zip_path, f"Bounding boxes {vehicle}"):
        return None

    # Extraire
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Trouver les fichiers CSV/TXT d'annotations
            for f in zf.namelist():
                if f.endswith('.csv') or f.endswith('.txt') or f.endswith('.json'):
                    zf.extract(f, dest_dir / "annotations" / vehicle)

        zip_path.unlink()  # Supprimer le ZIP
        return dest_dir / "annotations" / vehicle

    except Exception as e:
        print(f"  ❌ Erreur extraction annotations: {e}")
        return None


def download_vehicle(vehicle: str, images_dir: Path, temp_dir: Path) -> list:
    """Télécharge les images d'un véhicule."""
    url = GRAYSCALE_URLS.get(vehicle)
    if not url:
        print(f"  ⚠️  URL non trouvée pour {vehicle}")
        return []

    zip_path = temp_dir / f"{vehicle}.zip"

    # Télécharger l'archive
    if not download_file(url, zip_path, f"Images {vehicle}"):
        return []

    # Extraire les images aléatoires
    extracted = extract_random_images(zip_path, images_dir, IMAGES_PER_VEHICLE, vehicle)

    # Supprimer l'archive pour économiser de l'espace
    if zip_path.exists():
        zip_path.unlink()
        print(f"  🗑️  Archive supprimée")

    return extracted


def main():
    """Point d'entrée principal."""
    print("=" * 60)
    print("SVIRO Dataset Downloader")
    print(f"Téléchargement de {IMAGES_PER_VEHICLE} images par véhicule")
    print("=" * 60)

    # Créer les dossiers
    images_dir = SVIRO_DIR / "images"
    annotations_dir = SVIRO_DIR / "annotations"
    temp_dir = SVIRO_DIR / "temp"

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    all_images = {}
    vehicles = list(GRAYSCALE_URLS.keys())

    print(f"\n📦 {len(vehicles)} véhicules à télécharger\n")

    for i, vehicle in enumerate(vehicles, 1):
        print(f"\n[{i}/{len(vehicles)}] {vehicle.upper()}")
        print("-" * 40)

        # Télécharger les images
        images = download_vehicle(vehicle, images_dir, temp_dir)
        all_images[vehicle] = images

        # Télécharger les annotations
        download_bbox_annotations(vehicle, SVIRO_DIR)

    # Nettoyer le dossier temp
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Créer un fichier d'index
    index_file = SVIRO_DIR / "index.txt"
    with open(index_file, 'w') as f:
        for vehicle, images in all_images.items():
            for img in images:
                f.write(f"{img}\n")

    # Résumé
    print("\n" + "=" * 60)
    print("✅ TÉLÉCHARGEMENT TERMINÉ")
    print("=" * 60)

    total_images = sum(len(imgs) for imgs in all_images.values())
    print(f"\n📊 Résumé:")
    print(f"   - Images totales: {total_images}")
    print(f"   - Véhicules: {len(vehicles)}")
    print(f"   - Dossier: {SVIRO_DIR}")

    for vehicle, images in all_images.items():
        status = "✅" if len(images) >= IMAGES_PER_VEHICLE else f"⚠️ ({len(images)})"
        print(f"   - {vehicle}: {status}")


if __name__ == "__main__":
    main()
