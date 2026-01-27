# -*- coding: utf-8 -*-
"""
Script de téléchargement du dataset WIDER FACE pour benchmark de détection de visage.

Usage (depuis la racine du projet):
    python benchmarks/face_detection/scripts/download_datasets.py              # Télécharge WIDER FACE + annotations
    python benchmarks/face_detection/scripts/download_datasets.py --list       # Voir le statut
    python benchmarks/face_detection/scripts/download_datasets.py --annotations # Télécharger uniquement les annotations
"""

import sys
import zipfile
import urllib.request
from pathlib import Path
import ssl

# Ignorer les erreurs SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Chemins
DATASETS_DIR = Path(__file__).parent.parent / "datasets"
ANNOTATIONS_DIR = Path(__file__).parent.parent / "annotations"

# URLs
WIDER_FACE_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
ANNOTATIONS_URL = "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"


def download_file(url: str, dest_path: Path) -> bool:
    """Télécharge un fichier avec barre de progression."""

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded = count * block_size / (1024 * 1024)
            total = total_size / (1024 * 1024)
            bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
            sys.stdout.write(f"\r  [{bar}] {percent}% ({downloaded:.1f}/{total:.1f} MB)")
            sys.stdout.flush()

    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  Erreur: {e}")
        return False


def extract_zip(archive_path: Path, dest_dir: Path) -> bool:
    """Extrait une archive ZIP."""
    try:
        print("  Extraction...")
        with zipfile.ZipFile(archive_path, 'r') as z:
            z.extractall(dest_dir)
        archive_path.unlink()
        print("  Extraction terminée")
        return True
    except Exception as e:
        print(f"  Erreur extraction: {e}")
        return False


def download_wider_face() -> bool:
    """Télécharge le dataset WIDER FACE."""

    dest_dir = DATASETS_DIR / "wider_face"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier si déjà téléchargé
    existing = list(dest_dir.rglob("*.jpg"))
    if len(existing) > 100:
        print(f"  WIDER FACE déjà présent ({len(existing)} images)")
        return True

    print("\n  Téléchargement de WIDER FACE (~850MB)...")
    archive_path = dest_dir / "WIDER_val.zip"

    if download_file(WIDER_FACE_URL, archive_path):
        if extract_zip(archive_path, dest_dir):
            images = list(dest_dir.rglob("*.jpg"))
            print(f"  {len(images)} images extraites")
            return True
    return False


def download_annotations() -> bool:
    """Télécharge les annotations WIDER FACE."""

    annotations_dir = ANNOTATIONS_DIR / "wider_face_split"
    gt_file = annotations_dir / "wider_face_val_bbx_gt.txt"

    if gt_file.exists():
        print("  Annotations déjà présentes")
        return True

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = ANNOTATIONS_DIR / "wider_face_split.zip"

    print("\n  Téléchargement des annotations...")
    if download_file(ANNOTATIONS_URL, archive_path):
        return extract_zip(archive_path, ANNOTATIONS_DIR)
    return False


def list_status():
    """Affiche le statut des téléchargements."""

    print("\n" + "="*60)
    print("  STATUT WIDER FACE")
    print("="*60)

    # Dataset
    dest_dir = DATASETS_DIR / "wider_face"
    images = list(dest_dir.rglob("*.jpg")) if dest_dir.exists() else []
    status = f"[OK] {len(images)} images" if images else "[--] Non téléchargé"
    print(f"\n  Dataset: {status}")

    # Annotations
    gt_file = ANNOTATIONS_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    status = "[OK] Présentes" if gt_file.exists() else "[--] Non téléchargées"
    print(f"  Annotations: {status}")

    print("\n  Description: 3,226 images, 393K visages annotés")
    print("  Les annotations contiennent: position, blur, occlusion, pose")
    print("  → Permet de calculer les métriques Easy/Medium/Hard automatiquement")


def download_all():
    """Télécharge tout."""

    print("\n" + "="*60)
    print("  TÉLÉCHARGEMENT WIDER FACE")
    print("="*60)

    success_dataset = download_wider_face()
    success_annotations = download_annotations()

    print("\n" + "="*60)
    print("  RÉSUMÉ")
    print("="*60)
    print(f"  Dataset:     {'[OK]' if success_dataset else '[ERREUR]'}")
    print(f"  Annotations: {'[OK]' if success_annotations else '[ERREUR]'}")

    if success_dataset and success_annotations:
        print("\n  Prêt pour le benchmark !")
        print("  Lancez: python benchmarks/face_detection/benchmark_pro.py --fast --limit 500")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Téléchargement WIDER FACE")
    parser.add_argument("--list", action="store_true", help="Voir le statut")
    parser.add_argument("--annotations", action="store_true", help="Télécharger uniquement les annotations")
    args = parser.parse_args()

    if args.list:
        list_status()
    elif args.annotations:
        download_annotations()
    else:
        download_all()
