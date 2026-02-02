#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script unifié de téléchargement des datasets d'émotions.

Combine les fonctionnalités des scripts bash et Python pour un téléchargement
complet et automatisé de tous les datasets supportés.

Usage:
    python download_datasets.py                      # Liste les datasets
    python download_datasets.py fer2013             # Télécharge FER2013
    python download_datasets.py ferplus             # Télécharge FERPlus labels
    python download_datasets.py --all               # Tous les datasets
    python download_datasets.py --interactive       # Mode interactif

Datasets disponibles:
    - fer2013     : FER2013 (35K images, 7 classes) - Kaggle
    - ferplus     : FERPlus labels améliore FER2013 - GitHub
    - ckplus      : CK+ Extended Cohn-Kanade (593 séquences) - Kaggle
    - affectnet   : AffectNet (1M+ images, 8 classes) - Inscription requise
    - rafdb       : RAF-DB (30K images, 7 classes) - Email requis
    - expw        : ExpW (91K images, 7 classes) - Kaggle

Auteur: ProjectCare Team
"""

import argparse
import sys
import zipfile
import urllib.request
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "data"

# Couleurs pour le terminal
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

    @classmethod
    def disable(cls):
        """Désactive les couleurs (pour Windows ou redirection)."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.CYAN = cls.NC = ''


# Désactiver les couleurs si pas de TTY
if not sys.stdout.isatty():
    Colors.disable()


def print_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def print_success(msg: str):
    print(f"{Colors.GREEN}[OK]{Colors.NC} {msg}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"{Colors.CYAN}{title}{Colors.NC}")
    print('='*60)


# =============================================================================
# Fonctions de téléchargement
# =============================================================================

class DownloadProgressBar:
    """Barre de progression pour urllib."""

    def __init__(self, desc: str = ""):
        self.desc = desc
        self.downloaded = 0
        self.total = 0

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size > 0:
            self.total = total_size
            self.downloaded = block_num * block_size
            pct = min(100, self.downloaded / total_size * 100)
            mb = self.downloaded / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            print(f"\r  {self.desc}: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """
    Télécharge un fichier avec barre de progression.

    Args:
        url: URL du fichier
        dest: Chemin de destination
        desc: Description pour la barre de progression

    Returns:
        True si succès
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        progress = DownloadProgressBar(desc or dest.name)
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()  # Nouvelle ligne après la barre

        return dest.exists() and dest.stat().st_size > 0

    except Exception as e:
        print()
        print_error(f"Échec téléchargement: {e}")
        return False


def download_with_requests(url: str, dest: Path, desc: str = "") -> bool:
    """
    Télécharge avec requests (meilleur pour les gros fichiers).

    Args:
        url: URL du fichier
        dest: Chemin de destination
        desc: Description

    Returns:
        True si succès
    """
    try:
        import requests
    except ImportError:
        print_warning("requests non installé, utilisation de urllib")
        return download_file(url, dest, desc)

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        print_info(f"Téléchargement: {desc or url}")

        response = requests.get(url, stream=True, allow_redirects=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / 1024 / 1024
                        print(f"\r  Progression: {pct:.1f}% ({mb:.1f} MB)", end="", flush=True)

        print()
        return True

    except Exception as e:
        print()
        print_error(f"Échec: {e}")
        return False


def download_from_kaggle(dataset: str, dest_dir: Path) -> bool:
    """
    Télécharge depuis Kaggle.

    Args:
        dataset: Nom du dataset (ex: "msambare/fer2013")
        dest_dir: Dossier de destination

    Returns:
        True si succès
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        print_info(f"Téléchargement depuis Kaggle: {dataset}")
        dest_dir.mkdir(parents=True, exist_ok=True)

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=str(dest_dir), unzip=True)

        print_success("Téléchargé depuis Kaggle")
        return True

    except ImportError:
        print_warning("kaggle non installé")
        print("  Installer: pip install kaggle")
        print("  Configurer: ~/.kaggle/kaggle.json")
        return False

    except Exception as e:
        print_error(f"Erreur Kaggle: {e}")
        return False


def download_from_gdrive(file_id: str, dest: Path) -> bool:
    """
    Télécharge depuis Google Drive.

    Args:
        file_id: ID du fichier Google Drive
        dest: Chemin de destination

    Returns:
        True si succès
    """
    try:
        import gdown

        print_info("Téléchargement depuis Google Drive...")
        dest.parent.mkdir(parents=True, exist_ok=True)

        gdown.download(id=file_id, output=str(dest), quiet=False)
        return dest.exists()

    except ImportError:
        print_warning("gdown non installé, tentative avec urllib...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return download_file(url, dest, "Google Drive")


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """
    Extrait un fichier ZIP.

    Args:
        zip_path: Chemin du fichier ZIP
        dest_dir: Dossier de destination

    Returns:
        True si succès
    """
    try:
        print_info(f"Extraction: {zip_path.name}")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)

        print_success("Extraction terminée")
        return True

    except Exception as e:
        print_error(f"Erreur extraction: {e}")
        return False


def ask_yes_no(question: str, default: bool = True) -> bool:
    """Pose une question oui/non."""
    suffix = "[Y/n]" if default else "[y/N]"

    try:
        response = input(f"{question} {suffix} ").strip().lower()

        if not response:
            return default
        return response in ('y', 'yes', 'o', 'oui')

    except (EOFError, KeyboardInterrupt):
        print()
        return False


# =============================================================================
# Datasets
# =============================================================================

def show_structure(name: str, structure: str):
    """Affiche la structure attendue d'un dataset."""
    print(f"\nStructure attendue pour {name}:")
    print(structure)


def download_fer2013(interactive: bool = False) -> bool:
    """
    Télécharge FER2013.

    FER2013 contient 35,887 images 48x48 en niveaux de gris.
    7 classes: angry, disgust, fear, happy, sad, surprise, neutral
    """
    print_header("FER2013 Dataset")

    dest_dir = DATASETS_DIR / "fer2013"

    # Vérifier si déjà présent
    if (dest_dir / "train").exists() and (dest_dir / "test").exists():
        print_success("FER2013 déjà présent")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: Kaggle
    print_info("Tentative via Kaggle API...")

    if interactive:
        if ask_yes_no("Télécharger via Kaggle CLI?"):
            if download_from_kaggle("msambare/fer2013", dest_dir):
                return True
    else:
        if download_from_kaggle("msambare/fer2013", dest_dir):
            return True

    # Instructions manuelles
    print_warning("Téléchargement automatique échoué")
    print("""
Instructions manuelles:
  1. Créez un compte sur https://www.kaggle.com
  2. Téléchargez: https://www.kaggle.com/datasets/msambare/fer2013
  3. Extrayez dans: {dest_dir}
""".format(dest_dir=dest_dir))

    show_structure("FER2013", """
  fer2013/
  ├── train/
  │   ├── angry/
  │   ├── disgust/
  │   ├── fear/
  │   ├── happy/
  │   ├── sad/
  │   ├── surprise/
  │   └── neutral/
  └── test/
      └── (même structure)
""")

    return False


def download_ferplus(interactive: bool = False) -> bool:
    """
    Télécharge les labels FERPlus (amélioration de FER2013).

    FERPlus fournit des labels crowdsourcés plus précis pour FER2013.
    Téléchargé depuis le repo GitHub de Microsoft.
    """
    _ = interactive  # Non utilisé mais garde l'API cohérente
    print_header("FERPlus Labels")

    dest_dir = DATASETS_DIR / "ferplus"
    dest_dir.mkdir(parents=True, exist_ok=True)

    labels_file = dest_dir / "fer2013new.csv"

    # Vérifier si déjà présent
    if labels_file.exists():
        print_success("Labels FERPlus déjà présents")
        return True

    # Télécharger depuis GitHub
    url = "https://raw.githubusercontent.com/microsoft/FERPlus/master/data/fer2013new.csv"

    print_info("Téléchargement des labels depuis GitHub...")

    if download_file(url, labels_file, "fer2013new.csv"):
        print_success("Labels FERPlus téléchargés")

        print_warning("FERPlus nécessite aussi les images FER2013")
        print("  Téléchargez FER2013 et copiez fer2013.csv dans ferplus/")
        print("  Ou utilisez la structure de dossiers de fer2013")

        return True

    return False


def download_ckplus(interactive: bool = False) -> bool:
    """
    Télécharge CK+ (Extended Cohn-Kanade).

    CK+ contient 593 séquences vidéo de 123 sujets.
    Émotions labellisées sur les dernières frames de chaque séquence.
    """
    print_header("CK+ (Extended Cohn-Kanade) Dataset")

    dest_dir = DATASETS_DIR / "ckplus"

    # Vérifier si déjà présent
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print_success("CK+ déjà présent")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: Kaggle
    print_info("Tentative via Kaggle API...")

    kaggle_datasets = [
        "davilsena/ckdataset",
        "shawon10/ckplus",
    ]

    for dataset in kaggle_datasets:
        if interactive:
            if ask_yes_no(f"Télécharger '{dataset}' via Kaggle?"):
                if download_from_kaggle(dataset, dest_dir):
                    return True
        else:
            if download_from_kaggle(dataset, dest_dir):
                return True

    # Instructions manuelles
    print_warning("Téléchargement automatique échoué")
    print("""
CK+ est disponible sur:
  - Kaggle: https://www.kaggle.com/datasets/davilsena/ckdataset
  - Zenodo: https://zenodo.org/records/11221351
""")

    show_structure("CK+", """
  ckplus/
  ├── cohn-kanade-images/ (ou images/)
  │   └── S005/
  │       └── 001/
  │           └── *.png
  └── Emotion/ (ou labels/)
      └── S005/
          └── 001/
              └── *_emotion.txt
""")

    return False


def download_affectnet(interactive: bool = False) -> bool:
    """
    Instructions pour AffectNet.

    AffectNet est le plus grand dataset d'émotions (1M+ images).
    8 classes avec valence/arousal annotations.
    Nécessite une demande académique.
    """
    _ = interactive  # Non utilisé (téléchargement manuel requis)
    print_header("AffectNet Dataset")

    dest_dir = DATASETS_DIR / "affectnet"

    if dest_dir.exists() and (dest_dir / "train").exists():
        print_success("AffectNet déjà présent")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)

    print_warning("AffectNet nécessite une inscription académique")
    print("""
Instructions:
  1. Visitez: http://mohammadmahoor.com/affectnet/
  2. Remplissez le formulaire de demande (email académique requis)
  3. Attendez l'approbation (quelques jours)
  4. Téléchargez et extrayez dans: {dest_dir}
""".format(dest_dir=dest_dir))

    show_structure("AffectNet", """
  affectnet/
  ├── train/
  │   ├── 0/  (Neutral)
  │   ├── 1/  (Happy)
  │   ├── 2/  (Sad)
  │   ├── 3/  (Surprise)
  │   ├── 4/  (Fear)
  │   ├── 5/  (Disgust)
  │   ├── 6/  (Anger)
  │   └── 7/  (Contempt)
  └── val/
      └── (même structure)
""")

    return False


def download_rafdb(interactive: bool = False) -> bool:
    """
    Instructions pour RAF-DB.

    RAF-DB contient ~30K images annotées en conditions réelles.
    7 classes de base + 12 classes composées.
    Nécessite un email aux auteurs.
    """
    _ = interactive  # Non utilisé (téléchargement manuel requis)
    print_header("RAF-DB (Real-world Affective Faces) Dataset")

    dest_dir = DATASETS_DIR / "rafdb"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        print_success("RAF-DB déjà présent")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)

    print_warning("RAF-DB nécessite un email aux auteurs")
    print("""
Instructions:
  1. Visitez: http://www.whdeng.cn/RAF/model1.html
  2. Envoyez un email pour demander l'accès
  3. Téléchargez avec le mot de passe reçu
  4. Extrayez dans: {dest_dir}
""".format(dest_dir=dest_dir))

    show_structure("RAF-DB", """
  rafdb/
  ├── basic/
  │   ├── Image/
  │   │   ├── aligned/
  │   │   └── original/
  │   └── EmoLabel/
  │       └── list_patition_label.txt
  └── compound/ (optionnel)
      └── ...

  Ou structure simplifiée:
  rafdb/
  ├── train/
  │   ├── 1/  (Surprise)
  │   ├── 2/  (Fear)
  │   ├── 3/  (Disgust)
  │   ├── 4/  (Happiness)
  │   ├── 5/  (Sadness)
  │   ├── 6/  (Anger)
  │   └── 7/  (Neutral)
  └── test/
      └── (même structure)
""")

    return False


def download_expw(interactive: bool = False) -> bool:
    """
    Télécharge ExpW (Expression in-the-Wild).

    ExpW contient 91,793 images annotées extraites du web.
    7 classes d'émotions basiques.
    """
    print_header("ExpW (Expression in-the-Wild) Dataset")

    dest_dir = DATASETS_DIR / "expw"

    if dest_dir.exists() and any(dest_dir.iterdir()):
        print_success("ExpW déjà présent")
        return True

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Option 1: Kaggle
    print_info("Tentative via Kaggle API...")

    if interactive:
        if ask_yes_no("Télécharger via Kaggle CLI?"):
            if download_from_kaggle("shahzadabbas/expression-in-the-wild-expw-dataset", dest_dir):
                return True
    else:
        if download_from_kaggle("shahzadabbas/expression-in-the-wild-expw-dataset", dest_dir):
            return True

    # Instructions manuelles
    print_warning("Téléchargement automatique échoué")
    print("""
ExpW est disponible sur:
  - Kaggle: https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset
  - OpenDataLab: https://opendatalab.com/OpenDataLab/Expression_in-the-Wild
""")

    show_structure("ExpW", """
  expw/
  ├── origin/ (ou images/)
  │   └── *.jpg
  └── label/
      └── label.lst
""")

    return False


# =============================================================================
# Main
# =============================================================================

DATASETS = {
    "fer2013": {
        "func": download_fer2013,
        "desc": "FER2013 (35K images, 7 classes, 48x48)",
        "source": "Kaggle",
    },
    "ferplus": {
        "func": download_ferplus,
        "desc": "FERPlus labels (amélioration de FER2013)",
        "source": "GitHub",
    },
    "ckplus": {
        "func": download_ckplus,
        "desc": "CK+ Extended Cohn-Kanade (593 séquences)",
        "source": "Kaggle",
    },
    "affectnet": {
        "func": download_affectnet,
        "desc": "AffectNet (1M+ images, 8 classes)",
        "source": "Inscription requise",
    },
    "rafdb": {
        "func": download_rafdb,
        "desc": "RAF-DB (30K images, 7 classes)",
        "source": "Email requis",
    },
    "expw": {
        "func": download_expw,
        "desc": "ExpW (91K images, 7 classes)",
        "source": "Kaggle",
    },
}


def list_datasets():
    """Affiche la liste des datasets disponibles."""
    print_header("Datasets disponibles")

    print(f"\n{'Dataset':<12} {'Description':<45} {'Source':<20}")
    print("-" * 77)

    for name, info in DATASETS.items():
        print(f"{name:<12} {info['desc']:<45} {info['source']:<20}")

    print(f"\nDossier des datasets: {DATASETS_DIR}")

    # Vérifier les datasets présents
    print("\nStatut:")
    for name in DATASETS:
        path = DATASETS_DIR / name
        if path.exists() and any(path.iterdir()):
            print_success(f"  {name}: présent")
        else:
            print(f"  {Colors.YELLOW}○{Colors.NC} {name}: non téléchargé")


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargement des datasets d'émotions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python download_datasets.py                  # Liste les datasets
  python download_datasets.py fer2013          # Télécharge FER2013
  python download_datasets.py fer2013 ferplus  # Télécharge plusieurs
  python download_datasets.py --all            # Tous les datasets
  python download_datasets.py -i fer2013       # Mode interactif
"""
    )

    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets à télécharger"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Télécharger tous les datasets"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Lister les datasets disponibles"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Mode interactif (demande confirmation)"
    )

    args = parser.parse_args()

    # Créer le dossier datasets
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Liste les datasets si pas d'argument
    if args.list or (not args.datasets and not args.all):
        list_datasets()
        return 0

    # Déterminer les datasets à télécharger
    if args.all:
        to_download = list(DATASETS.keys())
    else:
        to_download = []
        for name in args.datasets:
            name = name.lower()
            if name in DATASETS:
                to_download.append(name)
            else:
                print_error(f"Dataset inconnu: {name}")
                print(f"  Datasets disponibles: {', '.join(DATASETS.keys())}")
                return 1

    # Télécharger
    results = {}
    for name in to_download:
        info = DATASETS[name]
        success = info["func"](interactive=args.interactive)
        results[name] = success

    # Résumé
    print_header("Résumé")

    for name, success in results.items():
        if success:
            print_success(f"  {name}")
        else:
            print_warning(f"  {name} (action manuelle requise)")

    print(f"\nPour vérifier les datasets:")
    print(f"  python benchmark.py --list")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrompu par l'utilisateur")
        sys.exit(130)
