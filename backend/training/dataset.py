# -*- coding: utf-8 -*-
"""
Dataset loaders pour FER2013 et L3.
Gere le chargement et la preparation des donnees.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

from training.config import (
    FER2013_DIR, L3_DIR, FER_CLASSES, L3_CLASSES,
    FER_BATCH_SIZE, L3_BATCH_SIZE, VAL_SPLIT, TEST_SPLIT,
    GRAYSCALE_TO_RGB, EMOTION_TO_COMFORT, SEED
)
from training.augmentation import get_train_transforms, get_val_transforms


class FER2013Dataset(Dataset):
    """
    Dataset pour FER2013.
    Le fichier fer2013.csv contient les pixels en format string.
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        """
        Args:
            csv_path: Chemin vers fer2013.csv
            split: "train", "val", ou "test"
            transform: Transformations a appliquer
        """
        self.transform = transform
        self.split = split

        # Charger le CSV
        df = pd.read_csv(csv_path)

        # FER2013 a une colonne "Usage" pour le split
        if "Usage" in df.columns:
            if split == "train":
                df = df[df["Usage"] == "Training"]
            elif split == "val":
                df = df[df["Usage"] == "PublicTest"]
            elif split == "test":
                df = df[df["Usage"] == "PrivateTest"]

        self.labels = df["emotion"].values
        self.pixels = df["pixels"].values

        print(f"FER2013 {split}: {len(self)} images")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Convertir les pixels string en array numpy
        pixels = np.array(self.pixels[idx].split(), dtype=np.uint8)
        image = pixels.reshape(48, 48)

        # Convertir en PIL Image
        image = Image.fromarray(image, mode="L")

        # Convertir grayscale en RGB si necessaire
        if GRAYSCALE_TO_RGB:
            image = image.convert("RGB")

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label


class FER2013ImageFolderDataset(Dataset):
    """
    Dataset pour FER2013 organise en dossiers (apres preprocessing).
    Structure attendue:
        fer2013/
            train/
                angry/
                disgust/
                ...
            test/
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = FER_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        print(f"FER2013 ImageFolder {split}: {len(self)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path)
        if GRAYSCALE_TO_RGB and image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class L3Dataset(Dataset):
    """
    Dataset pour L3 (inconfort thermique).
    Structure attendue:
        l3/
            confortable/
                img001.jpg
                ...
            inconfortable/
                img001.jpg
                ...
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = L3_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))

        print(f"L3 Dataset: {len(self)} images")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class FER2013ToL3Dataset(Dataset):
    """
    Dataset qui mappe FER2013 vers les classes L3 (confortable/inconfortable).
    Utile si on n'a pas de dataset L3 specifique.
    """

    def __init__(
        self,
        fer_dataset: FER2013Dataset,
        transform: Optional[Callable] = None
    ):
        self.fer_dataset = fer_dataset
        self.transform = transform

        # Mapping emotion index -> L3 class index
        # FER classes: ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        # L3 classes: ["confortable", "inconfortable"]
        self.emotion_to_l3 = {}
        for i, emotion in enumerate(FER_CLASSES):
            comfort = EMOTION_TO_COMFORT[emotion]
            self.emotion_to_l3[i] = L3_CLASSES.index(comfort)

    def __len__(self) -> int:
        return len(self.fer_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, fer_label = self.fer_dataset[idx]
        l3_label = self.emotion_to_l3[fer_label]
        return image, l3_label


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_fer2013_dataloaders(
    root_dir: str = None,
    batch_size: int = FER_BATCH_SIZE,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cree les dataloaders pour FER2013 (format ImageFolder).
    Le dataset doit etre organise en dossiers:
        fer2013/train/angry/, fer2013/train/happy/, etc.
        fer2013/test/angry/, fer2013/test/happy/, etc.
    """
    if root_dir is None:
        root_dir = FER2013_DIR

    # Charger train et test
    train_full = FER2013ImageFolderDataset(root_dir, split="train", transform=get_train_transforms())
    test_dataset = FER2013ImageFolderDataset(root_dir, split="test", transform=get_val_transforms())

    # Creer validation split depuis train (10%)
    train_size = int(len(train_full) * (1 - VAL_SPLIT))
    val_size = len(train_full) - train_size

    train_dataset, val_dataset = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Note: val_dataset utilise les memes transforms que train (via train_full)
    # On cree un wrapper pour utiliser val_transforms
    class ValWrapper(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
            image = Image.open(img_path)
            if GRAYSCALE_TO_RGB and image.mode != "RGB":
                image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label

    val_dataset = ValWrapper(val_dataset, get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def get_l3_dataloaders(
    root_dir: str = None,
    batch_size: int = L3_BATCH_SIZE,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cree les dataloaders pour L3.
    """
    if root_dir is None:
        root_dir = L3_DIR

    full_dataset = L3Dataset(root_dir, transform=None)

    # Split train/val/test
    total_size = len(full_dataset)
    test_size = int(total_size * TEST_SPLIT)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Appliquer les transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    test_dataset.dataset.transform = get_val_transforms()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test
    print("Testing datasets...")

    csv_path = FER2013_DIR / "fer2013.csv"
    if csv_path.exists():
        train_loader, val_loader, test_loader = get_fer2013_dataloaders()
        print(f"FER2013 - Train batches: {len(train_loader)}")
        print(f"FER2013 - Val batches: {len(val_loader)}")
        print(f"FER2013 - Test batches: {len(test_loader)}")
    else:
        print(f"FER2013 CSV not found at {csv_path}")
        print("Download from: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge")
