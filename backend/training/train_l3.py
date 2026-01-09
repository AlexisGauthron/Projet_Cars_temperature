# -*- coding: utf-8 -*-
"""
Script de fine-tuning sur L3 (inconfort thermique).
Adapte le modele FER2013 pour la classification confortable/inconfortable.

Usage:
    python training/train_l3.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ajouter le dossier backend au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    L3_DIR, FER_MODEL_PATH, L3_MODEL_PATH, L3_CLASSES, L3_NUM_CLASSES,
    L3_BATCH_SIZE, L3_LEARNING_RATE, L3_EPOCHS, L3_WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, LOG_INTERVAL,
    get_device, ensure_dirs
)
from training.model import create_l3_model, count_parameters
from training.dataset import get_l3_dataloaders, FER2013ToL3Dataset, FER2013Dataset
from training.augmentation import get_train_transforms, get_val_transforms


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """
    Entraine le modele pour une epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % LOG_INTERVAL == 0:
            pbar.set_postfix({
                "loss": f"{running_loss / (batch_idx + 1):.4f}",
                "acc": f"{100. * correct / total:.2f}%"
            })

    return {
        "loss": running_loss / len(dataloader),
        "accuracy": 100. * correct / total
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Valide le modele.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return {
        "loss": running_loss / len(dataloader),
        "accuracy": 100. * correct / total
    }


def train_l3(use_fer_as_l3: bool = False):
    """
    Fine-tuning sur le dataset L3.

    Args:
        use_fer_as_l3: Si True, utilise FER2013 mappe en classes L3
                       (utile si pas de dataset L3 disponible)
    """
    print("=" * 60)
    print("FINE-TUNING L3 - Inconfort Thermique")
    print("=" * 60)

    # Setup
    ensure_dirs()
    device = get_device()
    print(f"\nDevice: {device}")

    # Dataset
    if use_fer_as_l3:
        print("\n[MODE] Utilisation de FER2013 mappe en classes L3")
        from training.config import FER2013_DIR

        csv_path = FER2013_DIR / "fer2013.csv"
        if not csv_path.exists():
            print(f"ERREUR: FER2013 non trouve a {csv_path}")
            return

        # Creer datasets FER2013 et les mapper vers L3
        from torch.utils.data import DataLoader

        train_fer = FER2013Dataset(csv_path, split="train", transform=get_train_transforms())
        val_fer = FER2013Dataset(csv_path, split="val", transform=get_val_transforms())
        test_fer = FER2013Dataset(csv_path, split="test", transform=get_val_transforms())

        train_dataset = FER2013ToL3Dataset(train_fer)
        val_dataset = FER2013ToL3Dataset(val_fer)
        test_dataset = FER2013ToL3Dataset(test_fer)

        train_loader = DataLoader(train_dataset, batch_size=L3_BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=L3_BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=L3_BATCH_SIZE, shuffle=False, num_workers=4)

    else:
        # Verifier si dataset L3 existe
        l3_confort = L3_DIR / "confortable"
        l3_inconfort = L3_DIR / "inconfortable"

        confort_count = len(list(l3_confort.glob("*"))) if l3_confort.exists() else 0
        inconfort_count = len(list(l3_inconfort.glob("*"))) if l3_inconfort.exists() else 0

        if confort_count == 0 and inconfort_count == 0:
            print(f"\nATTENTION: Dataset L3 vide!")
            print(f"  - {l3_confort}: {confort_count} images")
            print(f"  - {l3_inconfort}: {inconfort_count} images")
            print("\nOptions:")
            print("  1. Ajoute des images dans ces dossiers")
            print("  2. Lance avec --use-fer-as-l3 pour simuler avec FER2013")
            print("\nExemple: python training/train_l3.py --use-fer-as-l3")
            return

        print(f"\nDataset L3:")
        print(f"  - Confortable: {confort_count} images")
        print(f"  - Inconfortable: {inconfort_count} images")

        train_loader, val_loader, test_loader = get_l3_dataloaders()

    # Model
    print("\nCreation du modele L3...")

    # Charger depuis le modele FER2013 si disponible
    fer_checkpoint = None
    if FER_MODEL_PATH.exists():
        fer_checkpoint = str(FER_MODEL_PATH)
        print(f"Chargement du backbone depuis: {fer_checkpoint}")

    model = create_l3_model(fer_checkpoint=fer_checkpoint)
    model.to(device)

    params = count_parameters(model)
    print(f"Parametres: {params['total']:,} total, {params['trainable']:,} entrainables")

    # Geler le backbone pour les premieres epochs (optionnel)
    # model.freeze_backbone()

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=L3_LEARNING_RATE,
        weight_decay=L3_WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=L3_EPOCHS)

    # Training loop
    print(f"\nDemarrage du fine-tuning...")
    print(f"Epochs: {L3_EPOCHS}")
    print(f"Batch size: {L3_BATCH_SIZE}")
    print(f"Learning rate: {L3_LEARNING_RATE}")
    print("-" * 60)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(L3_EPOCHS):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(f"\nEpoch {epoch+1}/{L3_EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": L3_CLASSES
            }
            torch.save(checkpoint, L3_MODEL_PATH)
            print(f"  [SAVED] Nouveau meilleur modele! Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping apres {epoch+1} epochs")
                break

    # Final evaluation
    print("\n" + "=" * 60)
    print("EVALUATION FINALE")
    print("=" * 60)

    checkpoint = torch.load(L3_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

    print(f"\nModele sauvegarde dans: {L3_MODEL_PATH}")
    print("Fine-tuning termine!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tuning L3")
    parser.add_argument(
        "--use-fer-as-l3",
        action="store_true",
        help="Utiliser FER2013 mappe en classes L3 (si pas de dataset L3)"
    )
    args = parser.parse_args()

    train_l3(use_fer_as_l3=args.use_fer_as_l3)
