# -*- coding: utf-8 -*-
"""
Script d'entrainement sur FER2013.
Fine-tuning EfficientNet-B0 pour la detection d'emotions (7 classes).

Usage:
    python training/train_fer.py
"""

import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

# Ajouter le dossier backend au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    FER2013_DIR, FER_MODEL_PATH, FER_CLASSES, FER_NUM_CLASSES,
    FER_BATCH_SIZE, FER_LEARNING_RATE, FER_EPOCHS, FER_WEIGHT_DECAY,
    WARMUP_EPOCHS, EARLY_STOPPING_PATIENCE, LOG_INTERVAL, SAVE_BEST_ONLY,
    get_device, ensure_dirs
)
from training.model import create_fer_model, count_parameters
from training.dataset import get_fer2013_dataloaders


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

        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Stats
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
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


def train(
    data_dir: str = None,
    resume_from: str = None
):
    """
    Fonction principale d'entrainement.
    """
    print("=" * 60)
    print("ENTRAINEMENT FER2013 - EfficientNet-B0")
    print("=" * 60)

    # Setup
    ensure_dirs()
    device = get_device()
    print(f"\nDevice: {device}")

    # Dataset
    if data_dir is None:
        data_dir = FER2013_DIR

    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        print(f"\nERREUR: Dataset non trouve a {data_dir}")
        print("Structure attendue:")
        print("  fer2013/train/angry/, fer2013/train/happy/, etc.")
        print("  fer2013/test/angry/, fer2013/test/happy/, etc.")
        return

    print(f"\nChargement du dataset depuis {data_dir}")
    train_loader, val_loader, test_loader = get_fer2013_dataloaders(data_dir)

    # Model
    print("\nCreation du modele...")
    model = create_fer_model(pretrained=True)
    model.to(device)

    params = count_parameters(model)
    print(f"Parametres: {params['total']:,} total, {params['trainable']:,} entrainables")

    # Resume si checkpoint fourni
    start_epoch = 0
    best_val_acc = 0.0
    if resume_from and Path(resume_from).exists():
        print(f"\nReprise depuis {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_val_acc = checkpoint.get("best_val_acc", 0.0)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=FER_LEARNING_RATE,
        weight_decay=FER_WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FER_EPOCHS - WARMUP_EPOCHS)

    # Training loop
    print(f"\nDemarrage de l'entrainement...")
    print(f"Epochs: {FER_EPOCHS}")
    print(f"Batch size: {FER_BATCH_SIZE}")
    print(f"Learning rate: {FER_LEARNING_RATE}")
    print("-" * 60)

    patience_counter = 0

    for epoch in range(start_epoch, FER_EPOCHS):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Scheduler step
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # Print metrics
        print(f"\nEpoch {epoch+1}/{FER_EPOCHS} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": FER_CLASSES
            }
            torch.save(checkpoint, FER_MODEL_PATH)
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

    # Load best model
    checkpoint = torch.load(FER_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = validate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")

    print(f"\nModele sauvegarde dans: {FER_MODEL_PATH}")
    print("Entrainement termine!")


if __name__ == "__main__":
    train()
