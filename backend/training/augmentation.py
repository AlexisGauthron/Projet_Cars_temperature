# -*- coding: utf-8 -*-
"""
Data Augmentation pour l'entrainement.
Transformations pour ameliorer la generalisation du modele.
"""

import torchvision.transforms as T
from training.config import IMAGE_SIZE, AUGMENTATION


def get_train_transforms():
    """
    Transformations pour l'entrainement avec data augmentation.
    """
    transforms_list = [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    # Random Horizontal Flip
    if AUGMENTATION.get("horizontal_flip", True):
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    # Random Rotation
    rotation = AUGMENTATION.get("rotation", 0)
    if rotation > 0:
        transforms_list.append(T.RandomRotation(rotation))

    # Color Jitter (brightness, contrast, saturation)
    brightness = AUGMENTATION.get("brightness", 0)
    contrast = AUGMENTATION.get("contrast", 0)
    saturation = AUGMENTATION.get("saturation", 0)
    if brightness > 0 or contrast > 0 or saturation > 0:
        transforms_list.append(
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation
            )
        )

    # Random Resized Crop
    if AUGMENTATION.get("random_crop", False):
        transforms_list.append(
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1))
        )

    # Convert to Tensor
    transforms_list.append(T.ToTensor())

    # Normalize (ImageNet stats car EfficientNet est pre-entraine dessus)
    transforms_list.append(
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    # Random Erasing (apres ToTensor)
    erasing_prob = AUGMENTATION.get("random_erasing", 0)
    if erasing_prob > 0:
        transforms_list.append(
            T.RandomErasing(p=erasing_prob, scale=(0.02, 0.2))
        )

    return T.Compose(transforms_list)


def get_val_transforms():
    """
    Transformations pour validation/test (pas d'augmentation).
    """
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms():
    """
    Transformations pour l'inference en production.
    """
    return get_val_transforms()


# Exemple d'utilisation
if __name__ == "__main__":
    print("Train transforms:")
    print(get_train_transforms())
    print("\nVal transforms:")
    print(get_val_transforms())
