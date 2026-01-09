# -*- coding: utf-8 -*-
"""
Modeles pour la detection d'emotions.
Supporte EfficientNet-B0 et MobileNetV3 (Small/Large).
Fine-tuning depuis les poids ImageNet.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)
from typing import Optional, Literal

from training.config import FER_NUM_CLASSES, L3_NUM_CLASSES, BACKBONE, get_device

# Types de backbones supportes
BackboneType = Literal["efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large"]

# Informations sur les backbones
BACKBONE_INFO = {
    "efficientnet_b0": {
        "model_fn": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.IMAGENET1K_V1,
        "features_attr": "features",
        "classifier_attr": "classifier",
        "in_features": 1280,
        "classifier_idx": 1,  # Index de la couche Linear dans le classifier
    },
    "mobilenet_v3_small": {
        "model_fn": mobilenet_v3_small,
        "weights": MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "features_attr": "features",
        "classifier_attr": "classifier",
        "in_features": 576,
        "classifier_idx": 3,  # MobileNetV3 a un classifier plus complexe
    },
    "mobilenet_v3_large": {
        "model_fn": mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "features_attr": "features",
        "classifier_attr": "classifier",
        "in_features": 960,
        "classifier_idx": 3,
    },
}


class EmotionClassifier(nn.Module):
    """
    Classificateur d'emotions multi-backbone.

    Supporte:
        - EfficientNet-B0 (5.3M params, 1280 features)
        - MobileNetV3-Small (2.5M params, 576 features) - Plus rapide
        - MobileNetV3-Large (5.4M params, 960 features) - Meilleur compromis

    Architecture:
        Backbone (ImageNet) -> Dropout -> FC (num_classes)
    """

    def __init__(
        self,
        num_classes: int = FER_NUM_CLASSES,
        backbone: BackboneType = BACKBONE,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Nombre de classes (7 pour FER, 2 pour L3)
            backbone: Type de backbone ("efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large")
            pretrained: Utiliser les poids ImageNet
            dropout: Taux de dropout avant la couche finale
        """
        super().__init__()

        if backbone not in BACKBONE_INFO:
            raise ValueError(f"Backbone '{backbone}' non supporte. Options: {list(BACKBONE_INFO.keys())}")

        self.backbone_type = backbone
        self.num_classes = num_classes
        info = BACKBONE_INFO[backbone]

        # Charger le backbone
        if pretrained:
            self.backbone = info["model_fn"](weights=info["weights"])
        else:
            self.backbone = info["model_fn"](weights=None)

        # Obtenir le nombre de features en sortie
        in_features = info["in_features"]

        # Remplacer le classifier selon le type de backbone
        if backbone == "efficientnet_b0":
            # EfficientNet: classifier = [Dropout, Linear]
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        else:
            # MobileNetV3: classifier = [Linear, Hardswish, Dropout, Linear]
            # On garde la structure mais on change la derniere couche
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1024, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch_size, 3, 224, 224)

        Returns:
            Logits de shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """Gele le backbone (pour fine-tuning de la tete seulement)."""
        features = getattr(self.backbone, "features")
        for param in features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Degele le backbone (pour fine-tuning complet)."""
        features = getattr(self.backbone, "features")
        for param in features.parameters():
            param.requires_grad = True

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les features avant le classifier.

        Args:
            x: Tensor de shape (batch_size, 3, 224, 224)

        Returns:
            Features de shape (batch_size, in_features)
        """
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def create_fer_model(
    pretrained: bool = True,
    backbone: BackboneType = BACKBONE
) -> EmotionClassifier:
    """
    Cree un modele pour FER2013 (7 classes).

    Args:
        pretrained: Charger les poids ImageNet
        backbone: Type de backbone a utiliser
    """
    return EmotionClassifier(
        num_classes=FER_NUM_CLASSES,
        backbone=backbone,
        pretrained=pretrained
    )


def create_l3_model(
    fer_checkpoint: Optional[str] = None,
    backbone: BackboneType = BACKBONE
) -> EmotionClassifier:
    """
    Cree un modele pour L3 (2 classes).

    Args:
        fer_checkpoint: Chemin vers les poids du modele FER2013.
                       Si fourni, charge les poids du backbone.
        backbone: Type de backbone a utiliser
    """
    model = EmotionClassifier(
        num_classes=L3_NUM_CLASSES,
        backbone=backbone,
        pretrained=(fer_checkpoint is None)
    )

    if fer_checkpoint:
        # Charger les poids FER2013
        checkpoint = torch.load(fer_checkpoint, map_location="cpu")

        # Gerer les deux formats: dict avec "model_state_dict" ou state_dict direct
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            fer_state = checkpoint["model_state_dict"]
        else:
            fer_state = checkpoint

        # Creer un modele FER temporaire pour extraire les features
        fer_model = EmotionClassifier(
            num_classes=FER_NUM_CLASSES,
            backbone=backbone,
            pretrained=False
        )
        fer_model.load_state_dict(fer_state)

        # Copier les poids du backbone (pas du classifier)
        model.backbone.features.load_state_dict(fer_model.backbone.features.state_dict())
        print(f"Loaded backbone weights from {fer_checkpoint}")

    return model


def load_model(
    checkpoint_path: str,
    num_classes: int,
    backbone: BackboneType = BACKBONE
) -> EmotionClassifier:
    """
    Charge un modele depuis un checkpoint.

    Args:
        checkpoint_path: Chemin vers le checkpoint
        num_classes: Nombre de classes du modele
        backbone: Type de backbone utilise
    """
    model = EmotionClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=False
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Gerer les deux formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Compte les parametres du modele.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable
    }


def get_backbone_info(backbone: BackboneType = BACKBONE) -> dict:
    """
    Retourne les informations sur un backbone.
    """
    info = BACKBONE_INFO[backbone].copy()
    info["name"] = backbone
    return info


if __name__ == "__main__":
    # Test des modeles
    device = get_device()
    print(f"Using device: {device}")
    print("=" * 60)

    # Tester chaque backbone
    for backbone_name in BACKBONE_INFO.keys():
        print(f"\n--- {backbone_name.upper()} ---")

        # Creer modele FER
        fer_model = create_fer_model(backbone=backbone_name)
        fer_model.to(device)

        params = count_parameters(fer_model)
        print(f"FER Model ({FER_NUM_CLASSES} classes):")
        print(f"  Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        output = fer_model(dummy_input)
        print(f"  Input: {dummy_input.shape} -> Output: {output.shape}")

        # Creer modele L3
        l3_model = create_l3_model(backbone=backbone_name)
        l3_model.to(device)
        params = count_parameters(l3_model)
        print(f"L3 Model ({L3_NUM_CLASSES} classes):")
        print(f"  Parameters: {params['total']:,} total")

    print("\n" + "=" * 60)
    print("Tous les backbones fonctionnent!")
