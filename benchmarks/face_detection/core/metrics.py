# -*- coding: utf-8 -*-
"""
Calculs de métriques pour le benchmark de détection de visage.
IoU, Matching, AP, etc.
"""

from typing import List, Tuple
from .structures import BBox


def compute_iou(box1: BBox, box2: BBox) -> float:
    """Calcule l'IoU (Intersection over Union) entre deux boxes."""
    # Coordonnées de l'intersection
    x1 = max(box1.x, box2.x)
    y1 = max(box1.y, box2.y)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    # Aire de l'intersection
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    # Aire de l'union
    union_area = box1.area + box2.area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_detections(
    predictions: List[BBox],
    ground_truth: List[BBox],
    iou_threshold: float = 0.5
) -> Tuple[int, int, int, List[Tuple[float, bool]]]:
    """
    Matche les détections avec le ground truth.

    Returns:
        tp: True Positives
        fp: False Positives
        fn: False Negatives
        scores: Liste de (confidence, is_tp) pour courbe PR
    """
    if not ground_truth:
        # Toutes les détections sont des FP
        return 0, len(predictions), 0, [(p.confidence, False) for p in predictions]

    if not predictions:
        # Tous les GT sont des FN
        return 0, 0, len(ground_truth), []

    # Trier les prédictions par confiance décroissante
    sorted_preds = sorted(predictions, key=lambda x: x.confidence, reverse=True)

    # Marquer les GT comme matchés ou non
    gt_matched = [False] * len(ground_truth)

    tp = 0
    fp = 0
    scores = []

    for pred in sorted_preds:
        best_iou = 0.0
        best_gt_idx = -1

        # Trouver le meilleur match
        for gt_idx, gt in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            if gt.invalid:  # Ignorer les visages invalides
                continue

            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Vérifier si le match est suffisant
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
            scores.append((pred.confidence, True))
        else:
            fp += 1
            scores.append((pred.confidence, False))

    # Compter les FN (GT non matchés)
    fn = sum(1 for i, matched in enumerate(gt_matched)
             if not matched and not ground_truth[i].invalid)

    return tp, fp, fn, scores


def get_difficulty_level(gt_boxes: List[BBox]) -> str:
    """
    Détermine le niveau de difficulté d'une image selon WIDER FACE.

    Easy: visages clairs, sans occlusion, pose typique
    Medium: flou léger ou occlusion partielle
    Hard: flou fort, occlusion forte, ou pose atypique
    """
    if not gt_boxes:
        return "easy"

    hard_count = 0
    medium_count = 0

    for box in gt_boxes:
        if box.invalid:
            continue

        # Hard: heavy blur, heavy occlusion, ou atypical pose
        if box.blur == 2 or box.occlusion == 2 or box.pose == 1:
            hard_count += 1
        # Medium: normal blur ou partial occlusion
        elif box.blur == 1 or box.occlusion == 1:
            medium_count += 1

    total = len([b for b in gt_boxes if not b.invalid])
    if total == 0:
        return "easy"

    # Majoritairement hard -> hard
    if hard_count / total > 0.3:
        return "hard"
    elif medium_count / total > 0.3:
        return "medium"
    else:
        return "easy"


def compute_ap(scores: List[Tuple[float, bool]]) -> float:
    """
    Calcule l'Average Precision (AP) à partir des scores.

    Args:
        scores: Liste de (confidence, is_true_positive)

    Returns:
        AP (0-1)
    """
    if not scores:
        return 0.0

    # Trier par confiance décroissante
    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []

    total_positives = sum(1 for _, is_tp in sorted_scores if is_tp)
    if total_positives == 0:
        return 0.0

    for _conf, is_tp in sorted_scores:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1

        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / total_positives

        precisions.append(precision)
        recalls.append(recall)

    # Interpolation (PASCAL VOC style)
    ap = 0.0
    for i in range(len(precisions) - 1, -1, -1):
        if i > 0:
            precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Calcul de l'aire sous la courbe
    for i in range(len(recalls) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return ap
