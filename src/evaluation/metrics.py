# src/evaluation/metrics.py

from __future__ import annotations
from typing import Dict, Any

import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
)

__all__ = ["compute_classification_metrics"]


def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        num_classes: int,
) -> Dict[str, Any]:
    assert y_true.shape[0] == y_pred.shape[0] == y_prob.shape[0]

    # Accuracy
    acc = (y_true == y_pred).mean().item() if y_true.size > 0 else 0.0

    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "roc_auc_ovr_macro": 0.0,
            "cohen_kappa_quadratic": 0.0,
            "confusion_matrix": [[0] * num_classes for _ in range(num_classes)],
        }

    # Macro metrics
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Multi-class AUC: one-vs-rest, macro average
    try:
        roc_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        roc_auc = 0.0

    # Cohen's kappa with quadratic weights
    kappa = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic",
    )

    # Confusion matrix with fixed label order [0..C-1]
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
    )

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "roc_auc_ovr_macro": float(roc_auc),
        "cohen_kappa_quadratic": float(kappa),
        "confusion_matrix": cm.tolist(),
    }
