"""Fixed evaluation infrastructure. The agent CANNOT modify this file."""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ACTIVITY_LABELS = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking",
    5: "running", 6: "cycling", 7: "nordic_walking",
    12: "ascending_stairs", 13: "descending_stairs",
    16: "vacuum_cleaning", 17: "ironing", 24: "rope_jumping",
}


def evaluate(y_true, y_pred):
    """Compute metrics. Returns dict with f1_macro, accuracy, per_class, confusion_matrix."""
    present_labels = sorted(set(y_true) | set(y_pred))

    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    f1_per = f1_score(y_true, y_pred, average=None, labels=present_labels, zero_division=0)
    per_class = {}
    for i, label in enumerate(present_labels):
        name = ACTIVITY_LABELS.get(label, f"class_{label}")
        per_class[name] = float(f1_per[i])

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    return {
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
