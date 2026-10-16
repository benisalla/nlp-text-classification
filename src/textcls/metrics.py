"""Accuracy, macro precision / recall / F1 and the confusion matrix, from scratch
(the notebooks used torchmetrics; these are a few lines and easy to test)."""

from __future__ import annotations

import torch


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, n_class: int) -> torch.Tensor:
    """cm[i, j] = number of samples whose TRUE class is i and PREDICTED class is j."""
    idx = labels.long() * n_class + preds.long()
    return torch.bincount(idx, minlength=n_class * n_class).reshape(n_class, n_class)


def classification_report(preds, labels, n_class: int) -> dict:
    cm = confusion_matrix(preds, labels, n_class).double()
    tp = cm.diag()
    predicted = cm.sum(0)  # column sums: how often each class was predicted
    actual = cm.sum(1)  # row sums: how often each class really occurs
    precision = torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(actual > 0, tp / actual.clamp(min=1), torch.zeros_like(tp))
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom.clamp(min=1e-12), torch.zeros_like(tp))
    return {
        "accuracy": (tp.sum() / cm.sum()).item(),
        "precision": precision.mean().item(),  # macro: every class counts the same
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.long().tolist(),
    }


def format_report(rep: dict, classes: list[str], loss: float | None = None) -> str:
    """The notebook's 'Test Results Summary' table, with class names on the matrix."""
    lines = ["=" * 40, f"{'Test Results Summary':^40}", "=" * 40, f"{'Metric':<20} | {'Value':>15}", "-" * 40]
    if loss is not None:
        lines.append(f"{'Average Loss':<20} | {loss:>15.4f}")
    lines += [
        f"{'Accuracy':<20} | {rep['accuracy'] * 100:>14.2f}%",
        f"{'Precision (macro)':<20} | {rep['precision']:>15.4f}",
        f"{'Recall (macro)':<20} | {rep['recall']:>15.4f}",
        f"{'F1 (macro)':<20} | {rep['f1']:>15.4f}",
        "=" * 40,
        "",
        "Confusion matrix (rows = true, columns = predicted):",
        " " * 10 + "".join(f"{c[:8]:>9}" for c in classes),
    ]
    for c, row in zip(classes, rep["confusion_matrix"]):
        lines.append(f"{c[:9]:<10}" + "".join(f"{v:>9}" for v in row))
    return "\n".join(lines)
