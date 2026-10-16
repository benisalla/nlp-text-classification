"""Figures saved as PNG (the notebook's live_plot_triple, written to disk instead of shown)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

TRAIN, VAL = "#2a78d6", "#eb6834"  # blue = train, orange = validation
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#2a78d6", "#0d3a73"])


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_history(history: dict, path: str | Path, title: str = "") -> None:
    """Loss, accuracy and the log10 update/weight ratio per epoch."""
    ep = list(range(1, len(history["train_loss"]) + 1))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.2))
    a1.plot(ep, history["train_loss"], color=TRAIN, lw=2, marker="o", ms=4, label="train")
    a1.plot(ep, history["val_loss"], color=VAL, lw=2, marker="o", ms=4, label="validation")
    _style(a1, "Loss", "epoch", "cross-entropy")
    a1.legend(frameon=False)

    a2.plot(ep, [100 * a for a in history["train_acc"]], color=TRAIN, lw=2, marker="o", ms=4, label="train")
    a2.plot(ep, [100 * a for a in history["val_acc"]], color=VAL, lw=2, marker="o", ms=4, label="validation")
    _style(a2, "Accuracy", "epoch", "accuracy (%)")
    a2.legend(frameon=False)

    a3.plot(ep, history["update_ratio"], color=TRAIN, lw=2, marker="o", ms=4)
    a3.axhline(-3, color=MUTED, ls="--", lw=1)
    a3.text(ep[-1], -3, " 1e-3 rule of thumb", color=MUTED, va="bottom", ha="right", fontsize=9)
    _style(a3, "Update / weight ratio", "epoch", "log10(|Δw| / |w|)")

    if title:
        fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], classes: list[str], path: str | Path, title: str = "") -> None:
    """Row-normalised confusion matrix (each row = 100% of that true class), counts in the cells."""
    import numpy as np

    m = np.array(cm, dtype=float)
    norm = m / m.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                f"{int(m[i, j])}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if norm[i, j] > 0.55 else INK,
            )
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right", color=MUTED)
    ax.set_yticks(range(len(classes)), classes, color=MUTED)
    ax.set_xlabel("predicted", color=MUTED)
    ax.set_ylabel("true", color=MUTED)
    ax.set_title(title or "Confusion matrix (test)", color=INK, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_comparison(rows: list[dict], path: str | Path, metric: str = "f1") -> None:
    """Horizontal bars: one per model, sorted, test accuracy and macro F1 side by side."""
    rows = sorted(rows, key=lambda r: r[metric])
    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    y = range(len(rows))
    h = 0.38
    ax.barh([i + h / 2 for i in y], [100 * r["accuracy"] for r in rows], height=h, color=TRAIN, label="accuracy")
    ax.barh([i - h / 2 for i in y], [100 * r["f1"] for r in rows], height=h, color=VAL, label="macro F1")
    for i, r in enumerate(rows):
        ax.text(
            100 * r["accuracy"] + 0.5, i + h / 2, f"{100 * r['accuracy']:.1f}", va="center", fontsize=8, color=MUTED
        )
        ax.text(100 * r["f1"] + 0.5, i - h / 2, f"{100 * r['f1']:.1f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(list(y), names)
    ax.set_xlim(0, 105)
    _style(ax, "Test set: accuracy and macro F1 (%)", "", "")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


CELL_COLORS = {"rnn": "#2a78d6", "lstm": "#eb6834", "gru": "#1baf7a"}


def plot_val_curves(rows: list[dict], path: str | Path) -> None:
    """Validation accuracy per epoch for every model: one panel per embedding,
    one colour per cell, solid = nn.* and dashed = from scratch."""
    embs = ["onehot", "glove"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, emb in zip(axes, embs):
        for r in rows:
            if r["embedding"] != emb:
                continue
            acc = [100 * a for a in r["history"]["val_acc"]]
            ax.plot(
                range(1, len(acc) + 1),
                acc,
                color=CELL_COLORS[r["cell"]],
                lw=2,
                ls="-" if r["impl"] == "torch" else "--",
                marker="o",
                ms=3,
                label=f"{r['cell'].upper()} {'nn.*' if r['impl'] == 'torch' else 'scratch'}",
            )
        _style(ax, f"{'One-hot' if emb == 'onehot' else 'GloVe'} embeddings", "epoch", "validation accuracy (%)")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.suptitle("Validation accuracy during training (all 12 models)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_before_after(rows: list[dict], notebook: dict, path: str | Path) -> None:
    """Test accuracy per model: the original notebooks vs this package."""
    rows = sorted(rows, key=lambda r: (r["embedding"], ["rnn", "lstm", "gru"].index(r["cell"]), r["impl"]))
    names = [r["name"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 1.3))
    h = 0.38
    for i, r in enumerate(rows):
        old = notebook[r["name"]][0]
        new = 100 * r["accuracy"]
        ax.barh(i + h / 2, old, height=h, color="#c3c2b7", label="notebook" if i == 0 else None)
        ax.barh(i - h / 2, new, height=h, color=TRAIN, label="this repo" if i == 0 else None)
        ax.text(old + 0.5, i + h / 2, f"{old:.1f}", va="center", fontsize=8, color=MUTED)
        ax.text(new + 0.5, i - h / 2, f"{new:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    _style(ax, "Test accuracy (%): original notebooks vs fixed package", "", "")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
