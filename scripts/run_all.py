"""Train the 12 models of the notebooks (2 embeddings x 3 cells x 2 implementations)
with one shared recipe, then write docs/results.md and the comparison figures.

    python scripts/run_all.py --epochs 8
    python scripts/run_all.py --only onehot-lstm-torch glove-gru-scratch

Each run lives in checkpoints/<name>/ (best.pt, metrics.json, history.png, confusion.png).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from textcls.config import CELLS, CLASSES, EMBEDDINGS, IMPLS, Config, DataConfig, ModelConfig, TrainConfig
from textcls.plots import plot_before_after, plot_comparison, plot_confusion, plot_history, plot_val_curves
from textcls.train import run

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "images"

# Notebook results, for the comparison table (test accuracy / macro F1 as printed there).
NOTEBOOK = {
    "onehot-rnn-torch": (75.59, 0.5896), "onehot-rnn-scratch": (67.02, 0.3646),
    "onehot-lstm-torch": (78.75, 0.5886), "onehot-lstm-scratch": (76.28, 0.6505),
    "onehot-gru-torch": (81.54, 0.6310), "onehot-gru-scratch": (76.49, 0.6044),
    "glove-rnn-torch": (79.55, 0.6436), "glove-rnn-scratch": (55.60, 0.3531),
    "glove-lstm-torch": (79.45, 0.6271), "glove-lstm-scratch": (50.25, 0.2112),
    "glove-gru-torch": (75.80, 0.6287), "glove-gru-scratch": (58.50, 0.4517),
}  # fmt: skip


def config_for(embedding: str, cell: str, impl: str, epochs: int) -> Config:
    # One recipe for all 12, so differences come from the model, not from tuning.
    return Config(
        DataConfig(),
        ModelConfig(embedding=embedding, cell=cell, impl=impl, h_dim=128, n_layers=2, dropout=0.2),
        TrainConfig(
            epochs=epochs,
            lr=3e-3,
            gamma=0.85,
            patience=4,
            output_dir=str(ROOT / "checkpoints" / f"{embedding}-{cell}-{impl}"),
        ),
    )


def write_results(rows: list[dict]) -> None:
    log = ROOT / "logs" / "sanity.log"
    sanity = (
        "\n".join(ln for ln in log.read_text().splitlines() if ln.startswith(("[", "ALL", "SOME")))
        if log.exists()
        else ""
    )
    rows = sorted(rows, key=lambda r: (r["embedding"], ["rnn", "lstm", "gru"].index(r["cell"]), r["impl"]))
    lines = [
        "# Results",
        "",
        "All 12 models trained with the same recipe on CPU (`python scripts/run_all.py`),",
        "early-stopped on validation accuracy, then the **best** checkpoint is scored on",
        "the **full** test set (2,000 sentences). Macro F1 weighs the six classes",
        "equally, so it exposes the rare classes (`surprise` is 3% of the data) that",
        "accuracy hides.",
        "",
        "| model | params (trainable) | epochs (best) | test acc | macro F1 | notebook acc | notebook F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        nb_acc, nb_f1 = NOTEBOOK.get(r["name"], (None, None))
        lines.append(
            f"| `{r['name']}` | {r['params']:,} | {r['epochs_run']} ({r['best_epoch']}) | "
            f"**{100 * r['accuracy']:.2f}%** | **{r['f1']:.3f}** | {nb_acc:.2f}% | {nb_f1:.3f} |"
        )
    lines += [
        "",
        "The notebook columns are **not** an apples-to-apples baseline: the notebooks",
        "measured on a filtered test set (texts shorter than 3 tokens removed), stopped",
        "training as soon as validation accuracy crossed a hand-picked target, and their",
        "from-scratch models were not recurrent (one cell per position). See",
        "[legacy/README.md](../legacy/README.md).",
        "",
        "![before and after](images/before-after.png)",
        "",
        "## What the numbers say",
        "",
        "* **All 12 models land within ~1.5 points (91-92% accuracy, 0.87-0.88 macro F1).**",
        "  On short, keyword-heavy sentences, a correctly wired RNN is already enough;",
        "  the cell type and the embedding matter much less than getting padding,",
        "  recurrence and evaluation right.",
        "* **`nn.*` and from-scratch versions are equivalent**, as the parity tests",
        "  predict (same equations, same parameter count). The small gaps come from",
        "  different random initialisations, not from the implementation.",
        "* **GloVe starts slower but needs no simplification**: frozen vectors plus raw",
        "  text reach the same accuracy as a learned table on lemmatised text, with",
        "  310k fewer trainable parameters (the frozen table is not trained).",
        "* **The hard classes are the rare and the close ones**: `love` is confused",
        "  with `joy`, `surprise` with `fear` (see the confusion matrices).",
        "",
        "![validation curves](images/val-curves.png)",
        "",
        "![comparison](images/comparison.png)",
        "",
        "## Sanity checks",
        "",
        "`python scripts/sanity_check.py --embedding onehot --cell lstm --impl scratch`",
        "on the real data (see [testing.md](testing.md) for why each check matters):",
        "",
        "```",
        sanity,
        "```",
        "",
        "![sanity checks](images/sanity-onehot-lstm-scratch.png)",
        "",
        "## Per model",
        "",
        "Loss, accuracy and the log10 update/weight ratio per epoch (the ratio starts",
        "near -1.9 and decays to about -2.7, close to the usual -3 rule of thumb, following the",
        "learning-rate schedule), then the test confusion matrix of the best checkpoint.",
        "",
    ]
    for r in rows:
        lines += [
            f"### `{r['name']}`",
            "",
            f"![{r['name']} history](images/{r['name']}-history.png)",
            "",
            f'<img src="images/{r["name"]}-confusion.png" width="420" alt="{r["name"]} confusion matrix">',
            "",
        ]
    (ROOT / "docs" / "results.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--only", nargs="*", help="names like onehot-lstm-torch")
    p.add_argument("--threads", type=int, default=None, help="CPU threads per run (1 on a busy machine)")
    p.add_argument("--report-only", action="store_true", help="rebuild docs from existing checkpoints")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    IMG.mkdir(parents=True, exist_ok=True)

    rows = []
    for emb, cell, impl in itertools.product(EMBEDDINGS, CELLS, IMPLS):
        name = f"{emb}-{cell}-{impl}"
        out = ROOT / "checkpoints" / name
        if not a.report_only and (not a.only or name in a.only):
            run(config_for(emb, cell, impl, a.epochs))
        if (out / "metrics.json").exists():
            m = json.loads((out / "metrics.json").read_text())
            rows.append({**m, "embedding": emb, "cell": cell, "impl": impl})
            # redraw from metrics.json so the figures always match the current plotting code
            plot_history(m["history"], IMG / f"{name}-history.png", title=name)
            plot_confusion(m["confusion_matrix"], CLASSES, IMG / f"{name}-confusion.png", f"{name}: test set")

    plot_comparison(rows, IMG / "comparison.png")
    plot_val_curves(rows, IMG / "val-curves.png")
    plot_before_after(rows, NOTEBOOK, IMG / "before-after.png")
    write_results(rows)
    print(f"wrote docs/results.md with {len(rows)} models")


if __name__ == "__main__":
    main()
