"""Overfit test for all 12 models on the REAL data: can each one memorise one batch
of 64 real training sentences? If a model cannot, its pipeline is broken.

    python scripts/overfit_all.py --steps 200

Writes docs/images/overfit-all.png and docs/overfit.md.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from textcls.config import CELLS, CLASSES, EMBEDDINGS, IMPLS, Config, DataConfig, ModelConfig
from textcls.data import load_splits, preprocess
from textcls.plots import CELL_COLORS, INK, MUTED, _style
from textcls.train import build_model, build_tokenizer, make_loader

ROOT = Path(__file__).resolve().parents[1]


def overfit(emb: str, cell: str, impl: str, steps: int, cache: dict) -> dict:
    cfg = Config(DataConfig(batch_size=64), ModelConfig(emb, cell, impl, dropout=0.0))
    if emb not in cache:
        splits = load_splits(cfg.data.data_dir, cfg.data.cache_dir)
        texts = preprocess(splits, cfg.data.min_freq, cfg.data.simplify, cfg.data.cache_dir)
        tok, glove = build_tokenizer(cfg, texts)
        torch.manual_seed(0)
        batch = next(iter(make_loader(*texts["train"], tok, cfg, train=True)))  # same 64 sentences for all
        cache[emb] = (tok, glove, batch)
    tok, glove, (x, lengths, y) = cache[emb]
    torch.manual_seed(0)
    model = build_model(cfg, tok, glove).train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    losses, accs = [], []
    for _ in range(steps):
        logits = model(x, lengths)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append((logits.argmax(1) == y).float().mean().item())
    model.eval()
    with torch.no_grad():
        final_acc = (model(x, lengths).argmax(1) == y).float().mean().item()
    return {
        "name": f"{emb}-{cell}-{impl}",
        "emb": emb,
        "cell": cell,
        "impl": impl,
        "losses": losses,
        "accs": accs,
        "final_acc": final_acc,
        "steps_to_100": next((i + 1 for i, a in enumerate(accs) if a == 1.0), None),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    a = p.parse_args()
    cache: dict = {}
    runs = []
    for emb, cell, impl in itertools.product(EMBEDDINGS, CELLS, IMPLS):
        r = overfit(emb, cell, impl, a.steps, cache)
        runs.append(r)
        print(
            f"{r['name']:22} loss {r['losses'][0]:.3f} -> {r['losses'][-1]:.5f} | "
            f"train acc {r['final_acc']:.0%} | 100% after {r['steps_to_100']} steps",
            flush=True,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, emb in zip(axes, EMBEDDINGS):
        for r in runs:
            if r["emb"] == emb:
                ax.plot(
                    range(1, a.steps + 1),
                    r["losses"],
                    color=CELL_COLORS[r["cell"]],
                    lw=1.8,
                    ls="-" if r["impl"] == "torch" else "--",
                    label=f"{r['cell'].upper()} {'nn.*' if r['impl'] == 'torch' else 'scratch'}",
                )
        ax.axhline(math.log(len(CLASSES)), color=MUTED, ls=":", lw=1)
        ax.text(
            a.steps, math.log(len(CLASSES)), "ln 6 = random guess ", color=MUTED, ha="right", va="bottom", fontsize=8
        )
        ax.set_yscale("log")
        _style(ax, f"{'One-hot' if emb == 'onehot' else 'GloVe'} embeddings", "training step", "loss (log scale)")
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("Overfit test: every model memorises 64 real sentences", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "overfit-all.png", dpi=110)

    lines = [
        "# Overfit test (all 12 models, real data)",
        "",
        "`python scripts/overfit_all.py` trains each model on ONE batch of 64 real training",
        "sentences (same batch for all, dropout off, Adam lr 3e-3). A model that works must",
        "drive the loss from ln 6 = 1.79 to ~0 and reach 100% accuracy on that batch.",
        "",
        "| model | loss at step 1 | loss at the end | accuracy on the batch | first step at 100% |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in runs:
        lines.append(
            f"| `{r['name']}` | {r['losses'][0]:.3f} | {r['losses'][-1]:.5f} | "
            f"{r['final_acc']:.0%} | {r['steps_to_100']} |"
        )
    lines += ["", "![overfit test](images/overfit-all.png)", ""]
    (ROOT / "docs" / "overfit.md").write_text("\n".join(lines))
    print("wrote docs/overfit.md and docs/images/overfit-all.png")


if __name__ == "__main__":
    main()
