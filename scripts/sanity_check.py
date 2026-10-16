"""Neural-network sanity checks on the REAL data (Karpathy, "A Recipe for Training
Neural Networks"). Run before any long training:

    python scripts/sanity_check.py --embedding onehot --cell lstm --impl scratch

1. loss at init       ~ ln(6) = 1.792: the model starts out unsure, not confidently wrong
2. overfit one batch  64 real sentences, loss -> ~0, accuracy -> 100%
3. blind baseline     same model trained on inputs replaced by <oov>: it can only learn
                      the class frequencies, so the real model must do clearly better
4. gradient check     every trainable tensor receives a non-zero gradient

Writes docs/images/sanity-<name>.png (overfit curve + real vs blind).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from textcls.config import CELLS, CLASSES, EMBEDDINGS, IMPLS, Config, DataConfig, ModelConfig
from textcls.data import load_splits, preprocess
from textcls.plots import MUTED, TRAIN, VAL, _style
from textcls.train import build_model, build_tokenizer, make_loader

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", choices=EMBEDDINGS, default="onehot")
    p.add_argument("--cell", choices=CELLS, default="lstm")
    p.add_argument("--impl", choices=IMPLS, default="torch")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--blind-epochs", type=int, default=2)
    a = p.parse_args()
    torch.manual_seed(0)
    name = f"{a.embedding}-{a.cell}-{a.impl}"
    cfg = Config(DataConfig(batch_size=64), ModelConfig(a.embedding, a.cell, a.impl, dropout=0.0))
    splits = load_splits(cfg.data.data_dir, cfg.data.cache_dir)
    texts = preprocess(splits, cfg.data.min_freq, cfg.data.simplify, cfg.data.cache_dir)
    tok, glove = build_tokenizer(cfg, texts)
    train = make_loader(*texts["train"], tok, cfg, train=True)
    val = make_loader(*texts["val"], tok, cfg, train=False)
    x, lengths, y = next(iter(train))
    ok = True

    # 1. initial loss
    model = build_model(cfg, tok, glove).eval()
    with torch.no_grad():
        init = F.cross_entropy(model(x, lengths), y).item()
    good = abs(init - math.log(len(CLASSES))) < 0.2
    ok &= good
    print(f"[{'ok' if good else 'FAIL'}] 1. loss at init {init:.3f}  (expected ~{math.log(6):.3f})")

    # 4. gradients (done before overfitting, on the fresh model)
    model.train()
    F.cross_entropy(model(x, lengths), y).backward()
    dead = [n for n, q in model.named_parameters() if q.requires_grad and (q.grad is None or q.grad.abs().sum() == 0)]
    ok &= not dead
    print(f"[{'ok' if not dead else 'FAIL'}] 4. gradients reach every trainable tensor {dead or ''}")
    model.zero_grad()

    # 2. overfit one batch
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=3e-3)
    curve = []
    for _ in range(a.steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x, lengths), y)
        loss.backward()
        opt.step()
        curve.append(loss.item())
    acc = (model(x, lengths).argmax(1) == y).float().mean().item()
    good = curve[-1] < 0.05 and acc == 1.0
    ok &= good
    print(
        f"[{'ok' if good else 'FAIL'}] 2. overfit 64 sentences: loss {curve[0]:.3f} -> {curve[-1]:.4f}, acc {acc:.0%}"
    )

    # 3. real vs blind
    def short_run(blind: bool) -> list[float]:
        torch.manual_seed(0)
        m = build_model(cfg, tok, glove)
        o = torch.optim.Adam([q for q in m.parameters() if q.requires_grad], lr=3e-3)
        accs = []
        for _ in range(a.blind_epochs):
            m.train()
            for xb, lb, yb in train:
                if blind:
                    xb, lb = torch.full((xb.size(0), 1), tok.oov_id), torch.ones_like(lb)
                o.zero_grad()
                F.cross_entropy(m(xb, lb), yb).backward()
                o.step()
            m.eval()
            right = n = 0
            with torch.no_grad():
                for xb, lb, yb in val:
                    if blind:
                        xb, lb = torch.full((xb.size(0), 1), tok.oov_id), torch.ones_like(lb)
                    right += (m(xb, lb).argmax(1) == yb).sum().item()
                    n += yb.numel()
            accs.append(right / n)
        return accs

    real, blind = short_run(False), short_run(True)
    good = real[-1] > blind[-1] + 0.2
    ok &= good
    print(
        f"[{'ok' if good else 'FAIL'}] 3. val acc after {a.blind_epochs} epochs: real {real[-1]:.3f} vs blind {blind[-1]:.3f}"
    )

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(curve, color=TRAIN, lw=2)
    a1.axhline(math.log(6), color=MUTED, ls="--", lw=1)
    a1.text(len(curve) - 1, math.log(6), "ln 6 ", color=MUTED, ha="right", va="bottom", fontsize=9)
    a1.set_yscale("log")
    _style(a1, "Overfit one batch (64 real sentences)", "step", "loss (log scale)")
    ep = range(1, len(real) + 1)
    a2.plot(ep, [100 * v for v in real], color=TRAIN, lw=2, marker="o", label="real text")
    a2.plot(ep, [100 * v for v in blind], color=VAL, lw=2, marker="o", label="blind (input erased)")
    a2.set_ylim(0, 100)
    a2.set_xticks(list(ep))
    _style(a2, "Validation accuracy: real vs blind", "epoch", "accuracy (%)")
    a2.legend(frameon=False)
    fig.suptitle(f"Sanity checks: {name}", x=0.01, ha="left")
    fig.tight_layout()
    out = ROOT / "docs" / "images" / f"sanity-{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} -> {out.relative_to(ROOT)}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
