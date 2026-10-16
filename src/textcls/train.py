"""Training loop and the `textcls-train` command.

    textcls-train --embedding onehot --cell lstm --impl torch --epochs 10

Writes to --output-dir: best.pt (best validation accuracy), metrics.json,
history.png (loss / accuracy / update ratio) and confusion.png (test set).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import CELLS, CLASSES, EMBEDDINGS, IMPLS, Config, DataConfig, ModelConfig, TrainConfig
from .data import TextDataset, load_splits, pad_collate_fn, preprocess
from .metrics import classification_report, format_report
from .model import TextClassifier, count_params
from .tokenizer import WordTokenizer, build_glove, tokenizer_from_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def get_lr(epoch: int, lr: float, gamma: float, min_lr: float) -> float:
    """Exponential decay with a floor (the notebook's get_lr)."""
    return max(lr * gamma**epoch, min_lr)


# ----------------------------------------------------------------------------- building blocks


def build_tokenizer(cfg: Config, texts: dict):
    """Returns (tokenizer, glove_weight or None)."""
    if cfg.model.embedding == "onehot":
        return WordTokenizer(cfg.data.vocab_size).fit(texts["train"][0]), None
    all_text = [t for split in texts.values() for t in split[0]]
    return build_glove(all_text, cfg.model.n_emb, cfg.data.cache_dir)


def make_loader(texts, labels, tok, cfg: Config, train: bool) -> DataLoader:
    ds = TextDataset(texts, labels, tok, cfg.data.max_seq_len, cfg.data.min_seq_len if train else 0)
    return DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=train,
        collate_fn=partial(pad_collate_fn, pad_id=tok.pad_id),
    )


def build_model(cfg: Config, tok, glove_weight=None) -> TextClassifier:
    return TextClassifier(cfg.model, tok.vocab_size, len(CLASSES), glove_weight)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]  # frozen GloVe excluded
    return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)


# ----------------------------------------------------------------------------- one epoch


def train_one_epoch(model, loader, criterion, optimizer, device, max_norm=None, max_steps=None) -> dict:
    """The notebook's trainer(). One fix: the update ratio is measured on the real
    AdamW step (weights after - weights before), not on lr * grad, which is not
    what Adam applies."""
    model.train()
    tot_loss = correct = seen = 0
    ratios = []
    params = [p for p in model.parameters() if p.requires_grad]
    # Ratio on weight matrices only: biases start at zero, so |dw| / |w| is meaningless for them.
    weights = [p for p in params if p.dim() >= 2]
    for step, (x, lengths, y) in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm:
            torch.nn.utils.clip_grad_norm_(params, max_norm)
        before = [p.detach().clone() for p in weights]
        optimizer.step()
        with torch.no_grad():
            r = [((p - b).norm() / b.norm()).item() for p, b in zip(weights, before)]
        ratios.append(sum(r) / len(r))

        tot_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    mean_ratio = sum(ratios) / max(len(ratios), 1)
    return {
        "loss": tot_loss / seen,
        "acc": correct / seen,
        "update_ratio": math.log10(mean_ratio) if mean_ratio > 0 else float("-inf"),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """The notebook's validator() + tester(): always the WHOLE split (the notebook
    validated on 10-20 random batches only)."""
    model.eval()
    tot_loss = seen = 0
    preds, labels = [], []
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        tot_loss += criterion(logits, y).item() * y.size(0)
        seen += y.size(0)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds), torch.cat(labels)
    rep = classification_report(preds, labels, len(CLASSES))
    return {"loss": tot_loss / seen, "acc": rep["accuracy"], "report": rep}


# ----------------------------------------------------------------------------- checkpoints


def save_checkpoint(path, model, cfg: Config, tok, extra: dict | None = None) -> None:
    torch.save(
        {"config": cfg.to_dict(), "tokenizer": tok.to_dict(), "model": model.state_dict(), **(extra or {})},
        path,
    )


def load_checkpoint(path, device="cpu"):
    """-> (model, tokenizer, config, raw checkpoint dict)"""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ck["config"])
    tok = tokenizer_from_dict(ck["tokenizer"])
    model = build_model(cfg, tok)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), tok, cfg, ck


# ----------------------------------------------------------------------------- full run


def run(cfg: Config, log=print) -> dict:
    """Load data, train with early stopping, test the best checkpoint, save everything."""
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    splits = load_splits(cfg.data.data_dir, cfg.data.cache_dir)
    texts = preprocess(splits, cfg.data.min_freq, cfg.data.simplify, cfg.data.cache_dir)
    if cfg.train.overfit:
        n = cfg.train.overfit
        texts = {"train": (texts["train"][0][:n], texts["train"][1][:n])}
        texts["val"] = texts["test"] = texts["train"]
    tok, glove_weight = build_tokenizer(cfg, texts)

    train_loader = make_loader(*texts["train"], tok, cfg, train=True)
    val_loader = make_loader(*texts["val"], tok, cfg, train=False)
    test_loader = make_loader(*texts["test"], tok, cfg, train=False)

    model = build_model(cfg, tok, glove_weight).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
    optimizer = make_optimizer(model, cfg.train)
    name = f"{cfg.model.embedding}-{cfg.model.cell}-{cfg.model.impl}"
    log(
        f"{name}: {count_params(model):,} trainable params | vocab {tok.vocab_size} | "
        f"train {len(train_loader.dataset)} val {len(val_loader.dataset)} test {len(test_loader.dataset)} | {device}"
    )

    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "update_ratio", "lr")}
    best_acc, best_epoch, bad = -1.0, 0, 0
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        lr = get_lr(epoch, cfg.train.lr, cfg.train.gamma, cfg.train.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.train.max_norm, cfg.train.max_steps_per_epoch
        )
        vl = evaluate(model, val_loader, criterion, device)
        for k, v in [
            ("train_loss", tr["loss"]),
            ("val_loss", vl["loss"]),
            ("train_acc", tr["acc"]),
            ("val_acc", vl["acc"]),
            ("update_ratio", tr["update_ratio"]),
            ("lr", lr),
        ]:
            hist[k].append(v)
        log(
            f"epoch {epoch + 1:3d} | lr {lr:.2e} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {vl['loss']:.4f} acc {vl['acc']:.4f} | log10 ratio {tr['update_ratio']:.2f} | "
            f"{time.time() - t0:.0f}s"
        )

        # Early stopping on validation accuracy. (The notebook also stopped as soon as
        # val acc crossed a hand-picked target such as 0.78; that is removed.)
        if vl["acc"] - best_acc > cfg.train.min_delta:
            best_acc, best_epoch, bad = vl["acc"], epoch + 1, 0
            save_checkpoint(out / "best.pt", model, cfg, tok)
        else:
            bad += 1
            if bad >= cfg.train.patience:
                log(f"early stop: no val improvement for {bad} epochs")
                break

    # Test the BEST checkpoint (the notebook tested whatever weights it ended with).
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    te = evaluate(model, test_loader, criterion, device)
    log(format_report(te["report"], CLASSES, te["loss"]))

    metrics = {
        "name": name,
        "params": count_params(model),
        "best_epoch": best_epoch,
        "epochs_run": len(hist["train_loss"]),
        "train_seconds": round(time.time() - t0, 1),
        "val_acc": best_acc,
        "test_loss": te["loss"],
        **{k: te["report"][k] for k in ("accuracy", "precision", "recall", "f1", "per_class_f1", "confusion_matrix")},
        "history": hist,
        "config": cfg.to_dict(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    from .plots import plot_confusion, plot_history

    plot_history(hist, out / "history.png", title=name)
    plot_confusion(te["report"]["confusion_matrix"], CLASSES, out / "confusion.png", f"{name}: test confusion matrix")
    return metrics


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    d, m, t = DataConfig(), ModelConfig(), TrainConfig()
    p = argparse.ArgumentParser(description="Train an emotion classifier (6 classes).")
    g = p.add_argument_group("model")
    g.add_argument("--embedding", choices=EMBEDDINGS, default=m.embedding)
    g.add_argument("--cell", choices=CELLS, default=m.cell)
    g.add_argument("--impl", choices=IMPLS, default=m.impl)
    g.add_argument("--n-emb", type=int, default=m.n_emb)
    g.add_argument("--h-dim", type=int, default=m.h_dim)
    g.add_argument("--n-layers", type=int, default=m.n_layers)
    g.add_argument("--unidirectional", action="store_true", help="default is bidirectional")
    g.add_argument("--dropout", type=float, default=m.dropout)
    g.add_argument("--act", choices=("relu", "tanh"), default=m.act)
    g.add_argument("--residual", action="store_true", help="scratch cells: add the 0.5 * x skip")
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default=d.data_dir, help="folder with train/val/test.txt (default: download)")
    g.add_argument("--cache-dir", default=d.cache_dir)
    g.add_argument("--min-freq", type=int, default=d.min_freq)
    g.add_argument("--simplify", dest="simplify", action="store_true", default=None)
    g.add_argument("--no-simplify", dest="simplify", action="store_false")
    g.add_argument("--max-seq-len", type=int, default=d.max_seq_len)
    g.add_argument("--min-seq-len", type=int, default=d.min_seq_len)
    g.add_argument("--vocab-size", type=int, default=d.vocab_size)
    g.add_argument("--batch-size", type=int, default=d.batch_size)
    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, default=t.epochs)
    g.add_argument("--lr", type=float, default=t.lr)
    g.add_argument("--min-lr", type=float, default=t.min_lr)
    g.add_argument("--gamma", type=float, default=t.gamma)
    g.add_argument("--weight-decay", type=float, default=t.weight_decay)
    g.add_argument("--label-smoothing", type=float, default=t.label_smoothing)
    g.add_argument("--max-norm", type=float, default=t.max_norm)
    g.add_argument("--patience", type=int, default=t.patience)
    g.add_argument("--seed", type=int, default=t.seed)
    g.add_argument("--device", default=t.device)
    g.add_argument("--threads", type=int, default=None, help="CPU threads (use 1 on a busy machine)")
    g.add_argument("--output-dir", default=None, help="default: checkpoints/<embedding>-<cell>-<impl>")
    g.add_argument("--overfit", type=int, default=0, help="sanity check: memorise N training samples")
    g.add_argument("--max-steps-per-epoch", type=int, default=None)
    return p


def config_from_args(a: argparse.Namespace) -> Config:
    model = ModelConfig(
        a.embedding, a.cell, a.impl, a.n_emb, a.h_dim, a.n_layers, not a.unidirectional, a.dropout, a.act, a.residual
    )
    data = DataConfig(
        a.data_dir, a.cache_dir, a.min_freq, a.simplify, a.max_seq_len, a.min_seq_len, a.vocab_size, a.batch_size
    )
    out = a.output_dir or f"checkpoints/{a.embedding}-{a.cell}-{a.impl}"
    train = TrainConfig(
        epochs=a.epochs,
        lr=a.lr,
        min_lr=a.min_lr,
        gamma=a.gamma,
        weight_decay=a.weight_decay,
        label_smoothing=a.label_smoothing,
        max_norm=a.max_norm,
        patience=a.patience,
        seed=a.seed,
        device=a.device,
        output_dir=out,
        overfit=a.overfit,
        max_steps_per_epoch=a.max_steps_per_epoch,
    )
    return Config(data, model, train)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    run(config_from_args(args))


if __name__ == "__main__":
    main()
