"""Classify sentences with a trained checkpoint: the `textcls-predict` command.

textcls-predict --checkpoint checkpoints/onehot-lstm-torch/best.pt "i feel so alone today"
"""

from __future__ import annotations

import argparse

import torch

from .config import CLASSES
from .data import clean_text, simplify_text
from .train import load_checkpoint


class Predictor:
    """Runs the exact same preprocessing the model was trained with."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.model, self.tok, self.cfg, _ = load_checkpoint(checkpoint, device)
        self.device = device

    def prepare(self, text: str) -> list[int]:
        text = clean_text(text)
        if self.cfg.data.simplify:
            text = simplify_text(text)
        ids = self.tok.encode(text)[: self.cfg.data.max_seq_len]
        return ids or [self.tok.oov_id]

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> torch.Tensor:
        """-> [len(texts), 6] probabilities, columns in CLASSES order."""
        ids = [self.prepare(t) for t in texts]
        lengths = torch.tensor([len(i) for i in ids])
        x = torch.full((len(ids), int(lengths.max())), self.tok.pad_id, dtype=torch.long)
        for r, seq in enumerate(ids):
            x[r, : len(seq)] = torch.tensor(seq)
        return torch.softmax(self.model(x.to(self.device), lengths), dim=-1).cpu()

    def predict(self, texts: list[str]) -> list[str]:
        return [CLASSES[i] for i in self.predict_proba(texts).argmax(1).tolist()]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Predict the emotion of one or more sentences.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("texts", nargs="+")
    a = p.parse_args(argv)
    pred = Predictor(a.checkpoint, a.device)
    for text, probs in zip(a.texts, pred.predict_proba(a.texts)):
        top = probs.argsort(descending=True)[:3].tolist()
        alts = ", ".join(f"{CLASSES[i]} {probs[i]:.2f}" for i in top[1:])
        print(f"{CLASSES[top[0]]:<9} {probs[top[0]]:.2f}  ({alts})  <- {text}")


if __name__ == "__main__":
    main()
