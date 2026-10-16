"""Two word-level tokenizers, text -> list of ints.

WordTokenizer   replaces the notebook's ZeroTokenizer (which wrapped the Keras
                Tokenizer, so TensorFlow was needed just to split words).
                Same behaviour: id 0 = <pad>, id 1 = <oov>, then words from
                most to least frequent, and only ids < num_words are kept.
GloveTokenizer  vocabulary = words that have a GloVe vector.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

import torch

PAD, OOV = "<pad>", "<oov>"
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"


class WordTokenizer:
    pad_id = 0
    oov_id = 1

    def __init__(self, num_words: int | None = None):
        self.num_words = num_words
        self.itos: list[str] = [PAD, OOV]
        self.stoi: dict[str, int] = {PAD: 0, OOV: 1}

    def fit(self, texts: list[str]) -> WordTokenizer:
        counts = Counter(w for t in texts for w in t.split())
        # most frequent first; ties keep first-seen order (Counter preserves insertion order)
        words = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        if self.num_words is not None:
            words = words[: self.num_words - 2]
        self.itos = [PAD, OOV] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.oov_id) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.itos[i] if 0 <= i < len(self.itos) else OOV for i in ids if i != self.pad_id)

    def to_dict(self) -> dict:
        return {"kind": "word", "num_words": self.num_words, "itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> WordTokenizer:
        tok = cls(d["num_words"])
        tok.itos = list(d["itos"])
        tok.stoi = {w: i for i, w in enumerate(tok.itos)}
        return tok


class GloveTokenizer(WordTokenizer):
    """Same interface; the vocabulary is the list of GloVe words we kept."""

    oov_id = 1  # "<oov>" plays the role of the notebook's "<unk>"

    def __init__(self, words: list[str]):
        super().__init__(None)
        self.itos = [PAD, OOV] + list(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def fit(self, texts):  # nothing to learn, the vocabulary comes from GloVe
        return self

    def to_dict(self) -> dict:
        return {"kind": "glove", "itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> GloveTokenizer:
        return cls(d["itos"][2:])


def tokenizer_from_dict(d: dict) -> WordTokenizer:
    return GloveTokenizer.from_dict(d) if d["kind"] == "glove" else WordTokenizer.from_dict(d)


# ----------------------------------------------------------------------------- GloVe


def _glove_table(dim: int, cache_dir: str) -> tuple[list[str], torch.Tensor]:
    """All 400k GloVe 6B words and their vectors. Downloads the zip (862 MB) once,
    then keeps a fast .pt copy of the requested dimension."""
    cache = Path(os.path.expanduser(cache_dir))
    cache.mkdir(parents=True, exist_ok=True)
    fast = cache / f"glove.6B.{dim}d.pt"
    if fast.exists():
        d = torch.load(fast, weights_only=True)
        return d["words"], d["vectors"]

    zpath = cache / "glove.6B.zip"
    if not zpath.exists():
        import urllib.request

        print(f"downloading GloVe 6B (862 MB) -> {zpath}")
        tmp = zpath.with_suffix(".part")
        urllib.request.urlretrieve(GLOVE_URL, tmp)
        tmp.rename(zpath)

    words, rows = [], []
    with zipfile.ZipFile(zpath) as zf, zf.open(f"glove.6B.{dim}d.txt") as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            parts = line.rstrip().split(" ")
            words.append(parts[0])
            rows.append([float(v) for v in parts[1:]])
    vectors = torch.tensor(rows, dtype=torch.float32)
    torch.save({"words": words, "vectors": vectors}, fast)
    return words, vectors


def build_glove(texts: list[str], dim: int, cache_dir: str) -> tuple[GloveTokenizer, torch.Tensor]:
    """Keep only the GloVe rows for words that occur in `texts`.

    Row 0 = <pad> (zeros), row 1 = <oov> (mean of all GloVe vectors, as in the
    notebook). The vectors are frozen, so choosing which rows to keep uses no
    label information; it just avoids carrying a 400k x dim table around.
    """
    all_words, all_vectors = _glove_table(dim, cache_dir)
    needed = {w for t in texts for w in t.split()}
    keep = [i for i, w in enumerate(all_words) if w in needed]
    tok = GloveTokenizer([all_words[i] for i in keep])
    weight = torch.cat([torch.zeros(1, dim), all_vectors.mean(0, keepdim=True), all_vectors[keep]])
    return tok, weight


def save_tokenizer(tok: WordTokenizer, path: str | Path) -> None:
    Path(path).write_text(json.dumps(tok.to_dict()))


def load_tokenizer(path: str | Path) -> WordTokenizer:
    return tokenizer_from_dict(json.loads(Path(path).read_text()))
