import random

import pytest
import torch

from textcls.config import CLASSES

# Tiny fake corpus: each class has its own trigger words, so a model CAN learn it.
TRIGGERS = {
    "sadness": ["sad", "lonely", "hopeless"],
    "surprise": ["amazed", "shocked", "surprised"],
    "anger": ["angry", "furious", "annoyed"],
    "love": ["loving", "romantic", "tender"],
    "joy": ["happy", "glad", "delighted"],
    "fear": ["scared", "afraid", "nervous"],
}
FILLER = ["i", "feel", "really", "so", "today", "about", "this", "the", "day", "very"]


def make_corpus(n: int, seed: int = 0):
    rng = random.Random(seed)
    texts, labels = [], []
    for k in range(n):
        label = CLASSES[k % len(CLASSES)]
        words = rng.choices(FILLER, k=rng.randint(2, 8))
        words.insert(rng.randint(0, len(words)), rng.choice(TRIGGERS[label]))
        texts.append(" ".join(words))
        labels.append(label)
    return texts, labels


@pytest.fixture
def corpus():
    return make_corpus(120)


@pytest.fixture
def data_dir(tmp_path):
    """train/val/test.txt in the notebooks' `text;label` format."""
    for split, n, seed in [("train", 240, 0), ("val", 60, 1), ("test", 60, 2)]:
        texts, labels = make_corpus(n, seed)
        (tmp_path / f"{split}.txt").write_text("".join(f"{t};{lbl}\n" for t, lbl in zip(texts, labels)))
    return tmp_path


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    random.seed(0)
