"""Load the emotion dataset, clean it and turn it into padded batches.

Pipeline (same steps as the notebooks):

    raw line "i didnt feel humiliated;sadness"
      -> clean_text            lower-case, strip odd characters
      -> remove_less_freq_words drop words seen <= min_freq times in TRAIN
      -> simplify_text          (optional) contractions, stop words, POS filter, lemmas
      -> tokenizer.encode       list of ints
      -> pad_collate_fn         (x [B, T], lengths [B], y [B])
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import CLASSES

HF_URL = "https://huggingface.co/datasets/dair-ai/emotion/resolve/main/split/{split}-00000-of-00001.parquet"
HF_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]  # label ids used on the HF hub
SPLITS = {"train": "train", "val": "validation", "test": "test"}

# fmt: off
# Word lists from the notebook. One fix: "were" -> "we are" was removed, it turned
# "they were happy" into "they we are happy".
contractions = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
    "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will",
    "it's": "it is", "let's": "let us", "ma'am": "madam", "mightn't": "might not", "mustn't": "must not", "o'clock": "of the clock", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "should've": "should have", "shouldn't": "should not", "that's": "that is", "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who's": "who is", "who've": "who have", "why's": "why is", "willn't": "will not", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have", "y'all": "you all",
    "ive": "i have", "youve": "you have", "yall": "you all", "dont": "do not", "doesnt": "does not", "havent": "have not", "hasnt": "has not",
    "isnt": "is not", "wasnt": "was not", "werent": "were not", "hadnt": "had not", "shant": "shall not", "mightnt": "might not", "mustnt": "must not",
    "neednt": "need not", "didnt": "did not", "im": "i am", "id": "i would", "youre": "you are", "hes": "he is", "shes": "she is", "its": "it is",
    "theyre": "they are", "weve": "we have", "youll": "you will", "youd": "you would",
    "kinda": "kind of", "sorta": "sort of", "lemme": "let me", "dunno": "do not know",
}

more_stop_words = [
    'across', 'also', 'among', 'around', 'beside', 'besides', 'beyond', 'could', 'every', 'except', 'get', 'go', 'going', 'gone', 'got', 'keep', 'kept',
    'like', 'made', 'make', 'makes', 'many', 'much', 'near', 'one', 'per', 'since', 'take', 'taken', 'throughout', 'unless', 'upon', 'via', 'whenever',
    'whichever', 'within', 'without', 'would', 'yet',
]

not_stop_words = [
    "not", "no", "nor", "never", "none",                          # critical for reversing sentiment
    "what", "how", "why", "when", "where",                        # often imply surprise, doubt, or curiosity
    "but", "if", "because", "although", "though",                 # influence sentiment structure
]
# fmt: on

PUNCT = set(".,!?;:")
PIPELINE_VERSION = 1  # bump when cleaning / simplify_text change, to invalidate caches
_contractions_re = re.compile(r"\b(" + "|".join(map(re.escape, contractions)) + r")\b")


# ----------------------------------------------------------------------------- cleaning


def clean_text(text: str) -> str:
    """Keep letters, digits, spaces and . , ! ? ; : then lower-case."""
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:]", "", text)
    return " ".join(text.lower().split())


def expand_contractions(text: str) -> str:
    return _contractions_re.sub(lambda m: contractions[m.group(0)], text)


def build_word_freq(texts: list[str]) -> Counter:
    return Counter(w for t in texts for w in t.split())


def remove_less_freq_words(texts: list[str], word_freq: Counter, threshold: int) -> list[str]:
    """Drop words seen <= threshold times. `word_freq` must come from the TRAIN set only.

    (The notebooks counted words separately inside val/test; counting on train
    is what a deployed model would see.) Texts are never dropped here, even if
    they become empty: the tokenizer maps an empty text to a single <oov>.
    """
    if threshold <= 0:
        return list(texts)
    return [" ".join(w for w in t.split() if word_freq[w] > threshold) for t in texts]


@lru_cache(maxsize=1)
def _nltk_tools():
    """Load NLTK once (the notebook rebuilt the stop-word set for every sentence)."""
    import nltk

    nltk_dir = os.path.expanduser("~/.cache/textcls/nltk_data")  # private dir (nltk refuses shared ones)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)
    for res, path in [
        ("stopwords", "corpora/stopwords"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
        ("wordnet", "corpora/wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res, download_dir=nltk_dir, quiet=True)
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop = set(stopwords.words("english")).union(more_stop_words) - set(not_stop_words)
    return stop, word_tokenize, pos_tag, WordNetLemmatizer()


def simplify_text(text: str) -> str:
    """Shorten a sentence for the RNN: expand contractions, drop stop words, keep only
    adjectives / adverbs / nouns / verbs / modals (+ punctuation), lemmatize.

    "i didnt feel humiliated" -> "not feel humiliated"
    """
    stop, word_tokenize, pos_tag, lemmatizer = _nltk_tools()
    text = clean_text(expand_contractions(clean_text(text)))
    words = word_tokenize(text)
    kept = [
        w for w, pos in pos_tag(words) if w in PUNCT or (w not in stop and pos.startswith(("J", "R", "N", "V", "MD")))
    ]
    return " ".join(w if w in PUNCT else lemmatizer.lemmatize(w) for w in kept if w.strip())


# ----------------------------------------------------------------------------- loading


def load_file(path: str | Path) -> tuple[list[str], list[str]]:
    """Read the notebooks' format: one `text;label` per line."""
    x, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split(";")
            x.append(clean_text(" ".join(parts[:-1])))
            y.append(clean_text(parts[-1]))
    return x, y


def download_split(split: str, cache_dir: str | Path) -> tuple[list[str], list[str]]:
    """Fetch one split of dair-ai/emotion (same 16k/2k/2k data as the notebooks) from the HF hub."""
    import pandas as pd

    cache = Path(os.path.expanduser(str(cache_dir)))
    cache.mkdir(parents=True, exist_ok=True)
    hf_split = SPLITS[split]
    path = cache / f"emotion-{hf_split}.parquet"
    if not path.exists():
        print(f"downloading {hf_split} split -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(HF_URL.format(split=hf_split), tmp)
        tmp.rename(path)
    df = pd.read_parquet(path)
    return [clean_text(t) for t in df["text"]], [HF_LABELS[i] for i in df["label"]]


def load_splits(data_dir: str | None, cache_dir: str) -> dict[str, tuple[list[str], list[str]]]:
    """{"train": (texts, labels), "val": ..., "test": ...}"""
    if data_dir:
        return {s: load_file(Path(data_dir) / f"{s}.txt") for s in SPLITS}
    return {s: download_split(s, cache_dir) for s in SPLITS}


def encode_labels(labels: list[str]) -> list[int]:
    unknown = sorted(set(labels) - set(CLASSES))
    if unknown:
        raise ValueError(f"unknown labels {unknown}; expected {CLASSES}")
    return [CLASSES.index(lbl) for lbl in labels]


def preprocess(
    splits: dict, min_freq: int, simplify: bool, cache_dir: str | None = None
) -> dict[str, tuple[list[str], list[int]]]:
    """Apply the notebook pipeline to every split, with statistics taken from train only.

    NLTK tagging of 20k sentences takes minutes, so the result is cached in
    cache_dir, keyed by a hash of the input texts and the settings.
    """
    cache = None
    if cache_dir:
        key = hashlib.sha1(json.dumps([splits, min_freq, simplify, PIPELINE_VERSION]).encode()).hexdigest()[:16]
        cache = Path(os.path.expanduser(cache_dir)) / f"preprocessed-{key}.json"
        if cache.exists():
            return {k: tuple(v) for k, v in json.loads(cache.read_text()).items()}
    out = _preprocess(splits, min_freq, simplify)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
    return out


def _preprocess(splits, min_freq, simplify):
    freq = build_word_freq(splits["train"][0])
    out = {}
    for name, (texts, labels) in splits.items():
        texts = remove_less_freq_words(texts, freq, min_freq)
        if simplify:
            texts = [simplify_text(t) for t in texts]
        out[name] = (texts, encode_labels(labels))
    return out


def text_stats(texts: list[str]) -> dict:
    """The numbers get_statistics() printed in the notebook."""
    lens = [len(t.split()) for t in texts]
    freq = build_word_freq(texts)
    return {
        "n_texts": len(texts),
        "min_len": min(lens),
        "max_len": max(lens),
        "mean_len": sum(lens) / len(lens),
        "vocab_size": len(freq),
    }


# ----------------------------------------------------------------------------- batching


class TextDataset(Dataset):
    """(token ids, label) pairs. Texts are truncated to max_len.

    min_len drops short texts, which is only reasonable on the TRAIN set: the
    notebooks also applied it to val/test, so their scores were measured on a
    subset of the test set. Here eval sets keep every sample (pass min_len=0).
    """

    def __init__(self, texts, labels, tokenizer, max_len: int, min_len: int = 0):
        self.data = []
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text)
            if len(ids) < min_len:
                continue
            ids = ids[:max_len] or [tokenizer.oov_id]  # an empty text still needs one token
            self.data.append((ids, label))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def pad_collate_fn(batch, pad_id: int):
    """Right-pad to the longest text of the batch and also return the true lengths.

    The lengths are what lets the model read the hidden state at the LAST REAL
    token instead of after a run of <pad> tokens (bug in the notebooks).
    """
    ids, labels = zip(*batch)
    lengths = torch.tensor([len(i) for i in ids], dtype=torch.long)
    x = torch.full((len(ids), int(lengths.max())), pad_id, dtype=torch.long)
    for row, seq in enumerate(ids):
        x[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return x, lengths, torch.tensor(labels, dtype=torch.long)
