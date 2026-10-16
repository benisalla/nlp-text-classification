from collections import Counter

import torch

from textcls.config import CLASSES, Config, DataConfig, ModelConfig
from textcls.data import (
    TextDataset,
    build_word_freq,
    clean_text,
    encode_labels,
    expand_contractions,
    load_file,
    pad_collate_fn,
    remove_less_freq_words,
    text_stats,
)
from textcls.tokenizer import WordTokenizer


def test_clean_text_lowercases_and_strips_symbols():
    assert clean_text("  I LOVE it!!! :) #yes  ") == "i love it!!! : yes"


def test_expand_contractions_whole_words_only():
    assert expand_contractions("i didnt know") == "i did not know"
    assert expand_contractions("idea") == "idea"  # "id" inside a word is untouched
    assert expand_contractions("they were here") == "they were here"  # the removed "were" rule


def test_load_file_reads_text_semicolon_label(tmp_path):
    p = tmp_path / "train.txt"
    p.write_text("i didnt feel humiliated;sadness\ni am happy; really;joy\n\n")
    x, y = load_file(p)
    assert x == ["i didnt feel humiliated", "i am happy really"]
    assert y == ["sadness", "joy"]


def test_rare_words_use_train_counts_and_never_drop_texts():
    train = ["a a a b", "a c"]
    freq = build_word_freq(train)
    assert freq == Counter({"a": 4, "b": 1, "c": 1})
    out = remove_less_freq_words(["b c", "a z"], freq, threshold=1)
    assert out == ["", "a"]  # empty text is KEPT (eval sets must not shrink)


def test_label_encoding_follows_notebook_order():
    assert CLASSES == ["sadness", "surprise", "anger", "love", "joy", "fear"]
    assert encode_labels(["sadness", "fear", "joy"]) == [0, 5, 4]


def test_unknown_label_raises():
    import pytest

    with pytest.raises(ValueError):
        encode_labels(["boredom"])


def test_dataset_truncates_and_filters_only_when_asked(corpus):
    texts, _ = corpus
    tok = WordTokenizer().fit(texts)
    ds = TextDataset(["a b c d e f", "x", ""], [0, 1, 2], tok, max_len=4, min_len=0)
    assert len(ds) == 3
    assert all(1 <= len(ids) <= 4 for ids, _ in ds)
    assert ds[2][0] == [tok.oov_id]  # empty text -> one <oov>
    assert len(TextDataset(["a b c", "x"], [0, 1], tok, max_len=4, min_len=2)) == 1


def test_collate_pads_right_and_returns_lengths():
    x, lengths, y = pad_collate_fn([([5, 6, 7], 1), ([8], 0)], pad_id=0)
    assert x.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert lengths.tolist() == [3, 1]
    assert y.tolist() == [1, 0]
    assert x.dtype == y.dtype == torch.long


def test_text_stats():
    s = text_stats(["a b", "a b c d"])
    assert (s["min_len"], s["max_len"], s["vocab_size"], s["mean_len"]) == (2, 4, 4, 3.0)


def test_config_defaults_follow_the_two_notebooks():
    oh = Config(DataConfig(), ModelConfig(embedding="onehot"))
    gl = Config(DataConfig(), ModelConfig(embedding="glove"))
    assert (oh.data.simplify, oh.data.max_seq_len, oh.data.min_freq) == (True, 20, 3)
    assert (gl.data.simplify, gl.data.max_seq_len, gl.data.min_freq) == (False, 40, 0)


def test_config_roundtrip_and_validation():
    import pytest

    cfg = Config()
    assert Config.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError):
        ModelConfig(cell="transformer")
    with pytest.raises(ValueError):
        ModelConfig(embedding="glove", n_emb=64)
