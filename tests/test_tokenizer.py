import torch

from textcls import tokenizer as T
from textcls.tokenizer import GloveTokenizer, WordTokenizer, load_tokenizer, save_tokenizer


def test_special_ids_and_frequency_order():
    tok = WordTokenizer().fit(["b a a", "a c b"])
    assert tok.itos[:2] == ["<pad>", "<oov>"]
    assert tok.encode("a b c") == [2, 3, 4]  # a (3x) before b (2x) before c (1x)
    assert tok.encode("zzz") == [tok.oov_id]


def test_num_words_caps_vocab_like_keras():
    """Keras keeps ids < num_words; with <pad>=0 and <oov>=1 that is num_words - 2 real words."""
    tok = WordTokenizer(num_words=4).fit(["a a a b b c"])
    assert tok.vocab_size == 4
    assert tok.encode("a b c") == [2, 3, tok.oov_id]


def test_encode_decode_roundtrip():
    tok = WordTokenizer().fit(["test simple test"])
    ids = tok.encode("test simple test ismail")
    assert tok.decode(ids) == "test simple test <oov>"
    assert tok.decode(ids + [0, 0]) == "test simple test <oov>"  # padding is dropped


def test_save_load(tmp_path):
    tok = WordTokenizer(10).fit(["x y z"])
    save_tokenizer(tok, tmp_path / "tok.json")
    tok2 = load_tokenizer(tmp_path / "tok.json")
    assert tok2.itos == tok.itos and tok2.encode("y q") == tok.encode("y q")
    g = GloveTokenizer(["the", "cat"])
    save_tokenizer(g, tmp_path / "g.json")
    g2 = load_tokenizer(tmp_path / "g.json")
    assert isinstance(g2, GloveTokenizer) and g2.encode("the dog cat") == [2, 1, 3]


def test_build_glove_keeps_only_needed_rows(monkeypatch):
    words = ["the", "cat", "sat", "unused"]
    vecs = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    monkeypatch.setattr(T, "_glove_table", lambda dim, cache: (words, vecs))
    tok, w = T.build_glove(["the cat", "cat dog"], dim=4, cache_dir="unused")
    assert tok.itos == ["<pad>", "<oov>", "the", "cat"]
    assert torch.equal(w[0], torch.zeros(4))  # <pad>
    assert torch.allclose(w[1], vecs.mean(0))  # <oov> = mean vector
    assert torch.equal(w[2], vecs[0]) and torch.equal(w[3], vecs[1])
    assert tok.encode("dog") == [tok.oov_id]
