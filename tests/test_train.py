"""Training-loop tests and the neural-network sanity checks (Karpathy's recipe),
all on a tiny fake corpus so they run on CPU in seconds."""

import json
import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textcls.config import CLASSES, Config, DataConfig, ModelConfig, TrainConfig
from textcls.data import TextDataset, encode_labels, pad_collate_fn
from textcls.metrics import classification_report, confusion_matrix
from textcls.model import TextClassifier
from textcls.predict import Predictor
from textcls.tokenizer import WordTokenizer
from textcls.train import evaluate, get_lr, load_checkpoint, make_optimizer, run, train_one_epoch


def loader_for(corpus, batch_size=32, shuffle=True):
    texts, labels = corpus
    tok = WordTokenizer().fit(texts)
    ds = TextDataset(texts, encode_labels(labels), tok, max_len=20)
    return tok, DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=partial(pad_collate_fn, pad_id=0))


def tiny_model(tok, cell="lstm", impl="torch"):
    cfg = ModelConfig(cell=cell, impl=impl, n_emb=32, h_dim=32, n_layers=1, dropout=0.0)
    return TextClassifier(cfg, tok.vocab_size, len(CLASSES))


# ----------------------------------------------------------------------------- metrics


def test_confusion_matrix_and_macro_scores_by_hand():
    labels = torch.tensor([0, 0, 1, 1, 2])
    preds = torch.tensor([0, 1, 1, 1, 0])
    cm = confusion_matrix(preds, labels, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    rep = classification_report(preds, labels, 3)
    assert rep["accuracy"] == pytest.approx(3 / 5)
    # precision per class: 1/2, 2/3, 0 ; recall: 1/2, 1, 0
    assert rep["precision"] == pytest.approx((1 / 2 + 2 / 3 + 0) / 3)
    assert rep["recall"] == pytest.approx((1 / 2 + 1 + 0) / 3)
    f1 = [0.5, 2 * (2 / 3) / (2 / 3 + 1), 0.0]
    assert rep["f1"] == pytest.approx(sum(f1) / 3)


def test_lr_schedule_decays_to_floor():
    assert get_lr(0, 1e-2, 0.5, 1e-3) == 1e-2
    assert get_lr(1, 1e-2, 0.5, 1e-3) == 5e-3
    assert get_lr(10, 1e-2, 0.5, 1e-3) == 1e-3


# ----------------------------------------------------------------------------- sanity checks


@pytest.mark.parametrize("cell", ["rnn", "lstm", "gru"])
@pytest.mark.parametrize("impl", ["torch", "scratch"])
def test_initial_loss_is_log_num_classes(corpus, cell, impl):
    """At init the model should be unsure: loss ~= -ln(1/6) = 1.79. Much higher means
    over-confident wrong logits (bad init)."""
    tok, loader = loader_for(corpus)
    m = tiny_model(tok, cell, impl).eval()
    x, lengths, y = next(iter(loader))
    loss = F.cross_entropy(m(x, lengths), y).item()
    assert abs(loss - math.log(len(CLASSES))) < 0.25


@pytest.mark.parametrize("cell", ["rnn", "lstm", "gru"])
@pytest.mark.parametrize("impl", ["torch", "scratch"])
def test_can_overfit_one_batch(corpus, cell, impl):
    """If the model cannot memorise 32 sentences, something is broken."""
    tok, loader = loader_for(corpus)
    m = tiny_model(tok, cell, impl).train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x, lengths, y = next(iter(loader))
    for _ in range(150):
        opt.zero_grad()
        loss = F.cross_entropy(m(x, lengths), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    assert (m(x, lengths).argmax(1) == y).all()


def test_model_beats_input_independent_baseline(corpus):
    """Train twice: real inputs vs all inputs replaced by one token. Real inputs must win,
    proving the model actually uses the text and not only label frequencies."""
    tok, loader = loader_for(corpus, shuffle=False)
    results = {}
    for mode in ("real", "blind"):
        torch.manual_seed(0)
        m = tiny_model(tok).train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(15):
            for x, lengths, y in loader:
                if mode == "blind":
                    x = torch.full_like(x, 2)
                opt.zero_grad()
                F.cross_entropy(m(x, lengths), y).backward()
                opt.step()
        results[mode] = evaluate(m, loader, torch.nn.CrossEntropyLoss(), "cpu")["loss"]
    assert results["real"] < 0.5 * results["blind"]


def test_train_one_epoch_lowers_loss_and_reports_update_ratio(corpus):
    tok, loader = loader_for(corpus)
    m = tiny_model(tok)
    crit = torch.nn.CrossEntropyLoss()
    opt = make_optimizer(m, TrainConfig(lr=5e-3))
    first = evaluate(m, loader, crit, "cpu")["loss"]
    for _ in range(5):
        stats = train_one_epoch(m, loader, crit, opt, "cpu", max_norm=5.0)
    assert evaluate(m, loader, crit, "cpu")["loss"] < first
    assert -6 < stats["update_ratio"] < 0  # log10 of a small positive ratio


def test_max_steps_per_epoch(corpus):
    tok, loader = loader_for(corpus, batch_size=8)
    m = tiny_model(tok)
    opt = make_optimizer(m, TrainConfig())
    calls = []
    m.register_forward_hook(lambda *a: calls.append(1))
    train_one_epoch(m, loader, torch.nn.CrossEntropyLoss(), opt, "cpu", max_steps=3)
    assert len(calls) == 3


# ----------------------------------------------------------------------------- end to end


@pytest.mark.parametrize("impl", ["torch", "scratch"])
def test_full_run_writes_artifacts_and_predicts(data_dir, tmp_path, impl):
    out = tmp_path / "run"
    cfg = Config(
        DataConfig(data_dir=str(data_dir), cache_dir=str(tmp_path), simplify=False, min_freq=0, batch_size=32),
        ModelConfig(cell="gru", impl=impl, n_emb=32, h_dim=32, n_layers=1),
        TrainConfig(epochs=12, lr=1e-2, gamma=0.95, patience=12, output_dir=str(out)),
    )
    metrics = run(cfg, log=lambda *a: None)
    for f in ("best.pt", "metrics.json", "history.png", "confusion.png"):
        assert (out / f).exists(), f
    assert metrics["accuracy"] > 0.8  # the fake classes are easy
    assert json.loads((out / "metrics.json").read_text())["name"] == f"onehot-gru-{impl}"

    # reload in a "fresh process" and check predictions are identical
    _, _, cfg2, _ = load_checkpoint(out / "best.pt")
    assert cfg2 == cfg
    p = Predictor(str(out / "best.pt"))
    assert p.predict(["i feel so happy and glad", "i am scared and afraid"]) == ["joy", "fear"]
    probs = p.predict_proba(["whatever"])
    assert probs.shape == (1, 6) and torch.allclose(probs.sum(), torch.tensor(1.0))


def test_overfit_mode_uses_n_samples(data_dir, tmp_path):
    cfg = Config(
        DataConfig(
            data_dir=str(data_dir), cache_dir=str(tmp_path), simplify=False, min_freq=0, batch_size=8, min_seq_len=0
        ),
        ModelConfig(n_emb=32, h_dim=32, n_layers=1, dropout=0.0),
        TrainConfig(epochs=40, lr=1e-2, gamma=1.0, patience=40, overfit=8, output_dir=str(tmp_path / "o")),
    )
    m = run(cfg, log=lambda *a: None)
    assert sum(map(sum, m["confusion_matrix"])) == 8
    assert m["accuracy"] == 1.0
