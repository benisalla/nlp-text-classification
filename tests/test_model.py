import itertools

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from textcls.config import CELLS, EMBEDDINGS, IMPLS, ModelConfig
from textcls.model import (
    GRUCell,
    LSTMCell,
    OneHotEmbedding,
    RNNCell,
    ScratchRNN,
    TextClassifier,
    TorchRNN,
    reverse_padded,
)

V, C = 30, 6
ALL = list(itertools.product(EMBEDDINGS, CELLS, IMPLS))


def make(embedding, cell, impl, **kw):
    cfg = ModelConfig(embedding=embedding, cell=cell, impl=impl, n_emb=50, h_dim=16, n_layers=2, dropout=0.0, **kw)
    glove = torch.randn(V, 50) if embedding == "glove" else None
    return TextClassifier(cfg, V, C, glove).eval()


def batch():
    lengths = torch.tensor([5, 2, 7, 1])
    x = torch.zeros(4, 7, dtype=torch.long)
    for i, L in enumerate(lengths):
        x[i, :L] = torch.randint(2, V, (int(L),))
    return x, lengths


@pytest.mark.parametrize("embedding,cell,impl", ALL)
def test_output_shape_all_12_models(embedding, cell, impl):
    x, lengths = batch()
    assert make(embedding, cell, impl)(x, lengths).shape == (4, C)


@pytest.mark.parametrize("embedding,cell,impl", ALL)
def test_padding_does_not_change_the_prediction(embedding, cell, impl):
    """The core bug of the notebooks: extra <pad> tokens must not change the logits."""
    m = make(embedding, cell, impl)
    x, lengths = batch()
    x_more_pad = torch.cat([x, torch.zeros(4, 9, dtype=torch.long)], dim=1)
    torch.testing.assert_close(m(x, lengths), m(x_more_pad, lengths))


@pytest.mark.parametrize("cell", CELLS)
def test_each_sample_is_independent_of_the_batch(cell):
    m = make("onehot", cell, "scratch")
    x, lengths = batch()
    full = m(x, lengths)
    one = m(x[2:3, : lengths[2]], lengths[2:3])
    torch.testing.assert_close(full[2:3], one)


@pytest.mark.parametrize(
    "ours,theirs",
    [
        (lambda: RNNCell(5, 7, "tanh"), lambda: nn.RNNCell(5, 7, nonlinearity="tanh")),
        (lambda: RNNCell(5, 7, "relu"), lambda: nn.RNNCell(5, 7, nonlinearity="relu")),
        (lambda: LSTMCell(5, 7), lambda: nn.LSTMCell(5, 7)),
        (lambda: GRUCell(5, 7), lambda: nn.GRUCell(5, 7)),
    ],
)
def test_scratch_cell_matches_pytorch_cell(ours, theirs):
    """Copy PyTorch's weights into our cell: same input -> same output."""
    a, b = ours(), theirs()
    with torch.no_grad():
        a.i2h.weight.copy_(b.weight_ih)
        a.i2h.bias.copy_(b.bias_ih)
        a.h2h.weight.copy_(b.weight_hh)
        a.h2h.bias.copy_(b.bias_hh)
    x, h, c = torch.randn(3, 5), torch.randn(3, 7), torch.randn(3, 7)
    if isinstance(a, LSTMCell):
        h_a, c_a = a(x, (h, c))
        h_b, c_b = b(x, (h, c))
        torch.testing.assert_close(c_a, c_b)
    else:
        h_a, h_b = a(x, h), b(x, h)
    torch.testing.assert_close(h_a, h_b)


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("bidirectional", [False, True])
def test_scratch_network_matches_torch_network(cell, bidirectional):
    """Whole encoder: 2 layers, (bi)directional, variable lengths -> identical to nn.LSTM & co."""
    torch_rnn = TorchRNN(cell, 6, 8, 2, bidirectional, 0.0, "tanh").eval()
    ours = ScratchRNN(cell, 6, 8, 2, bidirectional, 0.0, "tanh", False).eval()
    for layer in range(2):
        for d in range(2 if bidirectional else 1):
            sfx = f"_l{layer}" + ("_reverse" if d else "")
            c = ours.cells[layer][d]
            with torch.no_grad():
                c.i2h.weight.copy_(getattr(torch_rnn.rnn, "weight_ih" + sfx))
                c.i2h.bias.copy_(getattr(torch_rnn.rnn, "bias_ih" + sfx))
                c.h2h.weight.copy_(getattr(torch_rnn.rnn, "weight_hh" + sfx))
                c.h2h.bias.copy_(getattr(torch_rnn.rnn, "bias_hh" + sfx))
    x, lengths = torch.randn(4, 7, 6), torch.tensor([7, 3, 5, 1])
    torch.testing.assert_close(ours(x, lengths), torch_rnn(x, lengths), atol=1e-5, rtol=1e-4)


def test_reverse_padded():
    x = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]).unsqueeze(-1)
    out = reverse_padded(x, torch.tensor([3, 2])).squeeze(-1)
    assert out.tolist() == [[3, 2, 1, 0], [5, 4, 0, 0]]


def test_onehot_embedding_equals_the_notebooks_one_hot_matmul():
    emb = OneHotEmbedding(V, 12)
    x = torch.randint(0, V, (3, 5))
    notebook = emb.proj(F.one_hot(x, num_classes=V).float())
    torch.testing.assert_close(emb(x), notebook)


def test_glove_embedding_is_frozen():
    m = make("glove", "lstm", "torch")
    assert not m.emb.emb.weight.requires_grad
    assert all(p.requires_grad for n, p in m.named_parameters() if not n.startswith("emb."))


@pytest.mark.parametrize("embedding,cell,impl", ALL)
def test_every_trainable_parameter_gets_a_gradient(embedding, cell, impl):
    m = make(embedding, cell, impl).train()
    x, lengths = batch()
    F.cross_entropy(m(x, lengths), torch.tensor([0, 1, 2, 3])).backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_residual_option_still_runs_and_ignores_padding():
    m = make("onehot", "gru", "scratch", residual=True)
    x, lengths = batch()
    x2 = torch.cat([x, torch.zeros(4, 3, dtype=torch.long)], 1)
    torch.testing.assert_close(m(x, lengths), m(x2, lengths))


def test_lengths_are_inferred_from_padding_when_missing():
    m = make("onehot", "lstm", "torch")
    x, lengths = batch()
    torch.testing.assert_close(m(x), m(x, lengths))
