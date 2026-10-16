"""Embedding -> recurrent encoder -> MLP head.

The notebooks had 12 model classes (OneHotPyRNN, OneHotRNN, GloveLSTM, ...).
They all share one shape, so here they are one class with three switches:

    embedding  onehot | glove      how a token id becomes a vector
    cell       rnn | lstm | gru    which recurrence
    impl       torch | scratch     nn.LSTM etc., or our own cells below

    notebook name        TextClassifier(...)
    OneHotPyLSTM         embedding="onehot", cell="lstm", impl="torch"
    GloveGRU             embedding="glove",  cell="gru",  impl="scratch"

Shapes: B = batch, T = time (tokens), E = n_emb, H = h_dim, D = 1 or 2 directions.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .config import ModelConfig

# ----------------------------------------------------------------------------- embeddings


class OneHotEmbedding(nn.Module):
    """The notebook did `nn.Linear(v_size, n_emb)(F.one_hot(x))`.

    A one-hot vector times a matrix just selects one row (plus the bias), so we
    do exactly that with an index instead of building a [B, T, V] tensor of
    zeros. Same parameters, same output (tests/test_model.py checks it), far
    less memory.
    """

    def __init__(self, v_size: int, n_emb: int):
        super().__init__()
        self.v_size = v_size
        self.proj = nn.Linear(v_size, n_emb)  # same layer as the notebook

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T] -> [B, T, E]
        return self.proj.weight.t()[x] + self.proj.bias


class GloveEmbedding(nn.Module):
    """Frozen pretrained GloVe vectors (the notebook's GloveEmbedder)."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x)


# ----------------------------------------------------------------------------- cells (from scratch)
# One time step each: (x_t [B, E], state) -> new state. Gate order and formulas
# follow PyTorch exactly, so a cell can be checked against nn.RNNCell /
# nn.LSTMCell / nn.GRUCell with copied weights (tests/test_model.py).


class RNNCell(nn.Module):
    """h' = act(W_ih x + W_hh h + b)"""

    def __init__(self, in_dim: int, h_dim: int, act: str = "relu", residual: bool = False):
        super().__init__()
        self.h_dim = h_dim
        self.i2h = nn.Linear(in_dim, h_dim)
        self.h2h = nn.Linear(h_dim, h_dim)
        self.act = nn.ReLU() if act == "relu" else nn.Tanh()
        self.res = residual_proj(in_dim, h_dim) if residual else None
        nn.init.orthogonal_(self.h2h.weight)  # keeps ||h|| stable through time

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.size(0), self.h_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h_new = self.act(self.i2h(x) + self.h2h(h))
        if self.res is not None:
            h_new = h_new + 0.5 * self.res(x)
        return h_new


class LSTMCell(nn.Module):
    """i, f, g, o = gates(x, h);  c' = f*c + i*g;  h' = o*tanh(c')"""

    def __init__(self, in_dim: int, h_dim: int, residual: bool = False):
        super().__init__()
        self.h_dim = h_dim
        self.i2h = nn.Linear(in_dim, 4 * h_dim)
        self.h2h = nn.Linear(h_dim, 4 * h_dim)
        self.res = residual_proj(in_dim, h_dim) if residual else None
        with torch.no_grad():  # forget-gate bias = 1: remember by default (as in OneHotPyLSTM)
            self.i2h.bias[h_dim : 2 * h_dim].fill_(1.0)

    def init_state(self, x):
        z = x.new_zeros(x.size(0), self.h_dim)
        return (z, z)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = (self.i2h(x) + self.h2h(h)).chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        if self.res is not None:
            h_new = h_new + 0.5 * self.res(x)
        return (h_new, c_new)


class GRUCell(nn.Module):
    """r, z = gates(x, h);  n = tanh(W_in x + r * (W_hn h));  h' = (1-z)*n + z*h

    The notebook computed n = tanh(r * (W_in x + W_hn h) + (1 - r) * h): the
    reset gate scaled the input too and the old state leaked into n. Fixed.
    """

    def __init__(self, in_dim: int, h_dim: int, residual: bool = False):
        super().__init__()
        self.h_dim = h_dim
        self.i2h = nn.Linear(in_dim, 3 * h_dim)
        self.h2h = nn.Linear(h_dim, 3 * h_dim)
        self.res = residual_proj(in_dim, h_dim) if residual else None

    def init_state(self, x):
        return x.new_zeros(x.size(0), self.h_dim)

    def forward(self, x, h):
        xr, xz, xn = self.i2h(x).chunk(3, dim=1)
        hr, hz, hn = self.h2h(h).chunk(3, dim=1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        h_new = (1 - z) * n + z * h
        if self.res is not None:
            h_new = h_new + 0.5 * self.res(x)
        return h_new


def residual_proj(in_dim: int, h_dim: int) -> nn.Module:
    """The notebooks' skip connection: identity when sizes match, else a projection."""
    return nn.Identity() if in_dim == h_dim else nn.Linear(in_dim, h_dim, bias=False)


def _hidden(state):
    return state[0] if isinstance(state, tuple) else state


def _select(mask, new, old):
    """Per sample: take the new state where mask is True (a real token), keep the old one on padding."""
    if isinstance(new, tuple):
        return tuple(_select(mask, n, o) for n, o in zip(new, old))
    return torch.where(mask, new, old)


def reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverse each sequence inside its own length; padding stays at the end.
    [a b c PAD] -> [c b a PAD]"""
    T = x.size(1)
    t = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
    L = lengths.to(x.device).unsqueeze(1)  # [B, 1]
    idx = torch.where(t < L, L - 1 - t, t)  # [B, T]
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class ScratchRNN(nn.Module):
    """A multi-layer, optionally bidirectional recurrent network built from our cells.

    The notebooks' RNNBlock/LSTMBlock/GRUBlock created ONE CELL PER TIME STEP
    (n_unit = max_seq_len), so step 5 and step 6 had different weights: that is
    a 20-layer feed-forward net over positions, not an RNN, and it cannot share
    what it learns about a word across positions. Here one cell per layer and
    direction is reused at every step, which is what a recurrent network is.
    """

    def __init__(self, cell: str, in_dim, h_dim, n_layers, bidirectional, dropout, act, residual):
        super().__init__()
        self.n_dir = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)

        def make(i):
            if cell == "rnn":
                return RNNCell(i, h_dim, act, residual)
            return {"lstm": LSTMCell, "gru": GRUCell}[cell](i, h_dim, residual)

        self.cells = nn.ModuleList(
            nn.ModuleList(make(in_dim if layer == 0 else h_dim * self.n_dir) for _ in range(self.n_dir))
            for layer in range(n_layers)
        )

    def _run(self, cell, x, lengths):
        """One direction of one layer. x [B, T, I] -> outputs [B, T, H], last real hidden [B, H]."""
        state = cell.init_state(x[:, 0])
        outs = []
        for t in range(x.size(1)):
            new = cell(x[:, t], state)
            mask = (t < lengths).to(x.device).unsqueeze(1)  # [B, 1]: still inside the sentence?
            state = _select(mask, new, state)
            outs.append(_hidden(state))
        return torch.stack(outs, dim=1), _hidden(state)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x [B, T, E] -> [B, D*H]: last layer's final state (both directions concatenated)."""
        for li, layer in enumerate(self.cells):
            if li > 0:
                x = self.dropout(x)  # between layers, like nn.LSTM(dropout=...)
            out_f, h_f = self._run(layer[0], x, lengths)
            if self.n_dir == 1:
                x, finals = out_f, [h_f]
            else:
                out_b, h_b = self._run(layer[1], reverse_padded(x, lengths), lengths)
                x = torch.cat([out_f, reverse_padded(out_b, lengths)], dim=-1)
                finals = [h_f, h_b]
        return torch.cat(finals, dim=-1)


class TorchRNN(nn.Module):
    """nn.RNN / nn.LSTM / nn.GRU with packed sequences, so padding is skipped.

    The notebooks fed the padded batch straight in and read h_n: with
    right-padding, h_n was the state after up to 17 <pad> tokens, and the
    backward direction started on padding.
    """

    def __init__(self, cell, in_dim, h_dim, n_layers, bidirectional, dropout, act):
        super().__init__()
        kw = dict(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        if cell == "rnn":
            self.rnn = nn.RNN(nonlinearity=act, **kw)
        else:
            self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell](**kw)
        self.n_dir = 2 if bidirectional else 1
        self._init_weights(cell)

    def _init_weights(self, cell):
        # The notebook's _init_weights (xavier input, orthogonal recurrent, forget bias 1).
        # It tested `isinstance(module, nn.LSTM)` inside the GRU class too, so the GRU
        # never got it; here it applies to every cell.
        for name, p in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                if cell == "lstm" and "bias_ih" in name:
                    h = p.numel() // 4
                    with torch.no_grad():
                        p[h : 2 * h].fill_(1.0)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        if isinstance(h_n, tuple):  # LSTM returns (h_n, c_n)
            h_n = h_n[0]
        # h_n: [layers * D, B, H]; the last D rows belong to the top layer
        return torch.cat(list(h_n[-self.n_dir :]), dim=-1)  # [B, D*H]


class TextClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int, n_class: int, glove_weight=None):
        super().__init__()
        self.cfg = cfg
        if cfg.embedding == "onehot":
            self.emb = OneHotEmbedding(vocab_size, cfg.n_emb)
        else:
            if glove_weight is None:  # e.g. rebuilding before load_state_dict
                glove_weight = torch.zeros(vocab_size, cfg.n_emb)
            self.emb = GloveEmbedding(glove_weight)

        args = (cfg.cell, cfg.n_emb, cfg.h_dim, cfg.n_layers, cfg.bidirectional, cfg.dropout, cfg.act)
        if cfg.impl == "torch":
            self.rnn = TorchRNN(*args)
        else:
            self.rnn = ScratchRNN(*args, cfg.residual)

        d = cfg.h_dim * (2 if cfg.bidirectional else 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Sequential(nn.Linear(d, 2 * cfg.h_dim), nn.ReLU(), nn.Linear(2 * cfg.h_dim, n_class))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x [B, T] token ids (0 = pad), lengths [B] -> logits [B, n_class]"""
        if lengths is None:
            lengths = (x != 0).sum(1).clamp(min=1)
        h = self.rnn(self.emb(x), lengths)  # [B, D*H]
        return self.fc(self.dropout(h))


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
