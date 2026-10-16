"""Every hyperparameter in one place.

Defaults follow the original notebooks (`legacy/`): same class order, same
sequence lengths, same AdamW settings and the same exponential LR decay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Same order as the notebooks: class_cat = ['sadness', 'surprise', 'anger', 'love', 'joy', 'fear']
CLASSES = ["sadness", "surprise", "anger", "love", "joy", "fear"]

EMBEDDINGS = ("onehot", "glove")
CELLS = ("rnn", "lstm", "gru")
IMPLS = ("torch", "scratch")


@dataclass
class DataConfig:
    data_dir: str | None = None  # folder with train.txt / val.txt / test.txt ("text;label"); None = download
    cache_dir: str = "~/.cache/textcls"
    min_freq: int | None = None  # keep words seen MORE than this in train; None = 3 onehot, 0 glove
    simplify: bool | None = None  # stop words + POS filter + lemmatize; None = on for onehot, off for glove
    max_seq_len: int | None = None  # None = 20 for onehot (simplified text), 40 for glove
    min_seq_len: int = 3  # train only: drop very short texts (eval sets are never filtered)
    vocab_size: int = 3100  # onehot only: most frequent words kept (incl. <pad> and <oov>)
    batch_size: int = 64


@dataclass
class ModelConfig:
    embedding: str = "onehot"  # onehot | glove
    cell: str = "lstm"  # rnn | lstm | gru
    impl: str = "torch"  # torch (nn.LSTM, ...) | scratch (our own cells)
    n_emb: int = 100  # glove supports 50, 100, 200, 300
    h_dim: int = 128
    n_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1
    act: str = "relu"  # vanilla RNN only: relu | tanh
    residual: bool = False  # scratch only: the notebooks' "h + 0.5 * x" skip connection

    def __post_init__(self) -> None:
        if self.embedding not in EMBEDDINGS:
            raise ValueError(f"embedding must be one of {EMBEDDINGS}, got {self.embedding!r}")
        if self.cell not in CELLS:
            raise ValueError(f"cell must be one of {CELLS}, got {self.cell!r}")
        if self.impl not in IMPLS:
            raise ValueError(f"impl must be one of {IMPLS}, got {self.impl!r}")
        if self.act not in ("relu", "tanh"):
            raise ValueError(f"act must be relu or tanh, got {self.act!r}")
        if self.embedding == "glove" and self.n_emb not in (50, 100, 200, 300):
            raise ValueError(f"GloVe 6B comes in 50/100/200/300 dims, got n_emb={self.n_emb}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 2e-3
    min_lr: float = 1e-5
    gamma: float = 0.9  # lr(epoch) = max(lr * gamma**epoch, min_lr), as in get_lr() of the notebooks
    betas: tuple[float, float] = (0.9, 0.994)
    eps: float = 1e-8
    weight_decay: float = 1e-4
    label_smoothing: float = 0.01
    max_norm: float | None = 5.0  # gradient clipping
    patience: int = 5  # early stopping on validation accuracy (epochs without improvement)
    min_delta: float = 0.001
    seed: int = 1337
    device: str = "auto"
    output_dir: str = "checkpoints/run"
    overfit: int = 0  # >0: train AND validate on this many training samples (sanity check)
    max_steps_per_epoch: int | None = None  # cap batches per epoch (quick experiments)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        # Fill the embedding-dependent defaults exactly like the two notebooks did.
        onehot = self.model.embedding == "onehot"
        if self.data.min_freq is None:
            self.data.min_freq = 3 if onehot else 0  # GloVe already knows rare words
        if self.data.simplify is None:
            self.data.simplify = onehot
        if self.data.max_seq_len is None:
            self.data.max_seq_len = 20 if onehot else 40

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        train = dict(d["train"])
        train["betas"] = tuple(train["betas"])
        return cls(DataConfig(**d["data"]), ModelConfig(**d["model"]), TrainConfig(**train))
