# nlp-text-classification

**Read a short English sentence, tell which of six emotions it carries.**
RNN, LSTM and GRU in PyTorch, each built twice (with `nn.LSTM` & co. and from
scratch), fed either learned one-hot embeddings or frozen GloVe vectors:
12 models, one codebase, one shared training recipe.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org)
[![Dataset: dair-ai/emotion](https://img.shields.io/badge/dataset-dair--ai%2Femotion-orange.svg)](https://huggingface.co/datasets/dair-ai/emotion)

```
sadness | surprise | anger | love | joy | fear
```

```mermaid
flowchart LR
    T["sentence"] --> P["clean, (simplify)<br/>tokenize"]
    P --> E["embedding<br/>one-hot table or GloVe"]
    E --> R["RNN / LSTM / GRU<br/>nn.* or from scratch<br/>2 layers, bidirectional"]
    R --> H["MLP head"]
    H --> C["6 emotions"]
```

## Results

Training of the 12 models is running; the table and figures land in
[docs/results.md](docs/results.md) when it finishes.

## Quick start

```bash
# 1. install (Python >= 3.10)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e '.[dev]'

# 2. unit tests: CPU, offline, fake data (~40 s)
pytest

# 3. sanity checks on the real data (downloads the dataset)
python scripts/sanity_check.py --embedding onehot --cell lstm --impl scratch

# 4. train one model (~0.5-2 min per epoch on one CPU core)
textcls-train --embedding glove --cell gru --impl torch --epochs 8

# 5. use it
textcls-predict --checkpoint checkpoints/glove-gru-torch/best.pt "i feel so alone tonight"

# 6. (optional) train all 12 and regenerate docs/results.md
python scripts/run_all.py --epochs 8
```

On a machine that is busy with other jobs, add `--threads 1` (or
`OMP_NUM_THREADS=1`): small RNNs get *much slower* when PyTorch fights for cores.

## Documentation

| guide | what is inside |
|---|---|
| [docs/results.md](docs/results.md) | all 12 models: test accuracy, macro F1, curves, confusion matrices, sanity checks |
| [docs/training.md](docs/training.md) | install, data, every training flag, prediction from the CLI and from Python |
| [docs/architecture.md](docs/architecture.md) | preprocessing, embeddings, the cell equations, shapes, parameter counts |
| [docs/testing.md](docs/testing.md) | the unit tests and what each one proves; the neural-network sanity checks |
| [legacy/README.md](legacy/README.md) | the original notebooks and **the 13 things that were fixed** |

## Repository layout

```
src/textcls/          the package (read in this order)
  config.py           every hyperparameter: DataConfig, ModelConfig, TrainConfig
  data.py             download, cleaning, stop words / lemmas, dataset, padding collator
  tokenizer.py        WordTokenizer (one-hot vocabulary) and GloVe loading
  model.py            embeddings, RNN/LSTM/GRU cells from scratch, ScratchRNN, TextClassifier
  metrics.py          accuracy, macro precision / recall / F1, confusion matrix
  train.py            training loop, early stopping, checkpoints, `textcls-train`
  predict.py          Predictor class and `textcls-predict`
  plots.py            training curves, confusion matrix, comparison chart
scripts/
  run_all.py          train the 12 models, write docs/results.md and figures
  sanity_check.py     init loss, overfit a batch, real-vs-blind, gradient flow (real data)
tests/                pytest suite (CPU only, offline)
docs/                 guides, results and images
legacy/               the two original Colab notebooks + what changed
```

## Dataset

[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
(Saravia et al., 2018): English Twitter messages labelled with six emotions,
16,000 train / 2,000 validation / 2,000 test. Downloaded automatically on first
run. It is imbalanced (joy 33.5%, surprise 3.6%), which is why macro F1 is
reported next to accuracy.

## What changed from the notebooks

The notebooks' ideas, names and settings are kept; the package fixes what
was wrong. Highlights (full list with tests in [legacy/README.md](legacy/README.md)):

* **The from-scratch "RNNs" were not recurrent.** They built one cell per
  position (20 different weight sets), so a word learned at position 3 meant
  nothing at position 7. Now one cell is shared across time, and the tests
  prove our cells give *exactly* the numbers of `nn.RNN` / `nn.LSTM` / `nn.GRU`.
* **Padding leaked into the prediction.** The classifier read the hidden state
  after up to 17 `<pad>` tokens. Now sequences are packed (or masked), and a
  test checks that extra padding changes nothing, for all 12 models.
* **The GRU equation was wrong**, and an operator-precedence slip in the GloVe
  RNN skipped the recurrent update entirely.
* **Evaluation was optimistic and noisy**: training stopped when validation
  accuracy crossed a hand-picked target, validation used 10–20 random batches,
  short test sentences were removed, and the last (not best) weights were tested.

## Status and next steps

- [x] Package, CLIs, unit tests, neural-network sanity checks
- [x] 12 models trained on CPU with one shared recipe
- [ ] Class-weighted loss or oversampling for `surprise` / `love` (lowest F1)
- [ ] Attention pooling over all hidden states instead of the last state
- [ ] A small Transformer encoder as the 13th model

## License

[MIT](LICENSE)
