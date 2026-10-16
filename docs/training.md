# Training and using a model

## Install

```bash
python -m venv .venv && source .venv/bin/activate
# CPU-only PyTorch (small download); skip this line if you have a GPU build already
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'
```

## Data

Nothing to prepare. The first run downloads
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
(16,000 / 2,000 / 2,000 English sentences, 6 emotions; the same data the
notebooks read from Google Drive) into `~/.cache/textcls/`.

| class | train | share |
|---|---:|---:|
| joy | 5,362 | 33.5% |
| sadness | 4,666 | 29.2% |
| anger | 2,159 | 13.5% |
| fear | 1,937 | 12.1% |
| love | 1,304 | 8.2% |
| surprise | 572 | 3.6% |

Already have the Kaggle `train.txt / val.txt / test.txt` files (`text;label`
per line)? Use `--data-dir path/to/folder`.

GloVe runs also download `glove.6B.zip` (862 MB, once) and keep a 160 MB
`glove.6B.100d.pt` next to it. One-hot runs with `--simplify` (the default)
call NLTK, which takes a few minutes the first time; the processed text is
cached, so later runs start in seconds.

## Train one model

```bash
textcls-train --embedding onehot --cell lstm --impl torch            # notebook's OneHotPyLSTM
textcls-train --embedding glove  --cell gru  --impl scratch          # notebook's GloveGRU, fixed
textcls-train --cell rnn --impl scratch --unidirectional --n-layers 1  # the smallest model
```

Output in `checkpoints/<embedding>-<cell>-<impl>/` (or `--output-dir`):

| file | content |
|---|---|
| `best.pt` | weights + config + tokenizer of the epoch with the best validation accuracy |
| `metrics.json` | test accuracy, macro precision / recall / F1, per-class F1, confusion matrix, full history |
| `history.png` | loss, accuracy and log10 update ratio per epoch |
| `confusion.png` | test confusion matrix |

The log prints one line per epoch and the notebook-style test table at the end.

### Every flag

| flag | default | meaning |
|---|---|---|
| `--embedding` | onehot | `onehot` (learned table) or `glove` (frozen 6B vectors) |
| `--cell` | lstm | `rnn`, `lstm`, `gru` |
| `--impl` | torch | `torch` (`nn.LSTM`, ...) or `scratch` (our cells) |
| `--n-emb` | 100 | embedding size (GloVe: 50/100/200/300) |
| `--h-dim` | 128 | hidden size |
| `--n-layers` | 2 | stacked layers |
| `--unidirectional` | off | default is bidirectional |
| `--dropout` | 0.1 | between layers and before the head |
| `--act` | relu | vanilla RNN non-linearity (`relu` or `tanh`) |
| `--residual` | off | scratch cells: the notebooks' `h + 0.5·x` skip |
| `--simplify` / `--no-simplify` | on for onehot, off for glove | stop words + POS filter + lemmas |
| `--min-freq` | 3 onehot / 0 glove | drop words seen ≤ N times in train |
| `--max-seq-len` | 20 onehot / 40 glove | truncate longer texts |
| `--min-seq-len` | 3 | train only: skip shorter texts |
| `--vocab-size` | 3100 | one-hot vocabulary (incl. `<pad>`, `<oov>`) |
| `--batch-size` | 64 | |
| `--epochs` | 10 | maximum epochs (full passes over train) |
| `--lr`, `--gamma`, `--min-lr` | 2e-3, 0.9, 1e-5 | `lr(epoch) = max(lr·γ^epoch, min_lr)` |
| `--weight-decay` | 1e-4 | AdamW |
| `--label-smoothing` | 0.01 | |
| `--max-norm` | 5 | gradient clipping |
| `--patience` | 5 | early stopping on validation accuracy |
| `--overfit N` | 0 | sanity check: train and evaluate on N training sentences |
| `--max-steps-per-epoch` | none | cap batches per epoch for quick experiments |
| `--device` | auto | `cuda` if available, else `cpu` |
| `--data-dir`, `--cache-dir` | download, `~/.cache/textcls` | |

## Train all 12 and rebuild the results page

```bash
python scripts/run_all.py --epochs 8                     # writes docs/results.md + docs/images/
python scripts/run_all.py --only glove-lstm-scratch      # retrain one, table keeps the others
```

## Use a trained model

```bash
textcls-predict --checkpoint checkpoints/glove-gru-torch/best.pt \
    "i feel so alone since you left" "i cant believe they did that to me"
```

```python
from textcls.predict import Predictor

p = Predictor("checkpoints/glove-gru-torch/best.pt")
p.predict(["i am terrified of tomorrow"])        # ['fear']
p.predict_proba(["i am terrified of tomorrow"])  # tensor [1, 6], columns in CLASSES order
```

The predictor applies exactly the preprocessing the model was trained with
(it is stored in the checkpoint), so raw text goes in.

## Speed

Everything runs on a CPU. These are small models on short sentences; the
from-scratch versions are slower because their time loop runs in Python
(one small matmul per step instead of one fused cuDNN/ATen kernel). Measured
times are in [results.md](results.md).
