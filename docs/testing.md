# Testing guide

Three levels, from seconds to minutes:

1. **Unit tests** (`pytest`): every module, CPU only, no network, fake data.
2. **Sanity checks on the real data** (`scripts/sanity_check.py`): is the network trainable at all?
3. **Short real training runs** (`scripts/run_all.py`): the numbers in [results.md](results.md).

## 1. Unit tests

```bash
pytest                         # all of them
pytest tests/test_model.py -v  # one file
pytest -k padding              # by name
ruff check src tests scripts   # lint
```

On a busy machine, `OMP_NUM_THREADS=1 pytest` is much faster.

| file | what it proves |
|---|---|
| `test_data.py` | cleaning, contractions (whole words only), the `text;label` loader, rare-word filtering uses **train** counts and never drops a sentence, label order is the notebooks', the dataset truncates and only filters when asked, the collator right-pads and returns true lengths, config defaults match each notebook |
| `test_tokenizer.py` | `<pad>`=0 / `<oov>`=1, frequency order, the Keras-style `num_words` cap, encode/decode, save/load, GloVe keeps only the needed rows with `<pad>`=0 and `<oov>`=mean vector |
| `test_model.py` | see below |
| `test_train.py` | macro precision / recall / F1 checked by hand, the LR schedule, the NN sanity checks below, a full `run()` writes `best.pt`, `metrics.json` and both figures, the checkpoint reloads into the same config and `Predictor` gives the right emotion on easy sentences, `--overfit` uses exactly N samples |

### What the model tests prove

| test | why it matters |
|---|---|
| `test_output_shape_all_12_models` | every combination builds and runs |
| `test_padding_does_not_change_the_prediction` (×12) | appending 9 extra `<pad>` tokens leaves the logits **identical**: the notebooks' main bug cannot come back |
| `test_each_sample_is_independent_of_the_batch` | a sentence scored alone = the same sentence inside a padded batch |
| `test_scratch_cell_matches_pytorch_cell` (×4) | our RNN (tanh and relu), LSTM and GRU cells give PyTorch's numbers when given PyTorch's weights |
| `test_scratch_network_matches_torch_network` (×6) | the whole 2-layer, (bi)directional scratch encoder with variable lengths equals `nn.RNN` / `nn.LSTM` / `nn.GRU` + packing |
| `test_reverse_padded` | `[a b c PAD] → [c b a PAD]` for the backward direction |
| `test_onehot_embedding_equals_the_notebooks_one_hot_matmul` | the fast lookup = `Linear(F.one_hot(x))` |
| `test_glove_embedding_is_frozen` | GloVe gets no gradient, everything else does |
| `test_every_trainable_parameter_gets_a_gradient` (×12) | no dead / disconnected weights |

## 2. Neural-network sanity checks

From Karpathy's *A Recipe for Training Neural Networks*. They run in the unit
tests on fake data (all 6 cell × impl pairs) and on the real data with:

```bash
python scripts/sanity_check.py --embedding onehot --cell lstm --impl scratch
```

| check | expected | why |
|---|---|---|
| loss at init | ≈ ln 6 = 1.792 | an untrained model should spread probability evenly; much higher = confidently wrong (bad init) |
| gradients | every trainable tensor gets a non-zero gradient | catches disconnected layers and frozen-by-mistake weights |
| overfit one batch | 64 real sentences → loss < 0.05, 100% accuracy | if it cannot memorise 64 sentences, the pipeline is broken (labels shuffled, masks wrong, LR far off) |
| real vs blind | after 2 epochs, real-text model ≫ model whose input is erased | the blind model can only learn class frequencies (≈ 35% = always "joy"); the gap shows the model reads the text |

Results on the real data are in [results.md](results.md#sanity-checks).

Two more signals are logged during every training run (see each `history.png`):

* **update / weight ratio**, `log10(‖Δw‖ / ‖w‖)` of the real AdamW step,
  averaged over the weight matrices (biases start at zero, so the ratio is meaningless for them). Around −3 is healthy; much lower means the
  learning rate is too small to move the weights, much higher means it is
  thrashing them.
* **train vs validation curves**: the gap shows over-fitting, and early stopping
  keeps the best validation epoch.
