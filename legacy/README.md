# Legacy notebooks (reference only)

The original Colab notebooks, kept as they were:

| Notebook | What it does |
|---|---|
| `nlp_one_hot.ipynb` | one-hot word vectors (+ stop words, POS filter, lemmas), RNN / LSTM / GRU with `nn.*` and from scratch |
| `nlp_glove_embedding.ipynb` | frozen GloVe 6B 100-d vectors, the same six models |

They need Google Drive, TensorFlow (for the Keras tokenizer), `torchtext`
(deprecated) and `torchmetrics`. Everything now lives in `src/textcls/`.

## What changed, and why

The ideas, names and hyperparameters are kept; the package is the same
experiment with these bugs fixed. Each fix has a test that would fail on the
old code.

| # | In the notebooks | Problem | Now | Test |
|---|---|---|---|---|
| 1 | `RNNBlock` / `LSTMBlock` / `GRUBlock` build `n_unit = max_seq_len` cells and use cell *i* at step *i* | Every position has its own weights: a 20-layer feed-forward network, not a recurrent one. A word learned at position 3 is unknown at position 7. This is why the scratch models trailed the `nn.*` ones. | One cell per layer and direction, reused at every step (`ScratchRNN`) | `test_scratch_network_matches_torch_network` |
| 2 | Batches right-padded to `max_len`, then `h_n` of the RNN is read | `h_n` is the state **after** the padding (up to 17 `<pad>` steps), and the backward direction starts on padding | `pack_padded_sequence` for `nn.*`; masked updates for scratch cells | `test_padding_does_not_change_the_prediction` (all 12 models) |
| 3 | GRU: `g = tanh(r * hgt + (1 - r) * h)` with `hgt = W_in x + W_hn h` | Reset gate also scales the input; old state leaks into the candidate | `n = tanh(W_in x + r * (W_hn h))`, PyTorch's formula | `test_scratch_cell_matches_pytorch_cell` |
| 4 | GloVe `RNN`: `h = act(...) + res(x)*0.5 if self.res_proj else h + x*0.5` | Python reads it as `A if cond else (h + x*0.5)`. With `in_dim == h_dim` (100/100) the recurrent update was **skipped entirely**: the "RNN" just summed its inputs. Explains GloveRNN's 55%. | Residual is an explicit option (`--residual`), off by default | `test_residual_option_still_runs_and_ignores_padding` |
| 5 | `OneHotPyGRU._init_weights` tests `isinstance(module, nn.LSTM)` | Copy-paste: the GRU never got its init | Same init applied to every cell | — |
| 6 | Early stop `if patience_counter <= 0 or vl_acc > 0.78` | Stops as soon as a hand-picked target is crossed (printed as "lack of improvement") | Patience only | — |
| 7 | `validator(..., n_step)` | Validation on 10–20 random batches, so the best checkpoint was picked on noise | Whole validation set every epoch | — |
| 8 | Test with the last weights | The saved best checkpoint was never reloaded | Test the best checkpoint | `test_full_run_writes_artifacts_and_predicts` |
| 9 | `TextDataset(..., min_len=3)` on val/test; `remove_less_freq_words` on test with test counts | Test set silently shrank, and filtering used test statistics | Eval sets keep every sentence; word counts come from train only | `test_rare_words_use_train_counts_and_never_drop_texts` |
| 10 | Update ratio `‖-lr · grad‖ / ‖w‖` using a global `lr` | Adam does not apply `lr · grad`; and `lr` was whatever was defined last | Measured on the real step: `‖w_after − w_before‖ / ‖w_before‖` | `test_train_one_epoch_lowers_loss_and_reports_update_ratio` |
| 11 | `"were": "we are"` in the contractions | "they were happy" → "they we are happy" | Removed | `test_expand_contractions_whole_words_only` |
| 12 | `nn.Linear(V, E)(F.one_hot(x))` | Correct, but builds a `[B, T, 3100]` tensor of zeros | Row lookup in the same weight matrix | `test_onehot_embedding_equals_the_notebooks_one_hot_matmul` |
| 13 | "Epoch" = 10–20 batches | Numbers of epochs were not comparable with anything | One epoch = one full pass over the training set | — |
