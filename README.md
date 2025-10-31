# LSTM vs Causal RoPE MHA comparison

This repository contains a reproducible experiment that contrasts classic
recurrent baselines (LSTM and GRU) with causal multi-head self-attention models
that use either rotary positional embeddings (RoPE), the fixed sinusoidal
encoding introduced in *Attention Is All You Need*, or learned positional
embeddings on the monthly Airline Passengers time series dataset. The
comparison captures accuracy, parameter efficiency, wall-clock runtime, and a
matched-parameter analysis that measures how inductive bias differences impact
attention allocation and generalization.

## Running the experiment

```bash
pip install -r requirements.txt  # or install numpy, pandas, torch manually
# Optional: CPU-only PyTorch wheel -> pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python experiments/compare_lstm_vs_rope.py
```

The script will download the dataset, train all configured models, evaluate
them on a hold-out test split, compute their parameter counts, runtime, and
generalization gaps, inspect attention distributions for the transformer
variants, and write a Markdown report to `reports/lstm_vs_rope_report.md` that
includes a RoPE-vs-learned positional embedding analysis under a matched
parameter budget alongside a RoPE-vs-sinusoidal comparison of the original
Transformer encoding. In addition, it will export a suite of diagnostic
visualizations (loss curves, metric/parameter/time bar charts, and attention
profiles) to `reports/figures/` for quick visual inspection; this directory is
ignored by git because the figures are generated artifacts.

## Comparing ERNIE LoRA vs full fine-tuning

To reproduce a lightweight comparison between PaddleNLP's LoRA adapter and
full-parameter fine-tuning on the Chinese TNEWS classification benchmark using
`ernie-3.0-medium-zh`, install the additional Paddle dependencies and run:

```bash
pip install -r requirements.txt  # installs PaddlePaddle and PaddleNLP
python experiments/compare_ernie_lora_vs_full.py \
  --max-train-samples 1024 \
  --max-eval-samples 512 \
  --epochs 2 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1
```

The script freezes or unfreezes the transformer backbone depending on the
strategy, tracks trainable parameter counts, accuracy, and runtime for both
LoRA and full fine-tuning, and writes a Markdown summary to
`reports/ernie_lora_vs_full_report.md`. Adapter checkpoints are exported under
`reports/lora_artifacts/` and full fine-tuned weights under
`reports/full_finetune_artifacts/`. Increasing the LoRA rank to 16, scaling to
32, and applying a modest 0.1 dropout widens the adapter capacity enough to
track the full fine-tuning baseline more closely while still training a tiny
fraction of the original model parameters. You can further adjust
`--lora-r`, `--lora-alpha`, or `--lora-dropout` at the command line if a
different capacity/regularization trade-off is desired.
