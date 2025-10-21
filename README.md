# Stock price forecasting: recurrent baselines vs causal attention

This repository contains a reproducible experiment that contrasts classic
recurrent baselines (a tanh RNN, a residual/skip-connected RNN inspired by the
"LSTM as a rotated ResNet" intuition, LSTM, and GRU) with causal multi-head
self-attention models that use either rotary positional embeddings (RoPE), the
fixed sinusoidal encoding introduced in *Attention Is All You Need*, or learned
positional embeddings on a four-year daily stock OHLCV dataset
(`data/stock_data.csv`). The pipeline engineers momentum features such as
returns, log-returns, moving-average gaps, and volume ratios so the models ingest
rich numeric structure without categorical inputs.
The comparison captures accuracy, parameter efficiency, wall-clock runtime, and
a matched-parameter analysis that measures how inductive bias differences impact
attention allocation and generalization. Every architecture is tuned to keep the
trainable parameter count near 76,800 so the results emphasize inductive biases
instead of raw capacity.

## Running the experiment

```bash
pip install -r requirements.txt  # or install numpy, pandas, torch manually
# Optional: CPU-only PyTorch wheel -> pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python experiments/compare_lstm_vs_rope.py
```

The script will load the cached dataset, train all configured models, evaluate
them on a hold-out test split, compute their parameter counts, runtime, and
generalization gaps, inspect attention distributions for the transformer
variants, and write a Markdown report to `reports/lstm_vs_rope_report.md` that
includes: (1) a study of how residual skips change the behavior of a naive RNN
relative to LSTM/GRU gates, (2) a RoPE-vs-learned positional embedding analysis
under a matched parameter budget, and (3) a RoPE-vs-sinusoidal comparison of
the original Transformer encoding. In addition, it will export a single
composite visualization at `reports/figures/model_comparison_summary.png` that
aggregates metric, runtime, parameter, loss, and attention diagnostics; this
directory is ignored by git because the figures are generated artifacts.

The stock history is cached at `data/stock_data.csv`, so repeated runs do not
redownload anything. You can swap in another multivariate financial time series
by replacing this file and adjusting the feature engineering block in
`experiments/compare_lstm_vs_rope.py` if different covariates are required.
