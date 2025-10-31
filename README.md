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

## Interactive exam page prototype

The `experiments/exam_site` directory contains a static web prototype that
emulates the布局和交互 of the State Grid online exam platform. It supports CSV
题库导入 with 单选题、 多选题 and 判断题 question types. To try it locally:

```bash
cd experiments/exam_site
python -m http.server 8000
```

Then open <http://localhost:8000> in your browser and上传符合以下字段的 CSV：

- `题干`（必填）
- `题型`（必填，支持“单选题”“多选题”“判断题”及其简称）
- `选项A` 至 `选项H`（可选，用于展示题目选项）
- `正确答案`（必填）

一个包含示例题目的 `sample_questions.csv` 文件也提供在同一目录。
