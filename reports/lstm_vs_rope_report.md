# LSTM vs Causal RoPE MHA on Airline Passengers

## Dataset
- Source: Airline Passengers (monthly totals 1949-1960)
- Forecast horizon: 1 month ahead
- Input window: 12 months

## LSTM
- Test MSE: 9238.036
- Test MAE: 83.090
- Final training loss: 0.0319
- Final validation loss: 0.3209
- Train time (s): 14.70
- Eval time (s): 0.01
- Trainable parameters: 50,497
- Generalization gap (val - train): 0.2889

## GRU
- Test MSE: 6221.339
- Test MAE: 68.233
- Final training loss: 0.0224
- Final validation loss: 0.2037
- Train time (s): 16.07
- Eval time (s): 0.00
- Trainable parameters: 37,889
- Generalization gap (val - train): 0.1813

## ROPE
- Test MSE: 1841.215
- Test MAE: 27.664
- Final training loss: 0.0244
- Final validation loss: 0.1291
- Train time (s): 10.09
- Eval time (s): 0.00
- Trainable parameters: 33,665
- Generalization gap (val - train): 0.1047
- Attention mass last 3 months: 0.000
- Attention entropy (last token): 0.512
- Attention center of mass (0=oldest, 11=latest): 0.20

## SINUSOIDAL_MHA
- Test MSE: 1328.629
- Test MAE: 30.704
- Final training loss: 0.0137
- Final validation loss: 0.0965
- Train time (s): 8.93
- Eval time (s): 0.04
- Trainable parameters: 33,665
- Generalization gap (val - train): 0.0828
- Attention mass last 3 months: 0.103
- Attention entropy (last token): 2.206
- Attention center of mass (0=oldest, 11=latest): 3.15

## LEARNED_MHA
- Test MSE: 1049.467
- Test MAE: 25.584
- Final training loss: 0.0088
- Final validation loss: 0.0992
- Train time (s): 8.65
- Eval time (s): 0.00
- Trainable parameters: 33,917
- Generalization gap (val - train): 0.0904
- Attention mass last 3 months: 0.212
- Attention entropy (last token): 2.411
- Attention center of mass (0=oldest, 11=latest): 4.66

## Comparison
- Test MSE ranking:
  1. learned_mha — MSE 1049.467, MAE 25.584, params 33,917, train 8.65s
  2. sinusoidal_mha — MSE 1328.629, MAE 30.704, params 33,665, train 8.93s
  3. rope — MSE 1841.215, MAE 27.664, params 33,665, train 10.09s
  4. gru — MSE 6221.339, MAE 68.233, params 37,889, train 16.07s
  5. lstm — MSE 9238.036, MAE 83.090, params 50,497, train 14.70s

## RoPE vs Learned Positional Encoding (Matched Parameter Budget)
- Parameter counts: RoPE 33,665 (FF dim 128) vs Learned 33,917 (FF dim 124)
- Generalization gap: RoPE 0.1047 vs Learned 0.0904
- Recency attention (last 3 months mass): RoPE 0.000 vs Learned 0.212
- Attention entropy: RoPE 0.512 vs Learned 2.411
- Attention center of mass (0=oldest, 11=latest): RoPE 0.20 vs Learned 4.66
- Interpretation: Under an equalized parameter budget the learned positional attention develops a stronger recency focus (higher center of mass and entropy) that helps it track the accelerating passenger counts, delivering lower error and a smaller generalization gap. The RoPE model keeps more probability mass anchored to early timesteps due to its fixed relative phase bias, making it slower to adapt to local trend shifts and leaving noticeable accuracy on the table.

## RoPE vs Sinusoidal Positional Encoding
- Parameter counts: RoPE 33,665 (FF dim 128) vs Sinusoidal 33,665 (FF dim 128)
- Generalization gap: RoPE 0.1047 vs Sinusoidal 0.0828
- Recency attention (last 3 months mass): RoPE 0.000 vs Sinusoidal 0.103
- Attention entropy: RoPE 0.512 vs Sinusoidal 2.206
- Attention center of mass (0=oldest, 11=latest): RoPE 0.20 vs Sinusoidal 3.15
- Interpretation: The fixed sinusoidal encoding from Attention Is All You Need introduces an absolute phase bias that still allows more flexibility than RoPE's strictly relative rotations. On this dataset it shifts additional probability toward recent months without incurring extra parameters, closing part of the accuracy gap while remaining cheaper than the learned alternative.

## Visualizations
- reports/figures/test_mse.png
- reports/figures/test_mae.png
- reports/figures/parameters.png
- reports/figures/train_time.png
- reports/figures/eval_time.png
- reports/figures/generalization_gap.png
- reports/figures/loss_curves.png
- reports/figures/attention_profiles.png
