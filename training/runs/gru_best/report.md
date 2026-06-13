# Training Report

- Dataset: `data/training`
- Models: `gru`
- Target mode: `raw`
- Seq len: `6`
- Device: `cuda`

## Results

```text
model model_type split  train_loss  val_loss  physics_loss     rmse      mae      mse       r2  val_rmse  val_mae  val_mse   val_r2                          checkpoint_path                           history_path                           loss_curve_path                           physics_curve_path                                predictions_path  trainable_parameters  total_parameters  status
  gru    learned  test    0.587706  0.615459     15.555786 2.630983 0.432019 6.922072 0.910407  2.605279 0.433677 6.787478 0.911726 training/runs/gru_best/gru/checkpoint.pt training/runs/gru_best/gru/history.csv training/runs/gru_best/gru/loss_curve.png training/runs/gru_best/gru/physics_curve.png training/runs/gru_best/gru/test_predictions.csv                 46080             46080 trained
```
