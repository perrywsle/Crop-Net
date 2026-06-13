# Training Report

- Dataset: `data/training`
- Models: `lstm`
- Target mode: `raw`
- Seq len: `6`
- Device: `cuda`

## Results

```text
model model_type split  train_loss  val_loss  physics_loss     rmse     mae      mse       r2  val_rmse  val_mae  val_mse   val_r2                            checkpoint_path                             history_path                             loss_curve_path                             physics_curve_path                                  predictions_path  trainable_parameters  total_parameters  status
 lstm    learned  test    0.578714    0.6169     15.555449 2.601377 0.43658 6.767161 0.912412  2.573917 0.422119  6.62505 0.913838 training/runs/lstm_best/lstm/checkpoint.pt training/runs/lstm_best/lstm/history.csv training/runs/lstm_best/lstm/loss_curve.png training/runs/lstm_best/lstm/physics_curve.png training/runs/lstm_best/lstm/test_predictions.csv                 58848             58848 trained
```
