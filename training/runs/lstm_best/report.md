# Training Report

- Dataset: `data/training`
- Models: `lstm`
- Target mode: `raw`
- Seq len: `6`
- Device: `cpu`

## Results

```text
model model_type split  train_loss     val_loss  physics_loss       rmse       mae          mse       r2   val_rmse   val_mae       val_mse   val_r2                            checkpoint_path                             history_path                             loss_curve_path                             physics_curve_path                                  predictions_path  trainable_parameters  total_parameters  status
 lstm    learned  test 17486.38815 16990.137094 566324.964857 181.704097 44.243256 33016.378906 0.958502 353.340509 59.348892 124849.515625 0.895165 training/runs/lstm_best/lstm/checkpoint.pt training/runs/lstm_best/lstm/history.csv training/runs/lstm_best/lstm/loss_curve.png training/runs/lstm_best/lstm/physics_curve.png training/runs/lstm_best/lstm/test_predictions.csv                 58848             58848 trained
```
