# Training Report

- Dataset: `data/training`
- Models: `gru`
- Target mode: `raw`
- Seq len: `6`
- Device: `cpu`

## Results

```text
model model_type split   train_loss     val_loss  physics_loss       rmse       mae        mse       r2   val_rmse  val_mae       val_mse   val_r2                          checkpoint_path                           history_path                           loss_curve_path                           physics_curve_path                                predictions_path  trainable_parameters  total_parameters  status
  gru    learned  test 17486.399333 16990.145583 566324.964857 196.848133 48.823872 38749.1875 0.951296 354.768972 60.07428 125861.023438 0.894316 training/runs/gru_best/gru/checkpoint.pt training/runs/gru_best/gru/history.csv training/runs/gru_best/gru/loss_curve.png training/runs/gru_best/gru/physics_curve.png training/runs/gru_best/gru/test_predictions.csv                 46080             46080 trained
```
