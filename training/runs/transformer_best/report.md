# Training Report

- Dataset: `data/training`
- Models: `transformer_encoder`
- Target mode: `raw`
- Seq len: `6`
- Device: `cuda`

## Results

```text
              model model_type split  train_loss  val_loss  physics_loss    rmse      mae      mse       r2  val_rmse  val_mae  val_mse   val_r2                                                  checkpoint_path                                                   history_path                                                   loss_curve_path                                                   physics_curve_path                                                        predictions_path  trainable_parameters  total_parameters  status
transformer_encoder    learned  test    0.580107  0.615129     15.546748 2.69096 0.437755 7.241264 0.906275  2.806966 0.455414 7.879056 0.897529 training/runs/transformer_best/transformer_encoder/checkpoint.pt training/runs/transformer_best/transformer_encoder/history.csv training/runs/transformer_best/transformer_encoder/loss_curve.png training/runs/transformer_best/transformer_encoder/physics_curve.png training/runs/transformer_best/transformer_encoder/test_predictions.csv                172224            172224 trained
```
