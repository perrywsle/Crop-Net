# Training Report

- Dataset: `data/training`
- Models: `transformer_encoder`
- Target mode: `raw`
- Seq len: `6`
- Device: `cpu`

## Results

```text
              model model_type split  train_loss     val_loss  physics_loss      rmse       mae          mse       r2   val_rmse   val_mae       val_mse   val_r2                                                  checkpoint_path                                                   history_path                                                   loss_curve_path                                                   physics_curve_path                                                        predictions_path  trainable_parameters  total_parameters  status
transformer_encoder    learned  test 17486.38893 16990.166187 566324.964857 220.92393 52.203823 48807.382812 0.938654 353.244548 63.476479 124781.710938 0.895222 training/runs/transformer_best/transformer_encoder/checkpoint.pt training/runs/transformer_best/transformer_encoder/history.csv training/runs/transformer_best/transformer_encoder/loss_curve.png training/runs/transformer_best/transformer_encoder/physics_curve.png training/runs/transformer_best/transformer_encoder/test_predictions.csv                172224            172224 trained
```
