# Training Report

- Dataset: `data/training`
- Models: `tiny_mamba_ssm`
- Target mode: `raw`
- Seq len: `6`
- Device: `cuda`

## Results

```text
         model model_type split  train_loss  val_loss  physics_loss     rmse      mae      mse       r2  val_rmse  val_mae  val_mse   val_r2                                       checkpoint_path                                        history_path                                        loss_curve_path                                        physics_curve_path                                             predictions_path  trainable_parameters  total_parameters  status
tiny_mamba_ssm    learned  test    0.244758  0.315905     15.548885 2.577903 0.413866 6.645586 0.913985  2.509485 0.394849 6.297513 0.918098 training/runs/mamba_best/tiny_mamba_ssm/checkpoint.pt training/runs/mamba_best/tiny_mamba_ssm/history.csv training/runs/mamba_best/tiny_mamba_ssm/loss_curve.png training/runs/mamba_best/tiny_mamba_ssm/physics_curve.png training/runs/mamba_best/tiny_mamba_ssm/test_predictions.csv                 94208             94208 trained
```
