# Training Report

- Dataset: `data/training`
- Models: `tiny_mamba_ssm`
- Target mode: `raw`
- Seq len: `6`
- Device: `cpu`

## Results

```text
         model model_type split  train_loss    val_loss  physics_loss       rmse       mae          mse       r2   val_rmse   val_mae      val_mse   val_r2                                       checkpoint_path                                        history_path                                        loss_curve_path                                        physics_curve_path                                             predictions_path  trainable_parameters  total_parameters  status
tiny_mamba_ssm    learned  test 5828.865245 5663.663438 566324.964857 205.032562 49.582256 42038.351562 0.947162 365.119715 61.089142 133312.40625 0.888059 training/runs/mamba_best/tiny_mamba_ssm/checkpoint.pt training/runs/mamba_best/tiny_mamba_ssm/history.csv training/runs/mamba_best/tiny_mamba_ssm/loss_curve.png training/runs/mamba_best/tiny_mamba_ssm/physics_curve.png training/runs/mamba_best/tiny_mamba_ssm/test_predictions.csv                 94208             94208 trained
```
