# Training Report

- Dataset: `data/training`
- Models: `lstm`
- Target mode: `seasonal_residual`
- Seq len: `6`
- Device: `cpu`

## Results

```text
model model_type split  train_loss  val_loss  physics_loss  physics_latent_loss  physics_ag_loss  physics_ndvi_loss  physics_weather_loss  physics_weather_identity_loss  physics_weather_threshold_loss  physics_weather_drought_loss  physics_weather_bounded_loss  physics_consistency_loss  physics_growth_loss  physics_phenology_loss  physics_water_loss       rmse       mae         mse       r2   val_rmse   val_mae       val_mse  val_r2                                                                   checkpoint_path                                                                    history_path                                                                    loss_curve_path                                                                    physics_curve_path                                                                         predictions_path  trainable_parameters  total_parameters  status
 lstm    learned  test    0.396053  0.937371       1.12755             0.263417         0.040546           0.083788               3.51843                      14.777678                        2.211933                      0.023862                           0.0                  0.232539             0.008803                0.010026            0.012049 417.579708 67.330605 174372.8125 0.780831 331.400461 59.900627 109826.265625 0.90778 /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/lstm_best/lstm/checkpoint.pt /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/lstm_best/lstm/history.csv /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/lstm_best/lstm/loss_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/lstm_best/lstm/physics_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/lstm_best/lstm/test_predictions.csv                 58848             58848 trained
```
