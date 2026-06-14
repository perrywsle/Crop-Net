# Training Report

- Dataset: `data/training`
- Models: `gru`
- Target mode: `seasonal_residual`
- Seq len: `6`
- Device: `cpu`

## Results

```text
model model_type split  train_loss  val_loss  physics_loss  physics_latent_loss  physics_ag_loss  physics_ndvi_loss  physics_weather_loss  physics_weather_identity_loss  physics_weather_threshold_loss  physics_weather_drought_loss  physics_weather_bounded_loss  physics_consistency_loss  physics_growth_loss  physics_phenology_loss  physics_water_loss       rmse       mae           mse       r2   val_rmse   val_mae     val_mse   val_r2                                                                 checkpoint_path                                                                  history_path                                                                  loss_curve_path                                                                  physics_curve_path                                                                       predictions_path  trainable_parameters  total_parameters  status
  gru    learned  test    0.408288  0.842981       1.12236             0.218743         0.038905           0.083214               3.51843                      14.777678                        2.211933                      0.023862                           0.0                  0.191645             0.007607                0.007411            0.012079 431.980559 72.408493 186607.203125 0.765454 339.042494 61.316006 114949.8125 0.903478 /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/gru_best/gru/checkpoint.pt /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/gru_best/gru/history.csv /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/gru_best/gru/loss_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/gru_best/gru/physics_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/gru_best/gru/test_predictions.csv                 46080             46080 trained
```
