# Training Report

- Dataset: `data/training`
- Models: `tiny_mamba_ssm`
- Target mode: `seasonal_residual`
- Seq len: `6`
- Device: `cpu`

## Results

```text
         model model_type split  train_loss  val_loss  physics_loss  physics_latent_loss  physics_ag_loss  physics_ndvi_loss  physics_weather_loss  physics_weather_identity_loss  physics_weather_threshold_loss  physics_weather_drought_loss  physics_weather_bounded_loss  physics_consistency_loss  physics_growth_loss  physics_phenology_loss  physics_water_loss      rmse       mae          mse       r2   val_rmse   val_mae       val_mse   val_r2                                                                              checkpoint_path                                                                               history_path                                                                               loss_curve_path                                                                               physics_curve_path                                                                                    predictions_path  trainable_parameters  total_parameters  status
tiny_mamba_ssm    learned  test    0.390994  0.873623      1.100411             0.031748         0.032737           0.079715               3.51843                      14.777678                        2.211933                      0.023862                           0.0                   0.02558             0.002036                0.002471            0.001661 459.03775 75.714165 210715.65625 0.735152 347.297445 65.467674 120615.515625 0.898721 /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/mamba_best/tiny_mamba_ssm/checkpoint.pt /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/mamba_best/tiny_mamba_ssm/history.csv /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/mamba_best/tiny_mamba_ssm/loss_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/mamba_best/tiny_mamba_ssm/physics_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/mamba_best/tiny_mamba_ssm/test_predictions.csv                 58432             58432 trained
```
