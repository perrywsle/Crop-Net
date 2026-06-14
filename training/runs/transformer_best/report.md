# Training Report

- Dataset: `data/training`
- Models: `transformer_encoder`
- Target mode: `seasonal_residual`
- Seq len: `6`
- Device: `cpu`

## Results

```text
              model model_type split  train_loss  val_loss  physics_loss  physics_latent_loss  physics_ag_loss  physics_ndvi_loss  physics_weather_loss  physics_weather_identity_loss  physics_weather_threshold_loss  physics_weather_drought_loss  physics_weather_bounded_loss  physics_consistency_loss  physics_growth_loss  physics_phenology_loss  physics_water_loss       rmse       mae           mse       r2  val_rmse   val_mae       val_mse   val_r2                                                                                         checkpoint_path                                                                                          history_path                                                                                          loss_curve_path                                                                                          physics_curve_path                                                                                               predictions_path  trainable_parameters  total_parameters  status
transformer_encoder    learned  test    0.375355   0.79914      1.102776             0.051085         0.034259           0.079652               3.51843                      14.777678                        2.211933                      0.023862                           0.0                  0.038137             0.003516                0.007016            0.002416 447.872523 71.902649 200589.796875 0.747879 348.57193 61.664425 121502.390625 0.897976 /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/transformer_best/transformer_encoder/checkpoint.pt /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/transformer_best/transformer_encoder/history.csv /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/transformer_best/transformer_encoder/loss_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/transformer_best/transformer_encoder/physics_curve.png /mnt/c/Users/lobakkang/GITHUB/Crop-Net/training/runs/transformer_best/transformer_encoder/test_predictions.csv                172224            172224 trained
```
