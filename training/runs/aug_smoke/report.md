# Training Report

- Dataset: `data/training`
- Models: `lstm`
- Target mode: `seasonal_residual`
- Seq len: `6`
- Device: `cuda`

## Results

```text
model model_type split     rmse      mae       mse       r2  val_rmse  val_mae   val_mse   val_r2                                                                   checkpoint_path                                                                    history_path                                                                    loss_curve_path                                                                         predictions_path  trainable_parameters  total_parameters  status
 lstm    learned  test 3.301342 0.498349 10.898858 0.858935  4.801999 0.460804 23.059198 0.700104 /mnt/c/users/lobakkang/github/crop-net/training/runs/aug_smoke/lstm/checkpoint.pt /mnt/c/users/lobakkang/github/crop-net/training/runs/aug_smoke/lstm/history.csv /mnt/c/users/lobakkang/github/crop-net/training/runs/aug_smoke/lstm/loss_curve.png /mnt/c/users/lobakkang/github/crop-net/training/runs/aug_smoke/lstm/test_predictions.csv                 28259             28259 trained
```
