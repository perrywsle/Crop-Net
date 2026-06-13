# Yield Training Report

- Dataset: `data/yield_training`
- Feature group: `all`
- Target: `yield_bu_acre`
- Best trainable model: `Ridge`

## Results

```text
                         model model_type  val_rmse    val_r2  test_rmse   test_r2
             BaselineTrainMean   baseline 20.129665 -0.435084  21.673225 -0.023002
                         Ridge         ml 20.200230 -0.445163  20.435887  0.090471
                    ExtraTrees         ml 20.865334 -0.541895  19.613574  0.162195
                  RandomForest         ml 21.463285 -0.631536  19.534155  0.168966
BaselinePreviousYearSameCounty   baseline 32.919169 -2.837973  18.354003  0.266346
```
