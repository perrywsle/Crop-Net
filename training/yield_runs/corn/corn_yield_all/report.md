# Yield Training Report

- Dataset: `data/yield_training/corn`
- Feature group: `all`
- Target: `yield_bu_acre`
- Best trainable model: `ExtraTrees`

## Results

```text
                         model model_type  val_rmse    val_r2  test_rmse   test_r2
             BaselineTrainMean   baseline 20.129665 -0.435084  21.673225 -0.023002
                    ExtraTrees         ml 22.979287 -0.870154  19.748629  0.150617
                  RandomForest         ml 23.954138 -1.032195  19.591471  0.164082
                         Ridge         ml 26.189632 -1.429199  20.136402  0.116934
BaselinePreviousYearSameCounty   baseline 32.919169 -2.837973  18.354003  0.266346
```
