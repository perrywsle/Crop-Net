# Yield Training Report

- Dataset: `data/yield_training/soybeans`
- Feature group: `all`
- Target: `yield_bu_acre`
- Best trainable model: `RandomForest`

## Results

```text
                         model model_type  val_rmse    val_r2  test_rmse   test_r2
             BaselineTrainMean   baseline  7.168583 -3.015884   7.270251 -0.082497
                  RandomForest         ml  7.772168 -3.720619   6.962877  0.007100
                    ExtraTrees         ml  7.783563 -3.734471   6.962794  0.007124
                         Ridge         ml  8.177381 -4.225683   7.026976 -0.011265
BaselinePreviousYearSameCounty   baseline  9.029370 -5.371320   6.529807  0.126770
```
