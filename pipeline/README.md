# ML Pipeline (Random Forest)

This folder contains the Spark ML pipeline implementation.

## What changed
- Logistic Regression replaced with **RandomForestClassifier**
- Added **hyperparameter tuning** via `TrainValidationSplit`
- Added **overfitting prevention** through:
  - constrained tree depth (`maxDepth`)
  - minimum samples per node (`minInstancesPerNode`)
  - minimum info gain (`minInfoGain`)
  - row subsampling (`subsamplingRate`)
  - feature subsampling (`featureSubsetStrategy`)
- Uses a proper **train/test split** and evaluates on the hold-out test set to keep performance realistic (not perfect).

## Run
Requires a working Spark environment (e.g., `spark-submit` available) and PySpark.

Example:
```bash
spark-submit random_forest_pipeline.py
```

The script prints metrics and the tuned best hyperparameters.
