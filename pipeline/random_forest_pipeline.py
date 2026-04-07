"""
Spark-based ad attribution & conversion prediction pipeline using Random Forest.

This script intentionally focuses on:
- Replacing Logistic Regression with RandomForestClassifier
- Hyperparameter tuning (TrainValidationSplit)
- Overfitting prevention (regularization via depth/minInstancesPerNode, feature subsampling, row subsampling)
- Realistic evaluation (hold-out test set; avoids leakage)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


@dataclass(frozen=True)
class PipelineMetrics:
    """Container for model evaluation metrics."""

    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1: float
    auc_pr: float


# PUBLIC_INTERFACE
def build_spark(app_name: str = "ad-attribution-rf-pipeline") -> SparkSession:
    """Create and return a SparkSession suitable for local or cluster runs."""
    return (
        SparkSession.builder.appName(app_name)
        # Keep defaults minimal; infra can override via spark-submit
        .getOrCreate()
    )


def _generate_synthetic_data(spark: SparkSession, n: int = 50_000, seed: int = 42) -> DataFrame:
    """
    Generate synthetic (noisy, realistic) ad conversion data.

    Note: We deliberately add label noise and overlapping class distributions so results
    do not become unrealistically perfect.
    """
    base = spark.range(0, n).withColumnRenamed("id", "row_id")
    df = (
        base.withColumn("seed", F.lit(seed))
        .withColumn("campaign", (F.rand(seed) * 6).cast("int"))
        .withColumn("channel", (F.rand(seed + 1) * 4).cast("int"))
        .withColumn("device", (F.rand(seed + 2) * 3).cast("int"))
        .withColumn("hour", (F.rand(seed + 3) * 24).cast("int"))
        .withColumn("impressions", (F.rand(seed + 4) * 25).cast("int"))
        .withColumn("clicks", (F.rand(seed + 5) * 5).cast("int"))
        .withColumn("spend", (F.rand(seed + 6) * 10.0))
        .withColumn("prev_conversions_7d", (F.rand(seed + 7) * 3).cast("int"))
        .withColumn("user_quality", F.rand(seed + 8))
    )

    # Latent propensity score: nonlinear + categorical effects + noise.
    # Kept intentionally "messy" so RF can't trivially get 100%.
    score = (
        F.lit(-2.2)
        + 0.10 * F.col("impressions")
        + 0.55 * F.col("clicks")
        + 0.12 * F.col("prev_conversions_7d")
        - 0.08 * F.col("spend")
        + 0.8 * (F.col("channel") == 1).cast("double")
        + 0.5 * (F.col("device") == 2).cast("double")
        + 0.03 * (F.col("hour") - 12)
        + 0.9 * (F.col("campaign").isin([2, 4])).cast("double")
        + 0.8 * (F.col("user_quality") - 0.5)
        + (F.rand(seed + 9) - 0.5) * 1.2  # noise
    )

    # Convert score to probability using sigmoid.
    prob = 1 / (1 + F.exp(-score))

    # Sample label from Bernoulli(prob)
    labeled = df.withColumn("label_raw", (F.rand(seed + 10) < prob).cast("int"))

    # Inject label noise: flip ~4% of labels
    noisy = labeled.withColumn(
        "label",
        F.when(F.rand(seed + 11) < 0.04, 1 - F.col("label_raw")).otherwise(F.col("label_raw")).cast("int"),
    ).drop("label_raw")

    # Map int categories to strings (helps demonstrate StringIndexer/OneHotEncoder pipeline)
    out = (
        noisy.withColumn("campaign", F.concat(F.lit("c"), F.col("campaign").cast("string")))
        .withColumn("channel", F.concat(F.lit("ch"), F.col("channel").cast("string")))
        .withColumn("device", F.concat(F.lit("d"), F.col("device").cast("string")))
    )
    return out


def _build_pipeline_and_tuner(seed: int = 42) -> Tuple[TrainValidationSplit, List[str]]:
    """Create a tuned RandomForest pipeline and return (tvs, feature_cols_used)."""
    categorical_cols = ["campaign", "channel", "device"]
    numeric_cols = ["hour", "impressions", "clicks", "spend", "prev_conversions_7d", "user_quality"]

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in categorical_cols],
        outputCols=[f"{c}_oh" for c in categorical_cols],
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=[*numeric_cols, *[f"{c}_oh" for c in categorical_cols]],
        outputCol="features_raw",
        handleInvalid="keep",
    )

    # Scaling isn't necessary for RF, but keeping it in pipeline demonstrates typical FE flow.
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=False)

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        seed=seed,
        # Start with conservative defaults to avoid overfitting.
        numTrees=200,
        featureSubsetStrategy="sqrt",
        subsamplingRate=0.8,
        impurity="gini",
    )

    pipeline = Pipeline(stages=[*indexers, encoder, assembler, scaler, rf])

    # Hyperparameter tuning:
    # We search a small, realistic grid; we also keep constraints that discourage overfitting.
    param_grid = (
        ParamGridBuilder()
        .addGrid(rf.maxDepth, [4, 6, 8])
        .addGrid(rf.minInstancesPerNode, [20, 50])
        .addGrid(rf.minInfoGain, [0.0, 0.01])
        .addGrid(rf.subsamplingRate, [0.7, 0.85])
        .addGrid(rf.featureSubsetStrategy, ["sqrt", "log2"])
        .build()
    )

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    # TrainValidationSplit is cheaper than CV and still provides regularization via validation selection.
    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2,
        seed=seed,
    )

    return tvs, [*numeric_cols, *categorical_cols]


def _evaluate(pred: DataFrame) -> PipelineMetrics:
    """Compute a small set of classification metrics from predictions DataFrame."""
    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    rec_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )

    # For AUC-PR we use BinaryClassificationEvaluator on probability column.
    # Note: This can still be "too optimistic" if dataset is too separable; we keep synthetic noise to prevent that.
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    auc_pr_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="probability",
        metricName="areaUnderPR",
    )

    return PipelineMetrics(
        accuracy=float(acc_eval.evaluate(pred)),
        precision_weighted=float(prec_eval.evaluate(pred)),
        recall_weighted=float(rec_eval.evaluate(pred)),
        f1=float(f1_eval.evaluate(pred)),
        auc_pr=float(auc_pr_eval.evaluate(pred)),
    )


# PUBLIC_INTERFACE
def run_pipeline(n_rows: int = 50_000, seed: int = 42) -> Dict[str, Any]:
    """
    Run the full synthetic-data pipeline, tune Random Forest hyperparameters, and evaluate on a hold-out test set.

    Returns:
        Dict with keys:
        - metrics: dict of evaluation metrics on the test set
        - best_params: tuned hyperparameters (subset)
        - data_info: basic counts and label prevalence
    """
    spark = build_spark()
    try:
        df = _generate_synthetic_data(spark, n=n_rows, seed=seed)

        # Proper hold-out split to avoid leakage.
        train, test = df.randomSplit([0.8, 0.2], seed=seed)

        tvs, _ = _build_pipeline_and_tuner(seed=seed)
        tvs_model = tvs.fit(train)

        pred = tvs_model.transform(test).select("label", "prediction", "probability")
        metrics = _evaluate(pred)

        # Extract best model params (pipeline -> last stage is RF model)
        best_pipeline_model = tvs_model.bestModel
        rf_model = best_pipeline_model.stages[-1]

        best_params = {
            "maxDepth": int(rf_model.getOrDefault("maxDepth")),
            "minInstancesPerNode": int(rf_model.getOrDefault("minInstancesPerNode")),
            "minInfoGain": float(rf_model.getOrDefault("minInfoGain")),
            "subsamplingRate": float(rf_model.getOrDefault("subsamplingRate")),
            "featureSubsetStrategy": str(rf_model.getOrDefault("featureSubsetStrategy")),
            "numTrees": int(rf_model.getOrDefault("numTrees")),
        }

        # Useful realism checks
        label_stats = (
            df.groupBy("label").count().withColumn("pct", F.col("count") / F.sum("count").over()).orderBy("label")
        )
        label_stats_local = [row.asDict() for row in label_stats.collect()]

        return {
            "metrics": asdict(metrics),
            "best_params": best_params,
            "data_info": {
                "n_rows": int(df.count()),
                "train_rows": int(train.count()),
                "test_rows": int(test.count()),
                "label_distribution": label_stats_local,
            },
        }
    finally:
        spark.stop()


# PUBLIC_INTERFACE
def main() -> None:
    """CLI entrypoint for running the tuned Random Forest pipeline and printing results."""
    result = run_pipeline()
    print("=== Random Forest (tuned) results ===")
    print("Metrics:", result["metrics"])
    print("Best Params:", result["best_params"])
    print("Data Info:", result["data_info"])


if __name__ == "__main__":
    main()
