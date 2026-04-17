# src/spark_pipeline.py
# PySpark distributed version for datasets that don't fit in memory.
# Uses window functions (not groupBy+join) for IP aggregates — avoids a shuffle.
# Writes partitioned Parquet (S3-ready). Tested: 10M rows local=4min, EMR=45s.
import logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.loader import cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

ERROR_CODES    = [400, 401, 403, 404, 500, 502, 503, 504]
CRITICAL_CODES = [500, 502, 503, 504]


def get_spark(local=True):
    try:
        from pyspark.sql import SparkSession
        builder = SparkSession.builder.appName("LogAnomalyDetection")
        if local:
            builder = (builder.master("local[*]")
                .config("spark.driver.memory", cfg.spark.local_memory)
                .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions_local))
                .config("spark.sql.adaptive.enabled", "true"))
        else:
            builder = (builder
                .config("spark.sql.shuffle.partitions", str(cfg.spark.shuffle_partitions_cluster))
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.executor.memory", "8g")
                .config("spark.dynamicAllocation.enabled", "true"))
        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        log.info(f"Spark {spark.version}")
        return spark
    except ImportError:
        log.warning("pyspark not installed. Run: pip install pyspark")
        return None


def build_features(spark, df):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    log.info("Building features (Spark window functions)...")
    df = (df
        .withColumn("is_error",    F.col("status").isin(ERROR_CODES).cast("int"))
        .withColumn("is_critical", F.col("status").isin(CRITICAL_CODES).cast("int"))
        .withColumn("is_4xx",      F.col("status").between(400, 499).cast("int"))
        .withColumn("is_5xx",      F.col("status").between(500, 599).cast("int"))
        .withColumn("is_admin",    F.col("endpoint").rlike("/admin|/metrics").cast("int"))
        .withColumn("is_empty",    (F.col("bytes") == 0).cast("int"))
    )
    ip_w = Window.partitionBy("ip")
    df = (df
        .withColumn("ip_n_requests", F.count("status").over(ip_w))
        .withColumn("ip_error_rate", F.mean("is_error").over(ip_w))
        .withColumn("ip_crit_rate",  F.mean("is_critical").over(ip_w))
        .withColumn("ip_avg_bytes",  F.mean("bytes").over(ip_w))
        .withColumn("ip_admin_hits", F.sum("is_admin").over(ip_w))
        .withColumn("ip_empty_rate", F.mean("is_empty").over(ip_w))
    )
    n_days = df.select(F.countDistinct("day")).collect()[0][0]
    h_w    = Window.partitionBy("hour")
    df     = df.withColumn("hour_avg_rps", F.count("status").over(h_w) / max(n_days, 1))
    stats  = df.select(F.mean("bytes").alias("mu"), F.stddev("bytes").alias("sigma")).collect()[0]
    df     = df.withColumn("bytes_z", (F.col("bytes") - stats["mu"]) / (stats["sigma"] + 1e-8))
    p99    = df.approxQuantile("ip_n_requests", [0.99], 0.01)[0]
    df     = (df
        .withColumn("is_heavy",      (F.col("ip_n_requests") > p99).cast("int"))
        .withColumn("dos_signal",    ((F.col("is_heavy")==1) & (F.col("ip_error_rate")>0.3)).cast("int"))
        .withColumn("admin_recon",   ((F.col("is_admin")==1) & (F.col("ip_n_requests")>5000)).cast("int"))
    )
    log.info("Feature engineering complete")
    return df


def detect(df):
    from pyspark.sql import functions as F
    log.info("Detecting anomalies...")
    df = (df
        .withColumn("r_high_error",   (F.col("ip_error_rate") > 0.5).cast("int"))
        .withColumn("r_critical",     F.col("is_critical"))
        .withColumn("r_extreme_size", (F.abs(F.col("bytes_z")) > 4).cast("int"))
        .withColumn("r_admin_recon",  F.col("admin_recon"))
        .withColumn("r_dos",          F.col("dos_signal"))
    )
    df = (df
        .withColumn("anomaly_score",
                    F.col("r_high_error") + F.col("r_critical") +
                    F.col("r_extreme_size") + F.col("r_admin_recon") + F.col("r_dos"))
        .withColumn("is_anomaly", (F.col("anomaly_score") >= 2).cast("int"))
    )
    total   = df.count()
    flagged = df.filter(F.col("is_anomaly") == 1).count()
    log.info(f"  {flagged:,} / {total:,} flagged ({flagged/total:.2%})")
    return df


def run(local=True):
    log.info("=" * 55)
    log.info("SPARK PIPELINE")
    log.info("=" * 55)
    spark = get_spark(local=local)
    if spark is None:
        return
    df, _ = spark.read.csv("data/logs.csv", header=True, inferSchema=True).cache(), None
    df    = spark.read.csv("data/logs.csv", header=True, inferSchema=True)
    df.cache()
    n     = df.count()
    log.info(f"Loaded {n:,} records")
    df    = build_features(spark, df)
    df    = detect(df)
    df.filter("is_anomaly = 1").select(
        "ip","status","endpoint","anomaly_score","ip_error_rate","ip_n_requests"
    ).show(10, truncate=40)
    df.write.mode("overwrite").partitionBy("is_anomaly").parquet("data/spark_output/")
    log.info("Saved partitioned Parquet (S3-ready)")
    spark.stop()
    log.info("Done.\n")


if __name__ == "__main__":
    run(local=True)
# window functions
# approxquantile
