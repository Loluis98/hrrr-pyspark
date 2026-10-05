"""
Silver → Gold pipeline: join silver weather data with US state capitals and
produce two aggregated Gold tables.

Gold table layout
-----------------
hourly_capitals  – raw hourly observations spatially joined to each capital
                   city (append-only).
daily_capitals   – daily min / avg / max for every metric per city/date
                   (upserted via ``replaceWhere`` to stay idempotent).

The pipeline runs as a micro-batch stream (``foreachBatch``) with
``availableNow=True`` so it processes all pending silver data and then exits.
"""

import logging
import time

from pyspark.sql import DataFrame, SparkSession, functions as F

from utils.constants import (
    SILVER_TABLE_PATH,
    CAPITALS_TABLE_PATH,
    GOLD_CAPITALS_HOURLY_PATH,
    GOLD_CAPITALS_DAILY_PATH,
    CHECKPOINT_PATH_GOLD,
)
from utils.logging_utils import extract_streaming_metrics, log_run_summary, setup_logging
from utils.spark import build_spark_ingest

log = logging.getLogger(__name__)


_DROP_COLS = ["crain", "prate", "csnow", "source_path", "decode_error"]

METRICS = [
    "sp", "gust", "vis", "cape", "cin", "blh",
    "sdswrf", "sdlwrf", "suswrf", "sulwrf", "ishf", "slhtf", "gflux",
    "t_2m", "dpt", "u", "v",
]

_BBOX_COLS = ["min_lat", "max_lat", "min_lon", "max_lon", "lat", "lon"]


def _spatial_join(df_weather: DataFrame, df_capitals: DataFrame) -> DataFrame:
    """
    Join *df_weather* to *df_capitals* using each city's bounding box.

    The capitals table is broadcast because it is tiny (50 rows).
    """
    return df_weather.join(
        F.broadcast(df_capitals),
        (df_weather["latitude"]  >= df_capitals["min_lat"])
        & (df_weather["latitude"]  <= df_capitals["max_lat"])
        & (df_weather["longitude"] >= df_capitals["min_lon"])
        & (df_weather["longitude"] <= df_capitals["max_lon"]),
    )


def update_cities_hourly(df_batch: DataFrame, df_capitals: DataFrame) -> None:
    """
    Append the raw hourly batch data, spatially joined to capital cities,
    to the hourly Gold table.

    Args:
        df_batch:    One micro-batch from the silver stream.
        df_capitals: Broadcast-ready capitals reference DataFrame.
    """
    gold_hourly = (
        _spatial_join(df_batch, df_capitals)
        .drop(*_DROP_COLS, *_BBOX_COLS)
    )
    (
        gold_hourly.write
        .format("delta")
        .mode("append")
        .save(GOLD_CAPITALS_HOURLY_PATH)
    )


def update_cities_daily(df_batch: DataFrame, df_capitals: DataFrame) -> None:
    """
    Recalculate daily aggregations for every date present in *df_batch*.

    Reads back the full silver table for the affected dates to ensure daily
    aggregates are complete even if a date spans multiple micro-batches.
    Uses ``replaceWhere`` so the write is idempotent: running it twice for the
    same date is safe.

    Args:
        df_batch:    One micro-batch from the silver stream.
        df_capitals: Broadcast-ready capitals reference DataFrame.
    """
    dates = [
        row["date"] for row in df_batch.select("date").distinct().collect()
        if row["date"] is not None
    ]

    if not dates:
        return

    log.info("Recalculating Gold daily metrics for dates: %s", dates)

    spark: SparkSession = df_batch.sparkSession
    df_silver = spark.read.format("delta").load(SILVER_TABLE_PATH)
    df_silver_filtered = df_silver.filter(F.col("date").isin(dates))

    joined = _spatial_join(df_silver_filtered, df_capitals)

    # Build min / avg / max aggregation expressions for every metric.
    agg_exprs = []
    for metric in METRICS:
        agg_exprs.extend([
            F.round(F.avg(metric), 2).alias(f"avg_{metric}"),
            F.round(F.max(metric), 2).alias(f"max_{metric}"),
            F.round(F.min(metric), 2).alias(f"min_{metric}"),
        ])

    gold_daily = joined.groupBy("city", "state", "date").agg(*agg_exprs)

    date_strs = [f"'{d.strftime('%Y-%m-%d')}'" for d in dates]
    replace_condition = f"date IN ({','.join(date_strs)})"

    (
        gold_daily.write
        .format("delta")
        .mode("overwrite")
        .option("replaceWhere", replace_condition)
        .save(GOLD_CAPITALS_DAILY_PATH)
    )


def _process_batch(df_batch: DataFrame, batch_id: int) -> None:
    """
    ``foreachBatch`` callback: fan-out each micro-batch to both Gold writers.

    Args:
        df_batch: Current micro-batch DataFrame.
        batch_id: Monotonically increasing batch identifier from Spark.
    """
    if df_batch.isEmpty():
        return

    log.info("--- Processing batch %s ---", batch_id)

    spark: SparkSession = df_batch.sparkSession
    df_capitals = spark.read.format("delta").load(CAPITALS_TABLE_PATH)

    update_cities_hourly(df_batch, df_capitals)
    update_cities_daily(df_batch, df_capitals)


def run_gold_pipeline(spark: SparkSession) -> dict:
    """
    Stream the silver Delta table and write to both Gold tables via
    ``foreachBatch``.

    Returns:
        Dictionary containing streaming execution metrics (rows, batches).
    """
    log.info("Starting Silver → Gold pipeline...")

    df_silver_stream = (
        spark.readStream
        .format("delta")
        .load(SILVER_TABLE_PATH)
    )

    query = (
        df_silver_stream.writeStream
        .foreachBatch(_process_batch)
        .option("checkpointLocation", CHECKPOINT_PATH_GOLD)
        .trigger(availableNow=True)
        .start()
    )

    query.awaitTermination()
    metrics = extract_streaming_metrics(query)
    log.info(
        "Gold pipeline finished stream processing: %d silver row(s) processed across %d micro-batch(es).",
        metrics["rows"],
        metrics["batches"],
    )
    return metrics


def main() -> None:
    setup_logging()
    start_time = time.time()
    spark = build_spark_ingest()
    try:
        metrics = run_gold_pipeline(spark)
        log_run_summary(
            logger=log,
            stage_name="Silver -> Gold Pipeline (City Aggregations)",
            start_time=start_time,
            metrics=metrics,
            extra_info={
                "Source Delta Table": SILVER_TABLE_PATH,
                "Daily Target Delta Table": GOLD_CAPITALS_DAILY_PATH,
                "Hourly Target Delta Table": GOLD_CAPITALS_HOURLY_PATH,
                "Capitals Reference Table": CAPITALS_TABLE_PATH,
                "Checkpoint Path": CHECKPOINT_PATH_GOLD,
            },
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
