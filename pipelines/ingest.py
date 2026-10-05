"""
Bronze ingestion pipeline: NOAA HRRR S3 → Delta Lake (bronze layer).

Entry-point behaviour (``python -m pipelines.ingest``):
  1. If yesterday's data has never been ingested (no flag file present), ingest
     it first to ensure the table is fully caught up.
  2. Always ingest today's data incrementally.

The streaming trigger ``availableNow=True`` makes each run behave like a
one-shot batch: it ingests all files that arrived since the last checkpoint
and then shuts down.
"""

import datetime
import logging
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name
from pyspark.sql.types import (
    BinaryType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from utils.constants import BRONZE_TABLE_PATH, CHECKPOINT_BASE_PATH_BRONZE as CHECKPOINT_BASE_PATH
from utils.logging_utils import extract_streaming_metrics, log_run_summary, setup_logging
from utils.spark import build_spark_ingest

log = logging.getLogger(__name__)

_BINARY_SCHEMA = StructType([
    StructField("path",             StringType(),    False),
    StructField("modificationTime", TimestampType(), False),
    StructField("length",           LongType(),      False),
    StructField("content",          BinaryType(),    True),
])


# =============================================================================
# DATABRICKS VERSION (Auto Loader / cloudFiles)
# def ingest_hrrr_databricks_autoloader(spark: SparkSession) -> None:
#     target_table = "weather.hrrr.bronze"
#     checkpoint_dir = "/Volumes/weather/hrrr/checkpoints/bronze"
#
#     df_autoloader = (
#         spark.readStream.format("cloudFiles")
#         .option("cloudFiles.format", "binaryFile")
#         .option("pathGlobFilter", "*wrfsfcf00.grib2")
#         # Ingests across all date directories dynamically:
#         .load("s3a://noaa-hrrr-bdp-pds/hrrr.*/conus/")
#         .withColumn("ingested_at", current_timestamp())
#         .withColumn("source_file", input_file_name())
#     )
#
#     query = (
#         df_autoloader.writeStream
#         .format("delta")
#         .outputMode("append")
#         .option("checkpointLocation", checkpoint_dir)
#         .trigger(availableNow=True)
#         .toTable(target_table)
#     )
#     query.awaitTermination()
# =============================================================================


def ingest_hrrr_streaming(spark: SparkSession, date_str: str) -> dict:
    """
    Stream-ingest all ``*wrfsfcf00.grib2`` files for *date_str* from the NOAA
    S3 bucket into the bronze Delta table.

    Uses ``availableNow=True`` so the job terminates once all currently
    available files have been processed.

    Args:
        spark:    Active Spark session.
        date_str: Date in ``YYYYMMDD`` format (e.g. ``"20241015"``).

    Returns:
        Dictionary containing streaming execution metrics (rows, batches).
    """
    source_s3_path = f"s3a://noaa-hrrr-bdp-pds/hrrr.{date_str}/conus/"
    checkpoint_path = f"{CHECKPOINT_BASE_PATH}/{date_str}"

    log.info("Checking for new data in %s", source_s3_path)

    df_raw = (
        spark.readStream.format("binaryFile")
        .schema(_BINARY_SCHEMA)
        .option("pathGlobFilter", "*wrfsfcf00.grib2")
        .load(source_s3_path)
    )

    df_bronze = df_raw.withColumn("ingested_at", current_timestamp()).withColumn(
        "source_file", input_file_name()
    )

    query = (
        df_bronze.writeStream.format("delta")
        .option("checkpointLocation", checkpoint_path)
        .trigger(availableNow=True)
        .start(BRONZE_TABLE_PATH)
    )
    query.awaitTermination()
    metrics = extract_streaming_metrics(query)
    log.info(
        "Ingested %d new file(s)/row(s) for %s across %d micro-batch(es).",
        metrics["rows"],
        date_str,
        metrics["batches"],
    )
    return metrics


def ingest_yesterday(spark: SparkSession, day: str) -> dict:
    """
    Ingest the previous day's data exactly once.

    A flag file is written after a successful run so subsequent executions
    skip the (potentially expensive) re-ingest.

    Args:
        spark: Active Spark session.
        day:   Date in ``YYYYMMDD`` format.

    Returns:
        Dictionary containing streaming execution metrics.
    """
    flag_file = f"{CHECKPOINT_BASE_PATH}/yesterday_done_{day}.flag"

    if not os.path.exists(flag_file):
        log.info("First run for %s — verifying full data...", day)
        metrics = ingest_hrrr_streaming(spark, day)
        os.makedirs(CHECKPOINT_BASE_PATH, exist_ok=True)
        with open(flag_file, "w") as fh:
            fh.write("done")
        return metrics
    else:
        log.info("Already caught up on %s. Skipping.", day)
        return {"batches": 0, "rows": 0, "batch_details": []}


def main() -> None:
    setup_logging()
    start_time = time.time()

    spark = build_spark_ingest()
    try:
        today = datetime.datetime.now()
        yesterday = today - datetime.timedelta(days=1)
        ystdy_str = yesterday.strftime("%Y%m%d")
        today_str = today.strftime("%Y%m%d")

        yesterday_metrics = ingest_yesterday(spark, ystdy_str)

        log.info("Ingesting today's data (%s)...", today_str)
        today_metrics = ingest_hrrr_streaming(spark, today_str)

        total_rows = yesterday_metrics["rows"] + today_metrics["rows"]
        total_batches = yesterday_metrics["batches"] + today_metrics["batches"]

        log_run_summary(
            logger=log,
            stage_name="Bronze Ingestion (NOAA S3 -> Delta)",
            start_time=start_time,
            metrics={"rows": total_rows, "batches": total_batches},
            extra_info={
                "Target Delta Table": BRONZE_TABLE_PATH,
                "Partition (Today)": today_str,
                "Today Ingested Rows": f"{today_metrics['rows']:,}",
                "Yesterday Rows": f"{yesterday_metrics['rows']:,}",
            },
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

