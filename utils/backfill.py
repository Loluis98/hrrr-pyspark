"""
Historical backfill script: ingest the last N days of HRRR data into the
bronze Delta table in a single batch read.

Unlike the streaming ingest pipeline (``pipelines/ingest.py``), this script
uses ``spark.read`` (batch mode) so that Spark can plan across all days at
once and read them in parallel. This is considerably faster for large
historical windows.

Usage::

    python backfill.py             # backfill last 30 days (default)
    python backfill.py             # edit `days` kwarg below to change window

Note: if a specific day's directory doesn't exist in the NOAA bucket, Spark
may raise a PathNotFound error and abort the entire batch. In that case,
reduce the window or exclude the missing date manually.
"""

import datetime
import logging

from pyspark.sql.functions import current_timestamp, input_file_name

from utils.constants import BRONZE_TABLE_PATH
from utils.spark import build_spark_ingest

log = logging.getLogger(__name__)


def backfill_last_n_days(spark, days: int = 30) -> None:
    """
    Batch-ingest ``*wrfsfcf00.grib2`` files for the last *days* days.

    Days are counted backwards from yesterday (today is excluded because
    ``pipelines/ingest.py`` handles it in the daily incremental run).

    Args:
        spark: Active Spark session.
        days:  Number of historical days to backfill (default: 30).
    """
    final_day = datetime.datetime.now() - datetime.timedelta(days=1)

    paths = [
        f"s3a://noaa-hrrr-bdp-pds/hrrr.{(final_day - datetime.timedelta(days=i)).strftime('%Y%m%d')}/conus/"
        for i in range(1, days + 1)
    ]

    log.info("Starting backfill for the last %d days.", days)
    log.info("Earliest path : %s", paths[-1])
    log.info("Latest path   : %s", paths[0])

    try:
        df_raw = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*wrfsfcf00.grib2")
            .load(paths)
        )

        df_bronze = df_raw.withColumn("ingested_at", current_timestamp()).withColumn(
            "source_file", input_file_name()
        )

        log.info("Writing %d days of data to the bronze Delta table...", days)
        df_bronze.write.format("delta").mode("append").save(BRONZE_TABLE_PATH)
        log.info("Backfill complete.")

    except Exception as exc:
        log.error("Backfill failed: %s", exc)
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    spark = build_spark_ingest()
    try:
        backfill_last_n_days(spark, days=30)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
