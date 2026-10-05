"""
Spark session factory functions for the HRRR pipeline.

Two flavours are provided:
  - build_spark_ingest()  - heavy session with S3/Delta support for production
                            pipeline stages (ingest, silver, gold).
  - build_spark_explore() - lightweight session pre-wired to the local Delta
                            tables; suited for notebooks / exploratory work.
"""

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from utils.constants import (
    BRONZE_TABLE_PATH,
    SILVER_TABLE_PATH,
    CAPITALS_TABLE_PATH,
    GOLD_CAPITALS_HOURLY_PATH,
    GOLD_CAPITALS_DAILY_PATH,
)


def build_spark_ingest() -> SparkSession:
    """
    Build a full-featured Spark session for production pipeline runs.

    Features:
    - 4 local workers and 16 GB driver memory (GRIB files are large)
    - Delta Lake catalog + S3A connector with anonymous credentials
    - Parquet dictionary encoding disabled to avoid OOM on GRIB payloads
    """
    builder = (
        SparkSession.builder.appName("hrrr-local-ingest")
        .master("local[4]")
        .config("spark.driver.memory", "16g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
            "io.delta:delta-spark_2.12:3.2.0",
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
        )
        .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
        .config("spark.hadoop.fs.s3a.connection.maximum", "100")
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
        .config("spark.hadoop.fs.s3a.attempts.maximum", "10")
        .config("spark.hadoop.fs.s3a.paging.maximum", "5000")
        # GRIB data is massive; enabling the dictionary causes OOM errors.
        .config("spark.hadoop.parquet.enable.dictionary", "false")
    )

    spark = configure_spark_with_delta_pip(
        builder,
        extra_packages=["org.apache.hadoop:hadoop-aws:3.3.4"],
    ).getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    return spark


def build_spark_explore() -> SparkSession:
    """
    Build a lightweight Spark session for exploratory data analysis (EDA).

    Differences from the ingest session:
    - Only 2 local workers and 6 GB driver memory
    - No S3 connector (reads local Delta tables only)
    - All Delta tables are auto-registered into the SQL catalog so that
      ``spark.sql("SELECT ...")`` queries work out of the box in notebooks.
    """
    builder = (
        SparkSession.builder.appName("hrrr-local-explore")
        .master("local[2]")
        .config("spark.driver.memory", "6g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    _register_tables(spark)

    return spark


def _register_tables(spark: SparkSession) -> None:
    """Register local Delta tables into the active Spark SQL catalog."""
    registrations = [
        ("hrrr_bronze", BRONZE_TABLE_PATH),
        ("hrrr_silver", SILVER_TABLE_PATH),
        ("us_capitals", CAPITALS_TABLE_PATH),
        ("hrrr_gold_capitals_hourly", GOLD_CAPITALS_HOURLY_PATH),
        ("hrrr_gold_capitals_daily", GOLD_CAPITALS_DAILY_PATH),
    ]
    for table_name, path in registrations:
        spark.sql(
            f"CREATE TABLE IF NOT EXISTS {table_name} "
            f"USING DELTA LOCATION '{path}'"
        )
