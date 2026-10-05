"""
Bronze → Silver pipeline: decode raw GRIB2 binaries into structured weather
variables and write the results to the silver Delta table.

Each row in the bronze table holds the raw bytes of one GRIB2 file.
``decode_grib_partitions`` is a Pandas UDF (mapInPandas) that:
  1. Writes each file to a temp file on the executor.
  2. Opens it with cfgrib/xarray layer by layer.
  3. Merges the decoded variables into a flat DataFrame.

The silver table is partitioned by ``date`` (UTC valid time) for efficient
downstream filtering.
"""

import logging
import os
import tempfile
import time

import numpy as np
import pandas as pd
import xarray as xr

from pyspark.sql import SparkSession, functions as F

from utils.constants import BRONZE_TABLE_PATH, SILVER_TABLE_PATH, CHECKPOINT_PATH_SILVER as CHECKPOINT_PATH
from utils.logging_utils import extract_streaming_metrics, log_run_summary, setup_logging
from utils.spark import build_spark_ingest

log = logging.getLogger(__name__)

VARS = [
    "sp", "gust", "vis", "prate", "crain", "csnow",
    "cape", "cin", "blh",
    "sdswrf", "sdlwrf", "suswrf", "sulwrf", "ishf", "slhtf", "gflux",
    "t_2m", "dpt", "u", "v",
]

ALL_COLUMNS = ["valid_time", "latitude", "longitude"] + VARS + ["source_path", "decode_error"]

SILVER_SCHEMA = ", ".join(
    ["valid_time timestamp", "latitude float", "longitude float"]
    + [f"{v} float" for v in VARS]
    + ["source_path string", "decode_error string"]
)

LAYERS = [
    ({"typeOfLevel": "surface",         "stepType": "instant"}, {}),
    ({"typeOfLevel": "surface",         "stepType": "avg"},     {}),
    ({"typeOfLevel": "heightAboveGround", "level": 2,  "stepType": "instant"}, {"t2m": "t_2m", "d2m": "dpt"}),
    ({"typeOfLevel": "heightAboveGround", "level": 10, "stepType": "instant"}, {"u10": "u",   "v10": "v"}),
]

BBOX = None


def _decode_file(path: str) -> pd.DataFrame:
    """
    Decode a single GRIB2 file at *path* into a flat pandas DataFrame.

    Iterates over LAYERS, merging each decoded DataArray into a single
    xarray Dataset before converting to a DataFrame. Latitude/longitude
    coordinates are preserved from the first layer that exposes them.
    """
    arrays: dict = {}
    stamps: dict = {}
    layer_errors: list = []
    lat_da = lon_da = None

    for keys, rename in LAYERS:
        try:
            ds = xr.open_dataset(
                path,
                engine="cfgrib",
                backend_kwargs={"indexpath": "", "filter_by_keys": keys},
            )
        except Exception as exc:
            layer_errors.append(f"{keys}: {type(exc).__name__}: {exc}")
            continue

        with ds:
            if lat_da is None and "latitude" in ds.coords and "longitude" in ds.coords:
                lat_da = ds["latitude"].load()
                lon_da = ds["longitude"].load()
            if "valid_time" in ds.coords and "valid_time" not in stamps:
                stamps["valid_time"] = pd.Timestamp(ds["valid_time"].values)
            for name, da in ds.data_vars.items():
                out = rename.get(name, name)
                if out in VARS and out not in arrays:
                    arrays[out] = da.load().reset_coords(drop=True)

    if not arrays:
        raise RuntimeError("No layers decoded. " + " | ".join(layer_errors))

    merged = xr.Dataset(arrays)
    if lat_da is not None:
        merged = merged.assign_coords(latitude=lat_da, longitude=lon_da)

    df = merged.to_dataframe().reset_index()

    if BBOX:
        (lat_min, lat_max), (lon_min, lon_max) = BBOX["lat"], BBOX["lon"]
        df = df[
            (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
            & (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max)
        ]

    df["valid_time"] = stamps.get("valid_time", pd.NaT)
    df = df.reindex(columns=ALL_COLUMNS)
    float_cols = ["latitude", "longitude"] + VARS
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def decode_grib_partitions(iterator):
    """
    ``mapInPandas`` UDF: decode each GRIB2 binary row into weather records.

    For each row in the bronze table, the raw ``content`` bytes are written to
    a temporary file, decoded via ``_decode_file``, and yielded back to Spark.
    Errors are captured per-file and stored in the ``decode_error`` column so
    bad files never block the pipeline.
    """
    for pdf in iterator:
        for row in pdf.itertuples(index=False):
            source = getattr(row, "path", None)
            fd, tmp_path = tempfile.mkstemp(suffix=".grib2")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(row.content)
                df = _decode_file(tmp_path)
                df["source_path"] = source
                df["decode_error"] = None
                yield df
            except Exception as exc:
                err = pd.DataFrame({c: [None] for c in ALL_COLUMNS})
                err["source_path"] = source
                err["decode_error"] = f"{type(exc).__name__}: {exc}"
                yield err
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


def run_silver_pipeline(spark: SparkSession) -> dict:
    """
    Stream the bronze Delta table through the GRIB2 decoder and write results
    to the silver Delta table, partitioned by ``date``.

    Returns:
        Dictionary containing streaming execution metrics (rows, batches).
    """
    log.info("Starting Bronze → Silver pipeline...")

    # Process one GRIB2 file per Arrow batch to keep executor memory bounded.
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1")

    df_bronze = (
        spark.readStream.format("delta")
        .option("maxBytesPerTrigger", "8g")
        .load(BRONZE_TABLE_PATH)
    )

    df_silver = (
        df_bronze.mapInPandas(decode_grib_partitions, schema=SILVER_SCHEMA)
        .withColumn("date", F.to_date("valid_time"))
        .select("date", *ALL_COLUMNS)
    )

    query = (
        df_silver.writeStream
        .format("delta")
        .partitionBy("date")
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(availableNow=True)
        .start(SILVER_TABLE_PATH)
    )
    query.awaitTermination()
    metrics = extract_streaming_metrics(query)
    log.info(
        "Silver pipeline finished stream processing: %d bronze row(s)/file(s) decoded across %d micro-batch(es).",
        metrics["rows"],
        metrics["batches"],
    )
    return metrics


def main() -> None:
    setup_logging()
    start_time = time.time()
    spark = build_spark_ingest()
    try:
        metrics = run_silver_pipeline(spark)
        log_run_summary(
            logger=log,
            stage_name="Bronze -> Silver Pipeline (GRIB2 Decode)",
            start_time=start_time,
            metrics=metrics,
            extra_info={
                "Source Delta Table": BRONZE_TABLE_PATH,
                "Target Delta Table": SILVER_TABLE_PATH,
                "Checkpoint Path": CHECKPOINT_PATH,
            },
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
