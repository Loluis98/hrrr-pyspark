"""
Gold → ML pipeline: Rain classification from HRRR gold daily features.

Architecture note
-----------------
This pipeline intentionally does NOT create a new medallion layer.  The gold
daily table already provides the business-level aggregation (city, date, daily
avg/min/max for every metric).  The ML feature-engineering step (fetching rain
labels, scaling, training) lives *on top of* gold rather than inside the lake —
a common pattern sometimes called the "platinum" or "ML serving" layer.

Pipeline overview
-----------------
1. Load ``gold/daily_capitals`` via pandas + pyarrow (no Spark needed here).
2. Fetch historical daily precipitation totals from the Open-Meteo API for
   every (city, date) pair present in the gold table.
   API docs: https://open-meteo.com/en/docs/historical-weather-api
3. Build binary rain label:  rained = 1 if precipitation_sum > 0 else 0.
4. Merge labels with gold features → training DataFrame.
5. Train a Logistic Regression classifier and evaluate it.
6. Print a summary report (accuracy, confusion matrix, top coefficients).

Usage
-----
    # From the project root, using the databricks conda env:
    conda run -n databricks python -m pipelines.rain_prediction

    # Or in a notebook:
    from pipelines.rain_prediction import run_rain_pipeline
    results = run_rain_pipeline()
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.constants import GOLD_CAPITALS_DAILY_PATH
from pipelines.capital_cities import CAPITALS_DATA
from utils.spark import build_spark_explore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Open-Meteo historical endpoint (free, no API key required).
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

#: Precipitation threshold in mm to define a "rainy" day.
RAIN_THRESHOLD_MM = 0.1

#: Gold features included in the model.  We intentionally exclude the raw
#: pressure (sp) and radiation flux columns to keep the story crisp for demos;
#: focus on the meteorologically intuitive predictors.
FEATURE_COLS = [
    "avg_t_2m",   # temperature 2m
    "min_t_2m",
    "max_t_2m",
    "avg_dpt",    # dew-point (moisture)
    "avg_cape",   # convective available potential energy
    "max_cape",
    "avg_cin",    # convective inhibition
    "avg_vis",    # visibility
    "min_vis",
    "avg_gust",   # wind gust
    "max_gust",
    "avg_blh",    # boundary layer height
    "avg_slhtf",  # latent heat flux (evaporation proxy)
    "avg_ishf",   # sensible heat flux
]

#: Random seed for reproducibility.
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 1: Load gold daily table
# ---------------------------------------------------------------------------

def load_gold_daily() -> pd.DataFrame:
    """
    Read the ``hrrr_gold_capitals_daily`` Delta table via Spark and return a
    pandas DataFrame.

    Uses ``build_spark_explore()`` — the same lightweight session used by the
    EDA notebooks — which registers all Delta tables in the SQL catalog so we
    can query by name rather than by raw path.
    """
    log.info("Starting Spark session to read hrrr_gold_capitals_daily …")
    spark = build_spark_explore()
    try:
        df_spark = spark.read.table("hrrr_gold_capitals_daily")
        log.info(
            "Read %d rows from hrrr_gold_capitals_daily covering %d cities",
            df_spark.count(),
            df_spark.select("city").distinct().count(),
        )
        df = df_spark.toPandas()
    finally:
        spark.stop()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# Step 2: Fetch rain labels from Open-Meteo
# ---------------------------------------------------------------------------

def _build_city_coords() -> dict[str, dict]:
    """Build a dict {city_name: {lat, lon}} from the capitals data."""
    return {
        row["city"]: {"lat": row["lat"], "lon": row["lon"]}
        for row in CAPITALS_DATA
    }


def _fetch_rain_for_city(
    city: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    session: requests.Session,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily precipitation_sum from Open-Meteo for one city.

    Returns a DataFrame with columns [date, precipitation_sum] or None on error.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    try:
        resp = session.get(OPEN_METEO_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]).date,
            "precipitation_sum": data["daily"]["precipitation_sum"],
        })
        df["city"] = city
        return df
    except Exception as exc:
        log.warning("Failed to fetch rain for %s: %s", city, exc)
        return None


def fetch_rain_labels(gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Query Open-Meteo for each city in *gold_df* and return a DataFrame with
    columns [city, date, precipitation_sum, rained].

    Args:
        gold_df: The gold daily table (used to determine cities and date range).
    """
    city_coords = _build_city_coords()
    cities_needed = gold_df["city"].unique().tolist()

    start_date = str(gold_df["date"].min())
    end_date   = str(gold_df["date"].max())
    log.info(
        "Fetching rain labels for %d cities (%s → %s) via Open-Meteo …",
        len(cities_needed), start_date, end_date,
    )

    frames = []
    with requests.Session() as session:
        for i, city in enumerate(cities_needed):
            if city not in city_coords:
                log.warning("No coordinates for city: %s — skipping", city)
                continue
            coords = city_coords[city]
            df_rain = _fetch_rain_for_city(
                city, coords["lat"], coords["lon"],
                start_date, end_date, session,
            )
            if df_rain is not None:
                frames.append(df_rain)

            # Be polite to the free API: small sleep every 10 requests
            if (i + 1) % 10 == 0:
                log.info("  … fetched %d / %d cities", i + 1, len(cities_needed))
                time.sleep(0.3)

    if not frames:
        raise RuntimeError("No rain data fetched — check network connectivity.")

    rain_df = pd.concat(frames, ignore_index=True)
    rain_df["rained"] = (rain_df["precipitation_sum"].fillna(0) > RAIN_THRESHOLD_MM).astype(int)
    log.info(
        "Rain labels ready: %d rows, rain rate = %.1f%%",
        len(rain_df),
        100 * rain_df["rained"].mean(),
    )
    return rain_df


# ---------------------------------------------------------------------------
# Step 3: Build feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(gold_df: pd.DataFrame, rain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge gold features with rain labels.

    Returns a clean DataFrame with FEATURE_COLS + ['rained'] and no NaN rows.
    """
    merged = gold_df.merge(
        rain_df[["city", "date", "precipitation_sum", "rained"]],
        on=["city", "date"],
        how="inner",
    )
    available_features = [c for c in FEATURE_COLS if c in merged.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        log.warning("Features missing from gold table: %s", missing)

    model_df = merged[["city", "date"] + available_features + ["precipitation_sum", "rained"]].copy()
    before = len(model_df)
    model_df = model_df.dropna(subset=available_features + ["rained"])
    log.info("Feature matrix: %d rows (dropped %d with NaN)", len(model_df), before - len(model_df))
    return model_df, available_features


# ---------------------------------------------------------------------------
# Step 4: Train and evaluate Logistic Regression
# ---------------------------------------------------------------------------

def train_and_evaluate(model_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """
    Train a Logistic Regression pipeline with StandardScaler and evaluate via
    5-fold stratified cross-validation (no separate test split since the dataset
    is small — this is a demo).

    Returns a results dict with metrics and model artefacts.
    """
    X = model_df[feature_cols].values
    y = model_df["rained"].values

    log.info("Training Logistic Regression: %d samples, %d features", len(X), len(feature_cols))

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",   # handle class imbalance gracefully
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")

    # Fit on full data for coefficient inspection
    pipe.fit(X, y)
    y_pred = pipe.predict(X)

    # Feature importances (log-odds coefficients → sorted by abs value)
    coefs = pipe.named_steps["clf"].coef_[0]
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    return {
        "pipeline": pipe,
        "cv_scores": cv_scores,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "train_accuracy": accuracy_score(y, y_pred),
        "confusion_matrix": confusion_matrix(y, y_pred),
        "classification_report": classification_report(y, y_pred, target_names=["No Rain", "Rain"]),
        "feature_importance": importance_df,
        "model_df": model_df,
        "feature_cols": feature_cols,
        "X": X,
        "y": y,
    }


# ---------------------------------------------------------------------------
# Step 5: Pretty-print results
# ---------------------------------------------------------------------------

def print_results(results: dict) -> None:
    """Print a human-friendly summary of the training results."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("  HRRR Gold → Rain Classifier  |  Logistic Regression Demo")
    print(sep)

    print(f"\n📊 Dataset:  {len(results['model_df'])} city-day samples")
    rain_rate = results["y"].mean()
    print(f"🌧️  Rain rate: {100 * rain_rate:.1f}%  |  Dry rate: {100*(1-rain_rate):.1f}%")

    print(f"\n🔁 Cross-validation (5-fold stratified):")
    print(f"   Accuracy = {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    print(f"   Individual folds: {[f'{s:.3f}' for s in results['cv_scores']]}")

    print(f"\n📋 Classification report (trained on full dataset):")
    print(results["classification_report"])

    print(f"🧩 Confusion matrix (rows=actual, cols=predicted):")
    cm = results["confusion_matrix"]
    print(f"              Pred Dry  Pred Rain")
    print(f"  Actual Dry    {cm[0,0]:>5}     {cm[0,1]:>5}")
    print(f"  Actual Rain   {cm[1,0]:>5}     {cm[1,1]:>5}")

    print(f"\n🔑 Top feature importances (by log-odds coefficient magnitude):")
    print(f"   {'Feature':<20} {'Coefficient':>12}  {'Direction'}")
    print(f"   {'-'*20} {'-'*12}  {'-'*12}")
    for _, row in results["feature_importance"].head(10).iterrows():
        direction = "→ More rain" if row["coefficient"] > 0 else "→ Less rain"
        print(f"   {row['feature']:<20} {row['coefficient']:>12.4f}  {direction}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_rain_pipeline() -> dict:
    """
    Execute the full rain prediction pipeline.

    Returns the results dict produced by :func:`train_and_evaluate`.
    """
    # 1. Load gold
    gold_df = load_gold_daily()

    # 2. Fetch labels
    rain_df = fetch_rain_labels(gold_df)

    # 3. Feature matrix
    model_df, feature_cols = build_feature_matrix(gold_df, rain_df)

    # 4. Train
    results = train_and_evaluate(model_df, feature_cols)

    # 5. Report
    print_results(results)

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    run_rain_pipeline()


if __name__ == "__main__":
    main()
