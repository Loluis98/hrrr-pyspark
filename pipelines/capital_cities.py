"""
One-time seeder script: writes the US state-capital reference table to the
Delta Lake ``reference/us_capitals`` layer.

Run this script once (or whenever the reference data changes):

    python -m utils.capital_cities
    # or
    python utils/capital_cities.py
"""

import logging

import pandas as pd

from utils.constants import CAPITALS_TABLE_PATH
from utils.spark import build_spark_ingest

log = logging.getLogger(__name__)
BUFFER = 0.4

CAPITALS_DATA = [
    {"city": "Montgomery",    "state": "AL", "lat": 32.3615, "lon": -86.2791},
    {"city": "Juneau",        "state": "AK", "lat": 58.3019, "lon": -134.4197},
    {"city": "Phoenix",       "state": "AZ", "lat": 33.4484, "lon": -112.0740},
    {"city": "Little Rock",   "state": "AR", "lat": 34.7361, "lon":  -92.3311},
    {"city": "Sacramento",    "state": "CA", "lat": 38.5737, "lon": -121.4871},
    {"city": "Denver",        "state": "CO", "lat": 39.7392, "lon": -104.9847},
    {"city": "Hartford",      "state": "CT", "lat": 41.7670, "lon":  -72.6770},
    {"city": "Dover",         "state": "DE", "lat": 39.1619, "lon":  -75.5267},
    {"city": "Tallahassee",   "state": "FL", "lat": 30.4518, "lon":  -84.2727},
    {"city": "Atlanta",       "state": "GA", "lat": 33.7490, "lon":  -84.3880},
    {"city": "Honolulu",      "state": "HI", "lat": 21.3070, "lon": -157.8580},
    {"city": "Boise",         "state": "ID", "lat": 43.6187, "lon": -116.2146},
    {"city": "Springfield",   "state": "IL", "lat": 39.7817, "lon":  -89.6501},
    {"city": "Indianapolis",  "state": "IN", "lat": 39.7684, "lon":  -86.1581},
    {"city": "Des Moines",    "state": "IA", "lat": 41.5910, "lon":  -93.6030},
    {"city": "Topeka",        "state": "KS", "lat": 39.0400, "lon":  -95.6900},
    {"city": "Frankfort",     "state": "KY", "lat": 38.2009, "lon":  -84.8733},
    {"city": "Baton Rouge",   "state": "LA", "lat": 30.4583, "lon":  -91.1403},
    {"city": "Augusta",       "state": "ME", "lat": 44.3106, "lon":  -69.7795},
    {"city": "Annapolis",     "state": "MD", "lat": 38.9784, "lon":  -76.4922},
    {"city": "Boston",        "state": "MA", "lat": 42.3601, "lon":  -71.0589},
    {"city": "Lansing",       "state": "MI", "lat": 42.7325, "lon":  -84.5555},
    {"city": "St. Paul",      "state": "MN", "lat": 44.9537, "lon":  -93.0900},
    {"city": "Jackson",       "state": "MS", "lat": 32.2988, "lon":  -90.1848},
    {"city": "Jefferson City","state": "MO", "lat": 38.5767, "lon":  -92.1735},
    {"city": "Helena",        "state": "MT", "lat": 46.5891, "lon": -112.0391},
    {"city": "Lincoln",       "state": "NE", "lat": 40.8258, "lon":  -96.6852},
    {"city": "Carson City",   "state": "NV", "lat": 39.1638, "lon": -119.7674},
    {"city": "Concord",       "state": "NH", "lat": 43.2081, "lon":  -71.5326},
    {"city": "Trenton",       "state": "NJ", "lat": 40.2171, "lon":  -74.7429},
    {"city": "Santa Fe",      "state": "NM", "lat": 35.6870, "lon": -105.9378},
    {"city": "Albany",        "state": "NY", "lat": 42.6526, "lon":  -73.7562},
    {"city": "Raleigh",       "state": "NC", "lat": 35.7796, "lon":  -78.6382},
    {"city": "Bismarck",      "state": "ND", "lat": 46.8083, "lon": -100.7837},
    {"city": "Columbus",      "state": "OH", "lat": 39.9612, "lon":  -83.0007},
    {"city": "Oklahoma City", "state": "OK", "lat": 35.4676, "lon":  -97.5164},
    {"city": "Salem",         "state": "OR", "lat": 44.9429, "lon": -123.0351},
    {"city": "Harrisburg",    "state": "PA", "lat": 40.2732, "lon":  -76.8867},
    {"city": "Providence",    "state": "RI", "lat": 41.8240, "lon":  -71.4128},
    {"city": "Columbia",      "state": "SC", "lat": 34.0007, "lon":  -81.0348},
    {"city": "Pierre",        "state": "SD", "lat": 44.3683, "lon": -100.3510},
    {"city": "Nashville",     "state": "TN", "lat": 36.1627, "lon":  -86.7816},
    {"city": "Austin",        "state": "TX", "lat": 30.2672, "lon":  -97.7431},
    {"city": "Salt Lake City","state": "UT", "lat": 40.7608, "lon": -111.8910},
    {"city": "Montpelier",    "state": "VT", "lat": 44.2601, "lon":  -72.5754},
    {"city": "Richmond",      "state": "VA", "lat": 37.5407, "lon":  -77.4360},
    {"city": "Olympia",       "state": "WA", "lat": 47.0379, "lon": -122.9007},
    {"city": "Charleston",    "state": "WV", "lat": 38.3498, "lon":  -81.6326},
    {"city": "Madison",       "state": "WI", "lat": 43.0731, "lon":  -89.4012},
    {"city": "Cheyenne",      "state": "WY", "lat": 41.1399, "lon": -104.8202},
]


def build_capitals_dataframe() -> pd.DataFrame:
    """
    Build the capitals reference DataFrame with bounding-box columns.

    HRRR uses 0–360° longitude, so negative (western hemisphere) values are
    converted before the bounding box is calculated.
    """
    pdf = pd.DataFrame(CAPITALS_DATA)
    pdf["lon"] = pdf["lon"].apply(lambda x: x + 360 if x < 0 else x)

    pdf["min_lat"] = pdf["lat"] - BUFFER
    pdf["max_lat"] = pdf["lat"] + BUFFER
    pdf["min_lon"] = pdf["lon"] - BUFFER
    pdf["max_lon"] = pdf["lon"] + BUFFER

    return pdf


def seed_capitals_table() -> None:
    """Write (or overwrite) the us_capitals Delta table from the hard-coded data."""
    spark = build_spark_ingest()
    try:
        pdf = build_capitals_dataframe()
        df_capitals = spark.createDataFrame(pdf)
        df_capitals.write.format("delta").mode("overwrite").save(CAPITALS_TABLE_PATH)
        log.info("us_capitals Delta table written to %s", CAPITALS_TABLE_PATH)
    finally:
        spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    seed_capitals_table()
