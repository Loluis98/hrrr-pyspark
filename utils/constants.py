import os

# constants.py lives in utils/, so we walk two levels up to reach the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "weather_lake")

BRONZE_TABLE_PATH = f"{DATA_DIR}/bronze/hrrr"
SILVER_TABLE_PATH = f"{DATA_DIR}/silver/hrrr"
GOLD_CAPITALS_DAILY_PATH = f"{DATA_DIR}/gold/daily_capitals"
GOLD_CAPITALS_HOURLY_PATH = f"{DATA_DIR}/gold/hourly_capitals"
CAPITALS_TABLE_PATH = f"{DATA_DIR}/reference/us_capitals"

CHECKPOINT_BASE_PATH_BRONZE = f"{DATA_DIR}/checkpoints/hrrr_bronze"
CHECKPOINT_PATH_SILVER = f"{DATA_DIR}/checkpoints/hrrr_silver"
CHECKPOINT_PATH_GOLD = f"{DATA_DIR}/checkpoints/hrrr_gold"