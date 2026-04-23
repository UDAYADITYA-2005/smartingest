"""Central configuration for SmartIngest pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / os.getenv("DATA_DIR", "data")
RAW_DIR     = DATA_DIR / "raw"
BRONZE_DIR  = DATA_DIR / "bronze"
SILVER_DIR  = DATA_DIR / "silver"
GOLD_DIR    = DATA_DIR / "gold"
REPORTS_DIR = BASE_DIR / os.getenv("REPORTS_DIR", "reports")
LOGS_DIR    = BASE_DIR / os.getenv("LOGS_DIR", "logs")

for d in [RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── External APIs ──────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
WEATHER_API_URL  = "https://api.open-meteo.com/v1/forecast"

# Weather: fetch for New Delhi (representative Indian location for agriculture context)
WEATHER_PARAMS = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
    "timezone": "Asia/Kolkata",
    "past_days": 30,
}

# ── Data sources ───────────────────────────────────────────────────────────
# We generate realistic synthetic data that mirrors data.gov.in schemas
# so the project works offline and in any environment.
CROP_RECORDS     = 5_000
HOSPITAL_RECORDS = 3_000

# ── Quality thresholds ─────────────────────────────────────────────────────
NULL_RATE_THRESHOLD        = 0.10   # flag if >10 % nulls in any column
ROW_DROP_THRESHOLD         = 0.15   # flag if >15 % rows dropped in Silver
ANOMALY_CONTAMINATION      = 0.05   # Isolation Forest contamination param

# ── Pipeline metadata ──────────────────────────────────────────────────────
PIPELINE_VERSION = "1.0.0"