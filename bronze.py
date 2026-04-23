"""
Bronze Layer — Raw Ingestion
Ingests from 3 sources:
  1. CSV file  (crop production)
  2. JSON file (hospital infrastructure)
  3. REST API  (open-meteo weather — free, no key required)
Saves raw Parquet files with source metadata + timestamps.
"""
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import (
    RAW_DIR, BRONZE_DIR,
    WEATHER_API_URL, WEATHER_PARAMS,
)
from data_generator import generate_all
from utils import logger, timed, save_parquet, save_json, run_ts


# ── Source 1: CSV ─────────────────────────────────────────────────────────

@timed
def ingest_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    logger.info("Ingesting CSV: crop production…")
    df = pd.read_csv(path, dtype_backend="numpy_nullable")

    meta = {
        "source": "csv",
        "file": str(path),
        "ingestion_time": datetime.utcnow().isoformat(),
        "row_count": len(df),
        "columns": list(df.columns),
    }
    save_parquet(df, BRONZE_DIR / "crop_production.parquet", layer="bronze")
    logger.success(f"Bronze CSV: {len(df):,} rows ingested.")
    return df, meta


# ── Source 2: JSON ────────────────────────────────────────────────────────

@timed
def ingest_json(path: Path) -> tuple[pd.DataFrame, dict]:
    logger.info("Ingesting JSON: hospital infrastructure…")
    df = pd.read_json(path, dtype_backend="numpy_nullable")

    meta = {
        "source": "json",
        "file": str(path),
        "ingestion_time": datetime.utcnow().isoformat(),
        "row_count": len(df),
        "columns": list(df.columns),
    }
    save_parquet(df, BRONZE_DIR / "hospital_infrastructure.parquet", layer="bronze")
    logger.success(f"Bronze JSON: {len(df):,} rows ingested.")
    return df, meta


# ── Source 3: REST API ────────────────────────────────────────────────────

@timed
def ingest_weather_api() -> tuple[pd.DataFrame, dict]:
    logger.info("Ingesting REST API: open-meteo weather…")
    try:
        resp = requests.get(WEATHER_API_URL, params=WEATHER_PARAMS, timeout=15)
        resp.raise_for_status()
        payload = resp.json()

        daily = payload["daily"]
        df = pd.DataFrame({
            "date":             daily["time"],
            "temp_max_c":       daily["temperature_2m_max"],
            "temp_min_c":       daily["temperature_2m_min"],
            "precipitation_mm": daily["precipitation_sum"],
            "windspeed_max_kph": daily["windspeed_10m_max"],
        })
        df["date"] = pd.to_datetime(df["date"])

        meta = {
            "source": "api",
            "url": WEATHER_API_URL,
            "latitude": payload["latitude"],
            "longitude": payload["longitude"],
            "timezone": payload["timezone"],
            "ingestion_time": datetime.utcnow().isoformat(),
            "row_count": len(df),
        }
        logger.success(f"Bronze API: {len(df):,} rows ingested.")

    except requests.RequestException as e:
        logger.warning(f"Weather API unavailable ({e}). Using fallback stub data.")
        import numpy as np
        rng = np.random.default_rng(99)
        dates = pd.date_range(end=datetime.utcnow(), periods=30, freq="D")
        df = pd.DataFrame({
            "date":              dates,
            "temp_max_c":        rng.uniform(25, 42, 30).round(1),
            "temp_min_c":        rng.uniform(15, 28, 30).round(1),
            "precipitation_mm":  rng.uniform(0, 30, 30).round(1),
            "windspeed_max_kph": rng.uniform(5, 40, 30).round(1),
        })
        meta = {
            "source": "api_stub",
            "ingestion_time": datetime.utcnow().isoformat(),
            "row_count": len(df),
            "note": "fallback stub — live API unreachable",
        }

    save_parquet(df, BRONZE_DIR / "weather_api.parquet", layer="bronze")
    return df, meta


# ── Orchestrator ──────────────────────────────────────────────────────────

@timed
def run_bronze() -> dict:
    """Run full Bronze layer. Returns combined metadata dict."""
    logger.info("=" * 60)
    logger.info("BRONZE LAYER — starting raw ingestion")
    logger.info("=" * 60)

    # Ensure raw data exists (generate synthetic data if not)
    raw_csv  = RAW_DIR / "crop_production.csv"
    raw_json = RAW_DIR / "hospital_infrastructure.json"
    if not raw_csv.exists() or not raw_json.exists():
        logger.info("Raw data not found — generating synthetic datasets…")
        generate_all()

    df_csv,     meta_csv     = ingest_csv(raw_csv)
    df_json,    meta_json    = ingest_json(raw_json)
    df_weather, meta_weather = ingest_weather_api()

    bronze_meta = {
        "layer": "bronze",
        "run_ts": run_ts(),
        "sources": {
            "crop_csv":      meta_csv,
            "hospital_json": meta_json,
            "weather_api":   meta_weather,
        },
        "total_rows": meta_csv["row_count"] + meta_json["row_count"] + meta_weather["row_count"],
    }
    save_json(bronze_meta, BRONZE_DIR / "bronze_meta.json")
    logger.success(f"Bronze layer complete. Total rows ingested: {bronze_meta['total_rows']:,}")
    return bronze_meta


if __name__ == "__main__":
    run_bronze()