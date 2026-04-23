"""
Silver Layer — Clean, Validate, Deduplicate
Applies schema enforcement, null handling, deduplication, type casting.
Emits a quality report dict for each source.
"""
import pandas as pd
import numpy as np
from datetime import datetime

from config import BRONZE_DIR, SILVER_DIR, NULL_RATE_THRESHOLD, ROW_DROP_THRESHOLD
from utils import logger, timed, save_parquet, load_parquet, save_json, run_ts


# ── Schema definitions ────────────────────────────────────────────────────

CROP_SCHEMA = {
    "state_name":        "string",
    "district_name":     "string",
    "crop_year":         "int64",
    "season":            "string",
    "crop":              "string",
    "area_hectares":     "float64",
    "yield_kg_per_ha":   "float64",
    "production_tonnes": "float64",
}

HOSPITAL_SCHEMA = {
    "hospital_id":      "string",
    "hospital_name":    "string",
    "state":            "string",
    "district":         "string",
    "hospital_type":    "string",
    "speciality":       "string",
    "beds_total":       "int64",
    "beds_available":   "int64",
    "doctors_count":    "int64",
    "nurses_count":     "int64",
    "established_year": "int64",
    "accredited":       "bool",
    "latitude":         "float64",
    "longitude":        "float64",
    "last_updated":     "datetime64[ns]",
}

WEATHER_SCHEMA = {
    "date":              "datetime64[ns]",
    "temp_max_c":        "float64",
    "temp_min_c":        "float64",
    "precipitation_mm":  "float64",
    "windspeed_max_kph": "float64",
}

VALID_SEASONS  = {"Kharif", "Rabi", "Zaid"}
VALID_HOS_TYPES = {"Government", "Private", "Trust", "Charitable"}


# ── Core helpers ──────────────────────────────────────────────────────────

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip column names."""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _null_report(df: pd.DataFrame) -> dict:
    total = len(df)
    return {
        col: round(df[col].isna().sum() / total, 4) if total > 0 else 0.0
        for col in df.columns
    }


def _cast_schema(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
    """Cast columns to schema types; return df + list of failed casts."""
    errors = []
    for col, dtype in schema.items():
        if col not in df.columns:
            errors.append(f"MISSING_COLUMN:{col}")
            continue
        try:
            if dtype == "string":
                df[col] = df[col].astype("string")
            elif dtype in ("int64", "float64"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if dtype == "int64":
                    df[col] = df[col].round().astype("Int64")
            elif dtype == "bool":
                df[col] = df[col].astype("boolean")
            elif "datetime" in dtype:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception as e:
            errors.append(f"CAST_ERROR:{col}:{e}")
    return df, errors


def _quality_report(
    source: str,
    rows_in: int,
    rows_out: int,
    null_rates: dict,
    schema_errors: list,
    extra_flags: list = None,
) -> dict:
    drop_rate = round((rows_in - rows_out) / rows_in, 4) if rows_in > 0 else 0.0
    high_null_cols = [c for c, r in null_rates.items() if r > NULL_RATE_THRESHOLD]
    flags = []
    if drop_rate > ROW_DROP_THRESHOLD:
        flags.append(f"HIGH_DROP_RATE:{drop_rate:.2%}")
    flags += [f"HIGH_NULL:{c}={v:.2%}" for c, v in null_rates.items() if v > NULL_RATE_THRESHOLD]
    if extra_flags:
        flags += extra_flags

    return {
        "source":         source,
        "rows_in":        rows_in,
        "rows_out":       rows_out,
        "rows_dropped":   rows_in - rows_out,
        "drop_rate":      drop_rate,
        "null_rates":     null_rates,
        "high_null_cols": high_null_cols,
        "schema_errors":  schema_errors,
        "flags":          flags,
        "clean":          len(flags) == 0,
    }


# ── Source cleaners ───────────────────────────────────────────────────────

@timed
def clean_crop(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logger.info("Cleaning crop production…")
    rows_in = len(df)

    df = _standardize_columns(df)
    # Drop pipeline metadata cols from bronze
    df = df[[c for c in df.columns if not c.startswith("_pipeline")]]

    # Cast schema
    df, schema_errors = _cast_schema(df, CROP_SCHEMA)

    # Drop rows with null in key columns
    key_cols = ["state_name", "crop_year", "crop", "production_tonnes"]
    df = df.dropna(subset=key_cols)

    # Business rule: area and production must be positive
    df = df[(df["area_hectares"].fillna(0) >= 0) & (df["production_tonnes"].fillna(0) >= 0)]

    # Validate season
    extra_flags = []
    invalid_seasons = (~df["season"].isin(VALID_SEASONS)) & df["season"].notna()
    if invalid_seasons.sum() > 0:
        extra_flags.append(f"INVALID_SEASON_VALUES:{invalid_seasons.sum()}")
        df.loc[invalid_seasons, "season"] = pd.NA

    # Deduplicate
    dedup_cols = ["state_name", "district_name", "crop_year", "season", "crop"]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_cols)
    dupes_removed = before_dedup - len(df)
    if dupes_removed > 0:
        extra_flags.append(f"DUPES_REMOVED:{dupes_removed}")

    null_rates = _null_report(df)
    report = _quality_report("crop", rows_in, len(df), null_rates, schema_errors, extra_flags)

    save_parquet(df, SILVER_DIR / "crop_production.parquet", layer="silver")
    logger.success(f"Silver crop: {rows_in:,} → {len(df):,} rows. Flags: {report['flags']}")
    return df, report


@timed
def clean_hospital(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logger.info("Cleaning hospital infrastructure…")
    rows_in = len(df)

    df = _standardize_columns(df)
    df = df[[c for c in df.columns if not c.startswith("_pipeline")]]

    df, schema_errors = _cast_schema(df, HOSPITAL_SCHEMA)

    # Drop rows missing hospital_id or state
    df = df.dropna(subset=["hospital_id", "state"])

    # Beds available can't exceed total — fix or null
    mask = df["beds_available"] > df["beds_total"]
    df.loc[mask, "beds_available"] = df.loc[mask, "beds_total"]

    # Lat/lon range validation (India bounding box)
    extra_flags = []
    invalid_geo = (
        (df["latitude"].notna()  & ((df["latitude"]  < 6)  | (df["latitude"]  > 37))) |
        (df["longitude"].notna() & ((df["longitude"] < 67) | (df["longitude"] > 98)))
    )
    if invalid_geo.sum() > 0:
        extra_flags.append(f"INVALID_GEO:{invalid_geo.sum()}")
        df.loc[invalid_geo, ["latitude", "longitude"]] = pd.NA

    # Deduplicate on hospital_id
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["hospital_id"])
    dupes_removed = before_dedup - len(df)
    if dupes_removed > 0:
        extra_flags.append(f"DUPES_REMOVED:{dupes_removed}")

    null_rates = _null_report(df)
    report = _quality_report("hospital", rows_in, len(df), null_rates, schema_errors, extra_flags)

    save_parquet(df, SILVER_DIR / "hospital_infrastructure.parquet", layer="silver")
    logger.success(f"Silver hospital: {rows_in:,} → {len(df):,} rows. Flags: {report['flags']}")
    return df, report


@timed
def clean_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logger.info("Cleaning weather data…")
    rows_in = len(df)

    df = _standardize_columns(df)
    df = df[[c for c in df.columns if not c.startswith("_pipeline")]]

    df, schema_errors = _cast_schema(df, WEATHER_SCHEMA)
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"])

    # Sanity: temp_max >= temp_min
    extra_flags = []
    bad_temp = df["temp_max_c"] < df["temp_min_c"]
    if bad_temp.sum() > 0:
        extra_flags.append(f"TEMP_INVERSION:{bad_temp.sum()}")
        df.loc[bad_temp, ["temp_max_c", "temp_min_c"]] = df.loc[bad_temp, ["temp_min_c", "temp_max_c"]].values

    null_rates = _null_report(df)
    report = _quality_report("weather", rows_in, len(df), null_rates, schema_errors, extra_flags)

    save_parquet(df, SILVER_DIR / "weather.parquet", layer="silver")
    logger.success(f"Silver weather: {rows_in:,} → {len(df):,} rows. Flags: {report['flags']}")
    return df, report


# ── Orchestrator ──────────────────────────────────────────────────────────

@timed
def run_silver() -> dict:
    logger.info("=" * 60)
    logger.info("SILVER LAYER — starting cleaning & validation")
    logger.info("=" * 60)

    df_crop     = load_parquet(BRONZE_DIR / "crop_production.parquet")
    df_hospital = load_parquet(BRONZE_DIR / "hospital_infrastructure.parquet")
    df_weather  = load_parquet(BRONZE_DIR / "weather_api.parquet")

    _, rpt_crop     = clean_crop(df_crop)
    _, rpt_hospital = clean_hospital(df_hospital)
    _, rpt_weather  = clean_weather(df_weather)

    silver_meta = {
        "layer":    "silver",
        "run_ts":   run_ts(),
        "reports":  {
            "crop":     rpt_crop,
            "hospital": rpt_hospital,
            "weather":  rpt_weather,
        },
        "total_flags": (
            len(rpt_crop["flags"]) +
            len(rpt_hospital["flags"]) +
            len(rpt_weather["flags"])
        ),
    }
    save_json(silver_meta, SILVER_DIR / "silver_meta.json")
    logger.success("Silver layer complete.")
    return silver_meta


if __name__ == "__main__":
    run_silver()