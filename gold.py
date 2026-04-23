"""
Gold Layer — Aggregations & Analytics-Ready Tables
Uses DuckDB to run SQL on Silver Parquet files.
Produces 5 analytical tables covering JOINs, GROUP BY, and window functions.
"""
import duckdb
import pandas as pd
from pathlib import Path

from config import SILVER_DIR, GOLD_DIR
from utils import logger, timed, save_parquet, save_json, run_ts


def _duckdb_conn() :
    """Return an in-memory DuckDB connection with Silver Parquets registered."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"""
        CREATE VIEW crop     AS SELECT * FROM read_parquet('{SILVER_DIR}/crop_production.parquet');
        CREATE VIEW hospital AS SELECT * FROM read_parquet('{SILVER_DIR}/hospital_infrastructure.parquet');
        CREATE VIEW weather  AS SELECT * FROM read_parquet('{SILVER_DIR}/weather.parquet');
    """)
    return con


# ── Gold table builders ───────────────────────────────────────────────────

@timed
def gold_crop_state_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """State-level crop production aggregation by year."""
    logger.info("Building: crop_state_summary…")
    df = con.execute("""
        SELECT
            state_name,
            crop_year,
            season,
            COUNT(DISTINCT crop)               AS crop_varieties,
            ROUND(SUM(area_hectares), 2)        AS total_area_ha,
            ROUND(SUM(production_tonnes), 2)    AS total_production_t,
            ROUND(AVG(yield_kg_per_ha), 2)      AS avg_yield_kg_ha,
            ROUND(MAX(production_tonnes), 2)    AS max_production_t
        FROM crop
        WHERE state_name IS NOT NULL
          AND crop_year  IS NOT NULL
        GROUP BY state_name, crop_year, season
        ORDER BY state_name, crop_year, season
    """).df()
    save_parquet(df, GOLD_DIR / "crop_state_summary.parquet", layer="gold")
    df.to_csv(GOLD_DIR / "crop_state_summary.csv", index=False)
    logger.success(f"crop_state_summary: {len(df):,} rows")
    return df


@timed
def gold_crop_top_producers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Top 5 producing states per crop using window functions."""
    logger.info("Building: crop_top_producers…")
    df = con.execute("""
        WITH state_crop AS (
            SELECT
                state_name,
                crop,
                ROUND(SUM(production_tonnes), 2) AS total_production_t
            FROM crop
            WHERE state_name IS NOT NULL AND crop IS NOT NULL
            GROUP BY state_name, crop
        ),
        ranked AS (
            SELECT *,
                RANK() OVER (PARTITION BY crop ORDER BY total_production_t DESC) AS rank_in_crop
            FROM state_crop
        )
        SELECT * FROM ranked
        WHERE rank_in_crop <= 5
        ORDER BY crop, rank_in_crop
    """).df()
    save_parquet(df, GOLD_DIR / "crop_top_producers.parquet", layer="gold")
    df.to_csv(GOLD_DIR / "crop_top_producers.csv", index=False)
    logger.success(f"crop_top_producers: {len(df):,} rows")
    return df


@timed
def gold_hospital_state_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """State-level hospital infrastructure summary."""
    logger.info("Building: hospital_state_summary…")
    df = con.execute("""
        SELECT
            state,
            hospital_type,
            COUNT(*)                             AS hospital_count,
            SUM(beds_total)                      AS total_beds,
            ROUND(AVG(beds_total), 1)            AS avg_beds_per_hospital,
            SUM(doctors_count)                   AS total_doctors,
            SUM(nurses_count)                    AS total_nurses,
            ROUND(
                100.0 * SUM(CASE WHEN accredited THEN 1 ELSE 0 END) / COUNT(*), 2
            )                                    AS accreditation_pct
        FROM hospital
        WHERE state IS NOT NULL
        GROUP BY state, hospital_type
        ORDER BY state, hospital_type
    """).df()
    save_parquet(df, GOLD_DIR / "hospital_state_summary.parquet", layer="gold")
    df.to_csv(GOLD_DIR / "hospital_state_summary.csv", index=False)
    logger.success(f"hospital_state_summary: {len(df):,} rows")
    return df


@timed
def gold_weather_rolling(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Weather with 7-day rolling averages using window functions."""
    logger.info("Building: weather_rolling_avg…")
    df = con.execute("""
        SELECT
            date,
            temp_max_c,
            temp_min_c,
            precipitation_mm,
            windspeed_max_kph,
            ROUND(AVG(temp_max_c)        OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS temp_max_7d_avg,
            ROUND(AVG(temp_min_c)        OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS temp_min_7d_avg,
            ROUND(SUM(precipitation_mm)  OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS precip_7d_sum,
            ROUND(AVG(windspeed_max_kph) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS wind_7d_avg
        FROM weather
        ORDER BY date
    """).df()
    save_parquet(df, GOLD_DIR / "weather_rolling_avg.parquet", layer="gold")
    df.to_csv(GOLD_DIR / "weather_rolling_avg.csv", index=False)
    logger.success(f"weather_rolling_avg: {len(df):,} rows")
    return df


@timed
def gold_crop_weather_join(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Cross-domain JOIN: recent crop production vs weather patterns.
    Maps state-level crop data with national weather summary for the crop year.
    """
    logger.info("Building: crop_weather_join…")
    df = con.execute("""
        WITH weather_annual AS (
            SELECT
                YEAR(date)                         AS year,
                ROUND(AVG(temp_max_c), 2)          AS avg_temp_max,
                ROUND(AVG(temp_min_c), 2)          AS avg_temp_min,
                ROUND(SUM(precipitation_mm), 2)    AS total_precip_mm
            FROM weather
            GROUP BY YEAR(date)
        ),
        crop_annual AS (
            SELECT
                crop_year                          AS year,
                crop,
                ROUND(SUM(production_tonnes), 2)   AS national_production_t,
                ROUND(AVG(yield_kg_per_ha), 2)     AS avg_national_yield
            FROM crop
            WHERE crop IS NOT NULL
            GROUP BY crop_year, crop
        )
        SELECT
            c.*,
            w.avg_temp_max,
            w.avg_temp_min,
            w.total_precip_mm
        FROM crop_annual c
        LEFT JOIN weather_annual w ON c.year = w.year
        ORDER BY c.year, c.crop
    """).df()
    save_parquet(df, GOLD_DIR / "crop_weather_join.parquet", layer="gold")
    df.to_csv(GOLD_DIR / "crop_weather_join.csv", index=False)
    logger.success(f"crop_weather_join: {len(df):,} rows")
    return df


@timed
def gold_kpi_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Single-row KPI snapshot for the dashboard."""
    logger.info("Building: kpi_summary…")
    df = con.execute("""
        SELECT
            (SELECT COUNT(*)                      FROM crop)     AS crop_records,
            (SELECT COUNT(DISTINCT state_name)    FROM crop)     AS crop_states,
            (SELECT ROUND(SUM(production_tonnes)/1e6, 2) FROM crop) AS total_production_mt,
            (SELECT COUNT(*)                      FROM hospital) AS hospital_records,
            (SELECT SUM(beds_total)               FROM hospital) AS total_beds,
            (SELECT SUM(doctors_count)            FROM hospital) AS total_doctors,
            (SELECT COUNT(*)                      FROM weather)  AS weather_records,
            (SELECT ROUND(AVG(precipitation_mm), 2) FROM weather) AS avg_daily_precip_mm
    """).df()
    save_parquet(df, GOLD_DIR / "kpi_summary.parquet", layer="gold")
    df.to_csv(GOLD_DIR / "kpi_summary.csv", index=False)
    logger.success("kpi_summary built.")
    return df


# ── Orchestrator ──────────────────────────────────────────────────────────

@timed
def run_gold() -> dict:
    logger.info("=" * 60)
    logger.info("GOLD LAYER — starting aggregations")
    logger.info("=" * 60)

    con = _duckdb_conn()

    tables = {}
    tables["crop_state_summary"]    = gold_crop_state_summary(con)
    tables["crop_top_producers"]    = gold_crop_top_producers(con)
    tables["hospital_state_summary"] = gold_hospital_state_summary(con)
    tables["weather_rolling_avg"]   = gold_weather_rolling(con)
    tables["crop_weather_join"]     = gold_crop_weather_join(con)
    tables["kpi_summary"]           = gold_kpi_summary(con)

    con.close()

    gold_meta = {
        "layer":  "gold",
        "run_ts": run_ts(),
        "tables": {k: len(v) for k, v in tables.items()},
    }
    save_json(gold_meta, GOLD_DIR / "gold_meta.json")
    logger.success(f"Gold layer complete. {len(tables)} tables written.")
    return gold_meta


if __name__ == "__main__":
    run_gold()