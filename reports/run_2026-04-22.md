# SmartIngest Pipeline Report
**Date:** 2026-04-22 13:57 UTC
**Report source:** template_fallback
**Pipeline version:** 1.0.0

---

**SmartIngest Pipeline Health Report — 2026-04-22 13:57 UTC**

**Ingestion Summary.** The Bronze layer successfully ingested data from three sources: a crop production CSV, a hospital infrastructure JSON file, and a weather REST API. Total rows ingested across all sources reached 8,137. All three sources completed without critical errors, and raw Parquet files were persisted with source metadata and timestamps.

**Data Quality Summary.** During Silver layer processing, the crop dataset moved from 5,100 to 4,228 rows (drop rate: 17.1%), with 2 quality flag(s) raised. The hospital dataset processed 3,000 rows down to 3,000 (drop rate: 0.0%). Weather data retained 37 of 37 records. Combined quality flags for this run: 2. No anomalies were detected; the run is within normal parameters.

**Analytics Summary.** The Gold layer produced 6 analytical tables: state-level crop aggregations, top producers by crop using window functions, hospital infrastructure KPIs by state, 7-day rolling weather averages, and a cross-domain crop-weather join. All tables were saved as both Parquet and CSV. The pipeline ran end-to-end successfully and the data is ready for consumption by downstream dashboards and API endpoints.


---

## Raw Metrics Snapshot
```json
{
  "bronze_total_rows": 8137,
  "silver_total_flags": 2,
  "gold_tables": {
    "crop_state_summary": 754,
    "crop_top_producers": 50,
    "hospital_state_summary": 60,
    "weather_rolling_avg": 37,
    "crop_weather_join": 140,
    "kpi_summary": 1
  },
  "anomaly_detected": false,
  "anomaly_score": -0.4317
}
```
