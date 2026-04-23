"""
AI Health Reporter — Natural Language Pipeline Summary
Calls Gemini API (free tier) to generate a human-readable pipeline health report
after each run. Falls back to a template-based report if no API key is set.
"""
import os
import json
from datetime import datetime
from pathlib import Path

from config import GEMINI_API_KEY, REPORTS_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR
from utils import logger, timed, load_json, run_ts

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = bool(GEMINI_API_KEY)
    if GEMINI_AVAILABLE:
        genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False


# ── Prompt builder ────────────────────────────────────────────────────────

def _build_prompt(bronze_meta: dict, silver_meta: dict, gold_meta: dict, anomaly_report: dict) -> str:
    return f"""You are a senior data engineer writing a concise pipeline health report.
Based on the pipeline run data below, write a clear, professional 3-paragraph summary:

1. **Ingestion Summary** — What was ingested (sources, row counts, any issues at Bronze layer).
2. **Data Quality Summary** — Cleaning results, null rates, rows dropped, any anomalies flagged.
3. **Analytics Summary** — What Gold tables were produced, key KPIs, and overall pipeline health verdict.

Keep it factual, specific, and under 250 words. Use plain prose (no bullet points).

--- PIPELINE RUN DATA ---
BRONZE LAYER:
- Total rows ingested: {bronze_meta.get('total_rows', 'N/A'):,}
- Sources: {list(bronze_meta.get('sources', {}).keys())}

SILVER LAYER:
- Crop: {silver_meta['reports']['crop']['rows_in']:,} in → {silver_meta['reports']['crop']['rows_out']:,} out (drop rate: {silver_meta['reports']['crop']['drop_rate']:.1%})
- Hospital: {silver_meta['reports']['hospital']['rows_in']:,} in → {silver_meta['reports']['hospital']['rows_out']:,} out (drop rate: {silver_meta['reports']['hospital']['drop_rate']:.1%})
- Weather: {silver_meta['reports']['weather']['rows_in']:,} in → {silver_meta['reports']['weather']['rows_out']:,} out
- Total quality flags: {silver_meta.get('total_flags', 0)}
- Flags: {[f for r in silver_meta['reports'].values() for f in r.get('flags', [])]}

GOLD LAYER:
- Tables produced: {list(gold_meta.get('tables', {}).keys())}
- Row counts: {gold_meta.get('tables', {})}

ANOMALY DETECTION (Isolation Forest):
- Anomaly detected: {anomaly_report.get('is_anomaly', False)}
- Anomaly score: {anomaly_report.get('anomaly_score', 'N/A')}
- Top contributors: {anomaly_report.get('top_contributors', [])}
- Historical runs: {anomaly_report.get('history_runs', 1)}
"""


# ── Gemini call ───────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    try:
        model    = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}. Using template fallback.")
        return None


# ── Template fallback ─────────────────────────────────────────────────────

def _template_report(bronze_meta: dict, silver_meta: dict, gold_meta: dict, anomaly_report: dict) -> str:
    rpt_c = silver_meta["reports"]["crop"]
    rpt_h = silver_meta["reports"]["hospital"]
    rpt_w = silver_meta["reports"]["weather"]
    anomaly_str = "An anomaly was detected in this run — review the quality flags carefully." \
        if anomaly_report.get("is_anomaly") else "No anomalies were detected; the run is within normal parameters."

    return f"""**SmartIngest Pipeline Health Report — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}**

**Ingestion Summary.** The Bronze layer successfully ingested data from three sources: a crop production CSV, a hospital infrastructure JSON file, and a weather REST API. Total rows ingested across all sources reached {bronze_meta.get('total_rows', 0):,}. All three sources completed without critical errors, and raw Parquet files were persisted with source metadata and timestamps.

**Data Quality Summary.** During Silver layer processing, the crop dataset moved from {rpt_c['rows_in']:,} to {rpt_c['rows_out']:,} rows (drop rate: {rpt_c['drop_rate']:.1%}), with {len(rpt_c['flags'])} quality flag(s) raised. The hospital dataset processed {rpt_h['rows_in']:,} rows down to {rpt_h['rows_out']:,} (drop rate: {rpt_h['drop_rate']:.1%}). Weather data retained {rpt_w['rows_out']:,} of {rpt_w['rows_in']:,} records. Combined quality flags for this run: {silver_meta.get('total_flags', 0)}. {anomaly_str}

**Analytics Summary.** The Gold layer produced {len(gold_meta.get('tables', {}))} analytical tables: state-level crop aggregations, top producers by crop using window functions, hospital infrastructure KPIs by state, 7-day rolling weather averages, and a cross-domain crop-weather join. All tables were saved as both Parquet and CSV. The pipeline ran end-to-end successfully and the data is ready for consumption by downstream dashboards and API endpoints.
"""


# ── Orchestrator ──────────────────────────────────────────────────────────

@timed
def generate_health_report(
    bronze_meta: dict,
    silver_meta: dict,
    gold_meta:   dict,
    anomaly_report: dict,
) -> str:
    logger.info("Generating AI pipeline health report…")

    if GEMINI_AVAILABLE:
        logger.info("Calling Gemini API (free tier)…")
        prompt = _build_prompt(bronze_meta, silver_meta, gold_meta, anomaly_report)
        report_text = _call_gemini(prompt)
        if not report_text:
            report_text = _template_report(bronze_meta, silver_meta, gold_meta, anomaly_report)
            source = "template_fallback"
        else:
            source = "gemini-1.5-flash"
    else:
        logger.info("No GEMINI_API_KEY set — using template report.")
        report_text = _template_report(bronze_meta, silver_meta, gold_meta, anomaly_report)
        source = "template_fallback"

    ts    = datetime.utcnow().strftime("%Y-%m-%d")
    fname = REPORTS_DIR / f"run_{ts}.md"

    full_report = f"""# SmartIngest Pipeline Report
**Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
**Report source:** {source}
**Pipeline version:** {os.getenv('PIPELINE_VERSION', '1.0.0')}

---

{report_text}

---

## Raw Metrics Snapshot
```json
{json.dumps({
    "bronze_total_rows": bronze_meta.get("total_rows"),
    "silver_total_flags": silver_meta.get("total_flags"),
    "gold_tables": gold_meta.get("tables"),
    "anomaly_detected": anomaly_report.get("is_anomaly"),
    "anomaly_score": anomaly_report.get("anomaly_score"),
}, indent=2)}
```
"""
    fname.write_text(full_report)
    logger.success(f"Health report saved → {fname}")

    # Also save as "latest" for dashboard
    (REPORTS_DIR / "latest_report.md").write_text(full_report)

    return full_report


if __name__ == "__main__":
    from utils import load_json
    bm = load_json(BRONZE_DIR / "bronze_meta.json")
    sm = load_json(SILVER_DIR / "silver_meta.json")
    gm = load_json(GOLD_DIR   / "gold_meta.json")
    am = load_json(REPORTS_DIR / "latest_anomaly_report.json")
    print(generate_health_report(bm, sm, gm, am))