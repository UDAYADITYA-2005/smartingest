"""
SmartIngest Pipeline Orchestrator
Runs Bronze → Silver → Gold → Anomaly Detection → AI Report in sequence.
Simulates an Airflow-style DAG with timing, logging, and error handling.
"""
import sys
import time
import traceback
from datetime import datetime

from config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR, PIPELINE_VERSION
from utils import logger, save_json, load_json, run_ts

from bronze         import run_bronze
from silver         import run_silver
from gold           import run_gold
from quality_monitor import run_anomaly_detection
from ai_reporter    import generate_health_report


STEP_EMOJIS = {
    "bronze":   "🥉",
    "silver":   "🥈",
    "gold":     "🥇",
    "anomaly":  "🔍",
    "report":   "📄",
}


def _banner(title: str) -> None:
    width = 60
    logger.info("=" * width)
    logger.info(f"  SmartIngest v{PIPELINE_VERSION}  —  {title}")
    logger.info("=" * width)


def _step(name: str, fn, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    emoji = STEP_EMOJIS.get(name, "▶")
    logger.info(f"\n{emoji}  STEP: {name.upper()}")
    t0 = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.success(f"✅  {name} completed in {elapsed:.2f}s")
        return result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"❌  {name} FAILED after {elapsed:.2f}s: {e}")
        logger.debug(traceback.format_exc())
        raise


def run_pipeline(fail_fast: bool = True) -> dict:
    """
    Full pipeline run.
    Returns a summary dict with status of each step.
    """
    _banner("Starting Full Pipeline Run")
    pipeline_start = time.perf_counter()
    run_id  = run_ts()
    summary = {"run_id": run_id, "steps": {}, "status": "running"}

    steps = [
        ("bronze",  run_bronze,              []),
        ("silver",  run_silver,              []),
        ("gold",    run_gold,                []),
    ]

    metas = {}

    # ── Core ETL steps ─────────────────────────────────────────────────
    for step_name, fn, args in steps:
        try:
            result = _step(step_name, fn, *args)
            metas[step_name] = result
            summary["steps"][step_name] = {"status": "ok", "rows": result.get("total_rows", len(result.get("tables", {})))}
        except Exception as e:
            summary["steps"][step_name] = {"status": "failed", "error": str(e)}
            summary["status"] = "failed"
            if fail_fast:
                logger.error("Pipeline aborted (fail_fast=True).")
                _save_pipeline_summary(summary, run_id)
                return summary

    # ── AI Layer: anomaly detection ────────────────────────────────────
    try:
        silver_meta   = metas.get("silver") or load_json(SILVER_DIR / "silver_meta.json")
        anomaly_report = _step("anomaly", run_anomaly_detection, silver_meta)
        metas["anomaly"] = anomaly_report
        summary["steps"]["anomaly"] = {
            "status": "ok",
            "is_anomaly": anomaly_report.get("is_anomaly", False),
            "score": anomaly_report.get("anomaly_score"),
        }
    except Exception as e:
        summary["steps"]["anomaly"] = {"status": "failed", "error": str(e)}
        # anomaly failure is non-fatal
        anomaly_report = {"is_anomaly": False, "anomaly_score": 0.0, "top_contributors": []}

    # ── AI Layer: natural language report ─────────────────────────────
    try:
        bronze_meta = metas.get("bronze") or load_json(BRONZE_DIR / "bronze_meta.json")
        silver_meta = metas.get("silver") or load_json(SILVER_DIR / "silver_meta.json")
        gold_meta   = metas.get("gold")   or load_json(GOLD_DIR   / "gold_meta.json")

        report_text = _step("report", generate_health_report,
                            bronze_meta, silver_meta, gold_meta, anomaly_report)
        summary["steps"]["report"] = {"status": "ok", "path": str(REPORTS_DIR / "latest_report.md")}
    except Exception as e:
        summary["steps"]["report"] = {"status": "failed", "error": str(e)}

    # ── Final summary ──────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_start
    failed_steps  = [k for k, v in summary["steps"].items() if v["status"] == "failed"]
    summary["status"]        = "failed" if failed_steps else "success"
    summary["elapsed_sec"]   = round(total_elapsed, 2)
    summary["failed_steps"]  = failed_steps

    _save_pipeline_summary(summary, run_id)

    _banner(f"Pipeline {'SUCCEEDED' if not failed_steps else 'COMPLETED WITH ERRORS'} in {total_elapsed:.1f}s")
    if failed_steps:
        logger.warning(f"Failed steps: {failed_steps}")
    else:
        logger.success("All steps completed successfully! 🎉")
        logger.info(f"📄  Health report → {REPORTS_DIR}/latest_report.md")
        logger.info(f"📊  Gold tables   → {GOLD_DIR}/")

    return summary


def _save_pipeline_summary(summary: dict, run_id: str) -> None:
    save_json(summary, REPORTS_DIR / f"pipeline_run_{run_id}.json")
    save_json(summary, REPORTS_DIR / "latest_pipeline_run.json")


if __name__ == "__main__":
    fail_fast = "--no-fail-fast" not in sys.argv
    result    = run_pipeline(fail_fast=fail_fast)
    sys.exit(0 if result["status"] == "success" else 1)