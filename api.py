"""
SmartIngest FastAPI — Data Serving Endpoints
Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import GOLD_DIR, REPORTS_DIR, SILVER_DIR, BRONZE_DIR, PIPELINE_VERSION, BASE_DIR
from utils import load_json

app = FastAPI(
    title="SmartIngest Data API",
    description="Serve Gold layer analytics and pipeline health metrics.",
    version=PIPELINE_VERSION,
    docs_url="/docs",
)


# CORS + frontend serving (added for standalone HTML frontend)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles as _SF

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

_FRONTEND = BASE_DIR / 'frontend'
if _FRONTEND.exists():
    app.mount('/ui', _SF(directory=str(_FRONTEND), html=True), name='frontend')

@app.get('/dashboard', include_in_schema=False)
def serve_dashboard():
    idx = _FRONTEND / 'index.html'
    if idx.exists():
        return FileResponse(str(idx))
    raise HTTPException(404, 'Frontend not found.')


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_gold(name: str) -> pd.DataFrame:
    path = GOLD_DIR / f"{name}.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Gold table '{name}' not found. Run the pipeline first.")
    return pd.read_parquet(path, engine="pyarrow")


def _df_to_records(df: pd.DataFrame, limit: int = 500) -> list[dict]:
    return df.head(limit).fillna("").astype(str).to_dict(orient="records")


# ── Root ──────────────────────────────────────────────────────────────────

# ── Serve frontend ────────────────────────────────────────────────────────
_frontend_dir = BASE_DIR / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

@app.get("/ui", tags=["frontend"], include_in_schema=False)
def serve_frontend():
    """Serve the standalone HTML dashboard frontend."""
    path = _frontend_dir / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(str(path))


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "SmartIngest Data API",
        "version": PIPELINE_VERSION,
        "docs":    "/docs",
        "ui":      "/ui",
        "endpoints": [
            "/gold/kpi",
            "/gold/crop-summary",
            "/gold/top-producers",
            "/gold/hospital-summary",
            "/gold/weather",
            "/gold/crop-weather",
            "/health/latest",
            "/health/anomaly",
            "/pipeline/status",
        ],
    }


# ── Gold endpoints ────────────────────────────────────────────────────────

@app.get("/gold/kpi", tags=["gold"])
def get_kpi():
    """Single-row KPI snapshot from the latest pipeline run."""
    df = _load_gold("kpi_summary")
    return df.fillna(0).to_dict(orient="records")[0]


@app.get("/gold/crop-summary", tags=["gold"])
def get_crop_summary(
    state:  Optional[str] = Query(None, description="Filter by state name"),
    limit:  int        = Query(100,  ge=1, le=5000),
):
    """State-level crop production summary."""
    df = _load_gold("crop_state_summary")
    if state:
        df = df[df["state_name"].str.lower() == state.lower()]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    return {"count": len(df), "data": _df_to_records(df, limit)}


@app.get("/gold/top-producers", tags=["gold"])
def get_top_producers(
    crop: Optional[str] = Query(None, description="Filter by crop name"),
    limit: int       = Query(50,   ge=1, le=500),
):
    """Top 5 producing states per crop."""
    df = _load_gold("crop_top_producers")
    if crop:
        df = df[df["crop"].str.lower() == crop.lower()]
    return {"count": len(df), "data": _df_to_records(df, limit)}


@app.get("/gold/hospital-summary", tags=["gold"])
def get_hospital_summary(
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int        = Query(200,  ge=1, le=5000),
):
    """State-level hospital infrastructure summary."""
    df = _load_gold("hospital_state_summary")
    if state:
        df = df[df["state"].str.lower() == state.lower()]
    return {"count": len(df), "data": _df_to_records(df, limit)}


@app.get("/gold/weather", tags=["gold"])
def get_weather(limit: int = Query(30, ge=1, le=365)):
    """Weather time series with 7-day rolling averages."""
    df = _load_gold("weather_rolling_avg")
    return {"count": len(df), "data": _df_to_records(df, limit)}


@app.get("/gold/crop-weather", tags=["gold"])
def get_crop_weather(
    crop:  Optional[str] = Query(None),
    limit: int        = Query(200, ge=1, le=5000),
):
    """Cross-domain crop production + weather join."""
    df = _load_gold("crop_weather_join")
    if crop:
        df = df[df["crop"].str.lower() == crop.lower()]
    return {"count": len(df), "data": _df_to_records(df, limit)}


# ── Health endpoints ──────────────────────────────────────────────────────

@app.get("/health/latest", tags=["health"])
def get_latest_health():
    """Latest AI-generated pipeline health report (Markdown text)."""
    path = REPORTS_DIR / "latest_report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No health report found. Run the pipeline first.")
    return {"report": path.read_text(), "generated_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat()}


@app.get("/health/anomaly", tags=["health"])
def get_anomaly_report():
    """Latest Isolation Forest anomaly detection report."""
    path = REPORTS_DIR / "latest_anomaly_report.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No anomaly report found.")
    return load_json(path)


# ── Pipeline status ───────────────────────────────────────────────────────

@app.get("/pipeline/status", tags=["pipeline"])
def get_pipeline_status():
    """Latest pipeline run status summary."""
    path = REPORTS_DIR / "latest_pipeline_run.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No pipeline run found.")
    return load_json(path)


@app.post("/pipeline/run", tags=["pipeline"])
def trigger_pipeline():
    """Trigger a new pipeline run (async — returns immediately)."""
    import subprocess, sys
    proc = subprocess.Popen(
        [sys.executable, "pipeline.py"],
        cwd=Path(__file__).parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {"message": "Pipeline triggered", "pid": proc.pid}