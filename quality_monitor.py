"""
Quality Monitor — ML Anomaly Detection
Uses scikit-learn Isolation Forest to detect anomalous pipeline runs
based on quality metrics (null rates, row counts, drop rates).
Persists a metrics history so anomalies become meaningful over time.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import SILVER_DIR, GOLD_DIR, REPORTS_DIR, ANOMALY_CONTAMINATION
from utils import logger, timed, save_json, load_json, run_ts

METRICS_HISTORY_PATH = REPORTS_DIR / "metrics_history.json"
ANOMALY_REPORT_PATH  = REPORTS_DIR / "latest_anomaly_report.json"


# ── Metrics extraction ────────────────────────────────────────────────────

def extract_silver_metrics(silver_meta: dict) -> dict:
    """Flatten silver quality reports into a feature vector."""
    metrics = {"run_ts": silver_meta.get("run_ts", run_ts())}

    for source in ["crop", "hospital", "weather"]:
        rpt = silver_meta.get("reports", {}).get(source, {})
        metrics[f"{source}_rows_in"]    = rpt.get("rows_in", 0)
        metrics[f"{source}_rows_out"]   = rpt.get("rows_out", 0)
        metrics[f"{source}_drop_rate"]  = rpt.get("drop_rate", 0.0)
        metrics[f"{source}_flag_count"] = len(rpt.get("flags", []))

        # Average null rate across all columns
        null_rates = rpt.get("null_rates", {})
        null_vals  = [v for v in null_rates.values() if isinstance(v, (int, float))]
        metrics[f"{source}_avg_null_rate"] = round(np.mean(null_vals), 4) if null_vals else 0.0
        metrics[f"{source}_max_null_rate"] = round(max(null_vals), 4) if null_vals else 0.0

    return metrics


def _load_history() :
    try:
        if METRICS_HISTORY_PATH.exists():
            return load_json(METRICS_HISTORY_PATH)
    except Exception:
        pass
    return []


def _save_history(history: list[dict]) -> None:
    save_json(history, METRICS_HISTORY_PATH)


def _feature_cols() -> list[str]:
    sources = ["crop", "hospital", "weather"]
    return [
        f"{s}_{m}"
        for s in sources
        for m in ["rows_in", "rows_out", "drop_rate", "flag_count", "avg_null_rate", "max_null_rate"]
    ]


# ── Isolation Forest ──────────────────────────────────────────────────────

@timed
def run_anomaly_detection(silver_meta: dict) -> dict:
    """
    Extract metrics, append to history, train Isolation Forest,
    score the latest run.  Returns anomaly report dict.
    """
    logger.info("Running ML anomaly detection (Isolation Forest)…")

    current_metrics = extract_silver_metrics(silver_meta)
    history = _load_history()
    history.append(current_metrics)
    _save_history(history)

    feature_cols = _feature_cols()
    df_hist = pd.DataFrame(history)

    # Fill any missing columns with 0
    for col in feature_cols:
        if col not in df_hist.columns:
            df_hist[col] = 0.0

    df_hist[feature_cols] = df_hist[feature_cols].fillna(0).astype(float)
    X = df_hist[feature_cols].values

    # Need at least 3 samples for Isolation Forest to be meaningful
    if len(X) < 3:
        logger.info("Not enough history for Isolation Forest (< 3 runs). Padding with synthetic baseline.")
        # Create synthetic 'normal' baseline runs for cold-start
        baseline = np.tile(X[0], (max(3, 5 - len(X)), 1))
        noise    = np.random.default_rng(42).normal(0, 0.01, baseline.shape)
        X_train  = np.vstack([baseline + noise, X])
    else:
        X_train = X

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train Isolation Forest
    clf = IsolationForest(
        n_estimators=100,
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled)

    # Score latest run (last row of X_train corresponds to current run after padding or last row of X)
    latest_scaled  = scaler.transform(X[-1].reshape(1, -1))
    anomaly_score  = clf.score_samples(latest_scaled)[0]   # negative: lower = more anomalous
    is_anomaly     = clf.predict(latest_scaled)[0] == -1    # -1 = anomaly

    # Per-feature contribution (deviation from mean)
    X_mean  = X_train[:-1].mean(axis=0) if len(X_train) > 1 else X_train[0]
    X_std   = X_train[:-1].std(axis=0)  + 1e-9
    z_scores = np.abs((X[-1] - X_mean) / X_std)
    top_contributors = sorted(
        zip(feature_cols, z_scores.tolist()),
        key=lambda x: x[1], reverse=True
    )[:5]

    report = {
        "run_ts":            current_metrics["run_ts"],
        "is_anomaly":        bool(is_anomaly),
        "anomaly_score":     round(float(anomaly_score), 4),
        "history_runs":      len(history),
        "top_contributors":  [{"feature": f, "z_score": round(z, 3)} for f, z in top_contributors],
        "current_metrics":   {k: v for k, v in current_metrics.items() if k != "run_ts"},
        "threshold_note":    f"Isolation Forest contamination={ANOMALY_CONTAMINATION}",
    }

    save_json(report, ANOMALY_REPORT_PATH)

    level = "WARNING" if is_anomaly else "SUCCESS"
    msg   = "ANOMALY DETECTED" if is_anomaly else "No anomalies detected"
    logger.log(level, f"Anomaly detection: {msg} (score={anomaly_score:.4f})")
    return report


if __name__ == "__main__":
    # Quick smoke-test: load existing silver_meta or create stub
    meta_path = SILVER_DIR / "silver_meta.json"
    if meta_path.exists():
        silver_meta = load_json(meta_path)
        report = run_anomaly_detection(silver_meta)
        print(json.dumps(report, indent=2))
    else:
        print("Run silver.py first to generate silver_meta.json")