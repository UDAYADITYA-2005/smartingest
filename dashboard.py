"""
SmartIngest — Streamlit Monitoring Dashboard
Run: streamlit run dashboard.py
"""
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR

st.set_page_config(
    page_title="SmartIngest | Pipeline Monitor",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────

def load_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_parquet_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.DataFrame()


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/data-configuration.png", width=64)
    st.title("SmartIngest")
    st.caption("AI-Augmented Medallion Pipeline")
    st.divider()

    if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running pipeline…"):
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "pipeline.py"],
                capture_output=True, text=True, cwd=Path(__file__).parent
            )
            if result.returncode == 0:
                st.success("Pipeline completed!")
            else:
                st.error("Pipeline failed. Check logs.")
                st.code(result.stderr[-2000:])
        st.rerun()

    st.divider()
    st.markdown("**Data paths**")
    st.code(f"Bronze: {BRONZE_DIR}\nSilver: {SILVER_DIR}\nGold:   {GOLD_DIR}", language="text")


# ── Load metadata ─────────────────────────────────────────────────────────

bronze_meta  = load_json_safe(BRONZE_DIR / "bronze_meta.json")
silver_meta  = load_json_safe(SILVER_DIR / "silver_meta.json")
gold_meta    = load_json_safe(GOLD_DIR   / "gold_meta.json")
anomaly_rpt  = load_json_safe(REPORTS_DIR / "latest_anomaly_report.json")
pipeline_run = load_json_safe(REPORTS_DIR / "latest_pipeline_run.json")

no_data = not bronze_meta

# ── Title ─────────────────────────────────────────────────────────────────

st.title("🔁 SmartIngest Pipeline Monitor")
if no_data:
    st.warning("No pipeline run found. Click **▶ Run Full Pipeline** in the sidebar to get started.")
    st.stop()

run_ts_str = pipeline_run.get("run_id", "N/A")
status     = pipeline_run.get("status", "unknown")
elapsed    = pipeline_run.get("elapsed_sec", 0)
col_status, col_time, col_elapsed = st.columns(3)
col_status.metric("Pipeline Status", "✅ Success" if status == "success" else "❌ Failed")
col_time.metric("Last Run", run_ts_str.replace("T", " ").replace("-", "/")[:16] if "T" in run_ts_str else run_ts_str)
col_elapsed.metric("Duration", f"{elapsed}s")

st.divider()

# ── Row count summary ─────────────────────────────────────────────────────

st.subheader("📊 Row Counts per Layer")
sources = list(bronze_meta.get("sources", {}).keys())
bronze_counts = {s: bronze_meta["sources"][s].get("row_count", 0) for s in sources}

silver_reports = silver_meta.get("reports", {})
silver_counts  = {s: silver_reports.get(s, {}).get("rows_out", 0) for s in ["crop", "hospital", "weather"]}

gold_tables = gold_meta.get("tables", {})

row_data = []
for src, count in bronze_counts.items():
    row_data.append({"layer": "Bronze", "source": src, "rows": count})
for src, count in silver_counts.items():
    row_data.append({"layer": "Silver", "source": src, "rows": count})

if row_data:
    df_rows = pd.DataFrame(row_data)
    fig_rows = px.bar(
        df_rows, x="source", y="rows", color="layer",
        barmode="group",
        color_discrete_map={"Bronze": "#CD7F32", "Silver": "#C0C0C0"},
        title="Row Counts: Bronze vs Silver (after cleaning)",
        height=350,
    )
    st.plotly_chart(fig_rows, use_container_width=True)

# ── Gold table sizes ──────────────────────────────────────────────────────

if gold_tables:
    st.subheader("🥇 Gold Table Sizes")
    df_gold = pd.DataFrame({"table": list(gold_tables.keys()), "rows": list(gold_tables.values())})
    fig_gold = px.bar(df_gold, x="table", y="rows", color_discrete_sequence=["#FFD700"],
                      title="Gold Layer — Rows per Table", height=300)
    fig_gold.update_xaxes(tickangle=30)
    st.plotly_chart(fig_gold, use_container_width=True)

st.divider()

# ── Null rates heatmap ────────────────────────────────────────────────────

st.subheader("🔍 Silver Layer Null Rates")
null_tabs = st.tabs(["Crop", "Hospital", "Weather"])
for tab, src in zip(null_tabs, ["crop", "hospital", "weather"]):
    with tab:
        null_rates = silver_reports.get(src, {}).get("null_rates", {})
        if null_rates:
            df_null = pd.DataFrame(
                {"column": list(null_rates.keys()), "null_rate": list(null_rates.values())}
            ).sort_values("null_rate", ascending=False)
            df_null["null_pct"] = (df_null["null_rate"] * 100).round(2)
            fig_null = px.bar(
                df_null.head(15), x="column", y="null_pct",
                title=f"{src.capitalize()} — Top null % by column",
                color="null_pct",
                color_continuous_scale=["green", "yellow", "red"],
                height=300,
            )
            fig_null.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_null, use_container_width=True)
            # Flag columns
            flags = silver_reports.get(src, {}).get("flags", [])
            if flags:
                for f in flags:
                    st.warning(f"⚑  {f}")
            else:
                st.success("No quality flags on this source.")

st.divider()

# ── Anomaly detection ─────────────────────────────────────────────────────

st.subheader("🤖 ML Anomaly Detection (Isolation Forest)")
c1, c2, c3 = st.columns(3)
is_anomaly    = anomaly_rpt.get("is_anomaly", False)
anomaly_score = anomaly_rpt.get("anomaly_score", 0)
history_runs  = anomaly_rpt.get("history_runs", 0)

c1.metric("Anomaly Detected", "⚠️ YES" if is_anomaly else "✅ NO")
c2.metric("Anomaly Score", f"{anomaly_score:.4f}", help="More negative = more anomalous")
c3.metric("Historical Runs", history_runs)

top_contributors = anomaly_rpt.get("top_contributors", [])
if top_contributors:
    df_contrib = pd.DataFrame(top_contributors)
    fig_contrib = px.bar(
        df_contrib, x="feature", y="z_score",
        title="Top Anomaly Contributors (Z-Score deviation from mean)",
        color="z_score", color_continuous_scale=["green", "orange", "red"],
        height=280,
    )
    fig_contrib.update_xaxes(tickangle=30)
    fig_contrib.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_contrib, use_container_width=True)

st.divider()

# ── Weather chart ─────────────────────────────────────────────────────────

st.subheader("🌤 Weather — Rolling Averages")
df_weather = load_parquet_safe(GOLD_DIR / "weather_rolling_avg.parquet")
if not df_weather.empty and "date" in df_weather.columns:
    df_weather["date"] = pd.to_datetime(df_weather["date"])
    fig_w = go.Figure()
    fig_w.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_max_7d_avg"],
                               name="Temp Max 7d avg", line=dict(color="tomato")))
    fig_w.add_trace(go.Scatter(x=df_weather["date"], y=df_weather["temp_min_7d_avg"],
                               name="Temp Min 7d avg", line=dict(color="steelblue")))
    fig_w.add_trace(go.Bar(x=df_weather["date"], y=df_weather["precip_7d_sum"],
                           name="Precip 7d sum (mm)", yaxis="y2", opacity=0.4,
                           marker_color="cornflowerblue"))
    fig_w.update_layout(
        title="Temperature & Precipitation — 7-Day Rolling Window",
        yaxis=dict(title="Temperature (°C)"),
        yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right"),
        height=380, legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_w, use_container_width=True)

st.divider()

# ── KPI summary ───────────────────────────────────────────────────────────

st.subheader("📈 Key Performance Indicators")
df_kpi = load_parquet_safe(GOLD_DIR / "kpi_summary.parquet")
if not df_kpi.empty:
    k = df_kpi.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Crop Records",        f"{int(k.get('crop_records', 0)):,}")
    cols[1].metric("Total Production",    f"{k.get('total_production_mt', 0):,} MT")
    cols[2].metric("Hospital Records",    f"{int(k.get('hospital_records', 0)):,}")
    cols[3].metric("Total Hospital Beds", f"{int(k.get('total_beds', 0)):,}")

st.divider()

# ── AI Health Report ──────────────────────────────────────────────────────

st.subheader("📄 AI-Generated Health Report")
report_path = REPORTS_DIR / "latest_report.md"
if report_path.exists():
    report_text = report_path.read_text()
    with st.expander("View full report", expanded=True):
        st.markdown(report_text)
else:
    st.info("No report generated yet. Run the pipeline to generate one.")

# ── Footer ────────────────────────────────────────────────────────────────

st.divider()
st.caption("SmartIngest — AI-Augmented Medallion Pipeline | Built with 🐍 Python + DuckDB + scikit-learn + Streamlit")