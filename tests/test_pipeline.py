"""
SmartIngest — Pytest Test Suite
Run: pytest tests/ -v
"""
import sys
import json
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, REPORTS_DIR


# ── Data generation tests ──────────────────────────────────────────────────

class TestDataGenerator:
    def test_generate_crop_csv(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "RAW_DIR", tmp_path)
        monkeypatch.setattr(config, "CROP_RECORDS", 100)

        from data_generator import generate_crop_csv
        path = generate_crop_csv()
        df = pd.read_csv(path)
        assert len(df) >= 100
        assert "state_name" in df.columns
        assert "production_tonnes" in df.columns

    def test_generate_hospital_json(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "RAW_DIR", tmp_path)
        monkeypatch.setattr(config, "HOSPITAL_RECORDS", 50)

        from data_generator import generate_hospital_json
        path = generate_hospital_json()
        data = json.loads(path.read_text())
        assert len(data) == 50
        assert "hospital_id" in data[0]
        assert "beds_total" in data[0]


# ── Silver cleaning tests ──────────────────────────────────────────────────

class TestSilverCleaning:

    @pytest.fixture
    def sample_crop_df(self):
        return pd.DataFrame({
            "state_name":        ["Punjab", "Haryana", None, "Bihar"],
            "district_name":     ["D1", "D2", "D3", "D4"],
            "crop_year":         [2020, 2021, 2022, 2023],
            "season":            ["Rabi", "Kharif", "InvalidSeason", "Zaid"],
            "crop":              ["Wheat", "Rice", None, "Maize"],
            "area_hectares":     [1000.0, 2000.0, 500.0, None],
            "yield_kg_per_ha":   [3000.0, 2500.0, 1800.0, 2200.0],
            "production_tonnes": [3000.0, 5000.0, 900.0, None],
        })

    def test_clean_crop_drops_null_key_rows(self, sample_crop_df):
        from silver import clean_crop
        df_out, report = clean_crop(sample_crop_df)
        # row with null crop should be dropped
        assert df_out["crop"].isna().sum() == 0

    def test_clean_crop_invalidates_bad_season(self, sample_crop_df):
        from silver import clean_crop
        df_out, report = clean_crop(sample_crop_df)
        # "InvalidSeason" row should have season set to NA (or dropped)
        valid_seasons = {"Rabi", "Kharif", "Zaid", None, float("nan")}
        for val in df_out["season"].dropna().unique():
            assert val in {"Rabi", "Kharif", "Zaid"}, f"Invalid season: {val}"

    def test_clean_crop_report_structure(self, sample_crop_df):
        from silver import clean_crop
        _, report = clean_crop(sample_crop_df)
        assert "rows_in" in report
        assert "rows_out" in report
        assert "drop_rate" in report
        assert "null_rates" in report
        assert "flags" in report
        assert report["rows_in"] > 0

    @pytest.fixture
    def sample_hospital_df(self):
        return pd.DataFrame({
            "hospital_id":      ["H001", "H002", "H001", None],  # dup + null id
            "hospital_name":    ["A", "B", "A_dup", "C"],
            "state":            ["Punjab", "Haryana", "Punjab", None],
            "district":         ["D1", "D2", "D1", "D3"],
            "hospital_type":    ["Government", "Private", "Government", "Trust"],
            "speciality":       ["General", "Cardiology", "General", "Pediatrics"],
            "beds_total":       [100, 200, 100, 50],
            "beds_available":   [80, 250, 80, 30],  # H002 exceeds total
            "doctors_count":    [10, 20, 10, 5],
            "nurses_count":     [30, 60, 30, 15],
            "established_year": [2000, 2010, 2000, 1995],
            "accredited":       [True, False, True, True],
            "latitude":         [30.0, 28.0, 30.0, 0.0],   # last invalid
            "longitude":        [75.0, 77.0, 75.0, 200.0], # last invalid
            "last_updated":     ["2023-01-01", "2023-06-15", "2023-01-01", "2022-12-01"],
        })

    def test_clean_hospital_deduplicates(self, sample_hospital_df):
        from silver import clean_hospital
        df_out, _ = clean_hospital(sample_hospital_df)
        assert df_out["hospital_id"].duplicated().sum() == 0

    def test_clean_hospital_fixes_bed_overflow(self, sample_hospital_df):
        from silver import clean_hospital
        df_out, _ = clean_hospital(sample_hospital_df)
        valid = df_out.dropna(subset=["beds_available", "beds_total"])
        assert (valid["beds_available"] <= valid["beds_total"]).all()

    def test_clean_weather(self):
        from silver import clean_weather
        df = pd.DataFrame({
            "date":              pd.date_range("2024-01-01", periods=10),
            "temp_max_c":        [20.0] * 10,
            "temp_min_c":        [25.0] * 10,  # inversion — max < min
            "precipitation_mm":  [5.0] * 10,
            "windspeed_max_kph": [15.0] * 10,
        })
        df_out, report = clean_weather(df)
        valid = df_out.dropna(subset=["temp_max_c", "temp_min_c"])
        assert (valid["temp_max_c"] >= valid["temp_min_c"]).all(), "Temp inversion not fixed"


# ── Quality monitor tests ──────────────────────────────────────────────────

class TestQualityMonitor:

    def _make_silver_meta(self, rows_in=1000, rows_out=950, flags=None):
        return {
            "run_ts": "2024-01-01T00-00-00",
            "reports": {
                src: {
                    "rows_in":    rows_in,
                    "rows_out":   rows_out,
                    "drop_rate":  (rows_in - rows_out) / rows_in,
                    "flag_count": len(flags or []),
                    "flags":      flags or [],
                    "null_rates": {"col_a": 0.02, "col_b": 0.01},
                }
                for src in ["crop", "hospital", "weather"]
            },
        }

    def test_extract_metrics_keys(self):
        from quality_monitor import extract_silver_metrics
        meta    = self._make_silver_meta()
        metrics = extract_silver_metrics(meta)
        assert "crop_rows_in"       in metrics
        assert "hospital_drop_rate" in metrics
        assert "weather_avg_null_rate" in metrics

    def test_anomaly_detection_returns_report(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
        (tmp_path).mkdir(exist_ok=True)

        from quality_monitor import run_anomaly_detection
        meta   = self._make_silver_meta()
        report = run_anomaly_detection(meta)
        assert "is_anomaly" in report
        assert "anomaly_score" in report
        assert isinstance(report["is_anomaly"], bool)


# ── API smoke tests ────────────────────────────────────────────────────────

class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api import app
        return TestClient(app)

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "endpoints" in r.json()

    def test_kpi_404_when_no_data(self, client, monkeypatch):
        import config, tempfile
        monkeypatch.setattr(config, "GOLD_DIR", Path(tempfile.mkdtemp()))
        r = client.get("/gold/kpi")
        assert r.status_code == 404