"""
Generate realistic synthetic data mirroring data.gov.in schemas.
Run once to populate data/raw/ — or call generate_all() from bronze.py.
"""
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from config import RAW_DIR, CROP_RECORDS, HOSPITAL_RECORDS
from utils import logger

STATES = [
    "Uttar Pradesh", "Maharashtra", "Punjab", "Haryana", "Madhya Pradesh",
    "Rajasthan", "Bihar", "Gujarat", "Andhra Pradesh", "Karnataka",
    "Tamil Nadu", "West Bengal", "Odisha", "Telangana", "Assam",
]
CROPS = [
    "Wheat", "Rice", "Maize", "Sugarcane", "Cotton",
    "Soybean", "Groundnut", "Mustard", "Barley", "Jowar",
]
SEASONS = ["Kharif", "Rabi", "Zaid"]
HOSPITAL_TYPES = ["Government", "Private", "Trust", "Charitable"]
SPECIALITIES = ["General", "Cardiology", "Orthopedics", "Pediatrics", "Oncology", "Neurology"]

rng = np.random.default_rng(42)


def _inject_nulls(df: pd.DataFrame, rate: float = 0.04) -> pd.DataFrame:
    """Randomly inject nulls to simulate real-world dirty data."""
    df = df.copy()
    for col in df.columns:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = None
    return df


def generate_crop_csv() -> Path:
    """Generate crop production dataset (mirrors data.gov.in Crop Production Stats)."""
    logger.info(f"Generating crop CSV ({CROP_RECORDS:,} rows)…")
    years = list(range(2010, 2024))
    rows = []
    for _ in range(CROP_RECORDS):
        year  = random.choice(years)
        state = random.choice(STATES)
        crop  = random.choice(CROPS)
        season = random.choice(SEASONS)
        area   = round(rng.uniform(100, 50_000), 2)
        yield_ = round(rng.uniform(500, 5_000), 2)
        production = round(area * yield_ / 1000, 2)
        # intentional duplicates (~2 %)
        rows.append({
            "state_name": state,
            "district_name": f"{state[:4]}_District_{random.randint(1, 20):02d}",
            "crop_year": year,
            "season": season,
            "crop": crop,
            "area_hectares": area,
            "yield_kg_per_ha": yield_,
            "production_tonnes": production,
        })

    df = pd.DataFrame(rows)
    # add ~2 % exact duplicates
    dupe_idx = rng.choice(len(df), size=int(CROP_RECORDS * 0.02), replace=False)
    df = pd.concat([df, df.iloc[dupe_idx]], ignore_index=True)
    df = _inject_nulls(df, rate=0.04)

    path = RAW_DIR / "crop_production.csv"
    df.to_csv(path, index=False)
    logger.success(f"Crop CSV saved → {path} ({len(df):,} rows)")
    return path


def generate_hospital_json() -> Path:
    """Generate hospital infrastructure dataset (mirrors data.gov.in Health Infra)."""
    logger.info(f"Generating hospital JSON ({HOSPITAL_RECORDS:,} records)…")
    records = []
    for i in range(HOSPITAL_RECORDS):
        state = random.choice(STATES)
        records.append({
            "hospital_id": f"HOSP{i+1:05d}",
            "hospital_name": f"{state[:3].upper()} {random.choice(SPECIALITIES)} Hospital {i+1}",
            "state": state,
            "district": f"{state[:4]}_District_{random.randint(1, 20):02d}",
            "hospital_type": random.choice(HOSPITAL_TYPES),
            "speciality": random.choice(SPECIALITIES),
            "beds_total": int(rng.integers(10, 800)),
            "beds_available": None if rng.random() < 0.05 else int(rng.integers(0, 800)),
            "doctors_count": int(rng.integers(1, 200)) if rng.random() > 0.03 else None,
            "nurses_count": int(rng.integers(2, 500)),
            "established_year": int(rng.integers(1950, 2023)),
            "accredited": bool(rng.random() > 0.4),
            "latitude": round(float(rng.uniform(8.0, 35.0)), 4),
            "longitude": round(float(rng.uniform(68.0, 97.0)), 4),
            "last_updated": (datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 365)))).isoformat(),
        })

    path = RAW_DIR / "hospital_infrastructure.json"
    path.write_text(json.dumps(records, indent=2))
    logger.success(f"Hospital JSON saved → {path} ({len(records):,} records)")
    return path


def generate_all() :
    """Generate all synthetic raw datasets. Returns dict of {name: path}."""
    return {
        "crop_csv":      generate_crop_csv(),
        "hospital_json": generate_hospital_json(),
    }


if __name__ == "__main__":
    generate_all()
    logger.info("Raw data generation complete.")