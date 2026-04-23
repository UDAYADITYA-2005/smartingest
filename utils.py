"""Shared utilities: logging, timing, Parquet I/O."""
import time
import json
from datetime import datetime
from pathlib import Path
from functools import wraps
from loguru import logger
import pandas as pd

from config import LOGS_DIR

# ── Logger setup ──────────────────────────────────────────────────────────
logger.remove()
logger.add(
    LOGS_DIR / "pipeline_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
)
logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


def timed(fn):
    """Decorator that logs execution time of a function."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info(f"{fn.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


def save_parquet(df: pd.DataFrame, path: Path, layer: str) -> None:
    """Save DataFrame as Parquet with layer metadata."""
    df["_pipeline_layer"]    = layer
    df["_pipeline_ts"]       = datetime.utcnow().isoformat()
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.debug(f"Saved {len(df):,} rows → {path}")


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file and return a DataFrame."""
    df = pd.read_parquet(path, engine="pyarrow")
    logger.debug(f"Loaded {len(df):,} rows ← {path}")
    return df


def save_json(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.debug(f"Saved JSON → {path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def run_ts() -> str:
    """Return current UTC timestamp string suitable for filenames."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")