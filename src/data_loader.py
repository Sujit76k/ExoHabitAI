"""
=====================================================
🚀 ExoHabitAI — Production Data Loader
Robust NASA Exoplanet Archive CSV reader
=====================================================
"""

import os
import logging
import pandas as pd

from src.config import RAW_DATA_PATH

# -----------------------------------------------------
# LOGGER (production style)
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------

def _detect_delimiter(sample_text: str) -> str:
    """
    Auto-detect delimiter used in dataset.
    NASA files are usually comma-separated,
    but some exports use | or ;.
    """
    import csv
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", "|", ";", "\t"])
        return dialect.delimiter
    except Exception:
        return ","


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names:
    - remove spaces
    - lowercase
    - remove hidden characters
    """
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
        .str.lower()
    )
    return df


# -----------------------------------------------------
# MAIN LOADER
# -----------------------------------------------------

def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load NASA Exoplanet dataset safely.

    Features:
    ✔ skips metadata (# lines)
    ✔ detects delimiter automatically
    ✔ cleans column names
    ✔ handles encoding issues
    ✔ production-safe logging
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Dataset not found at: {path}")

    logger.info("📂 Loading raw dataset...")

    # Read small sample for delimiter detection
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(5000)

    delimiter = _detect_delimiter(sample)

    logger.info(f"🔎 Detected delimiter: '{delimiter}'")

    # Load dataset
    df = pd.read_csv(
        path,
        comment="#",         
        sep=delimiter,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
    )

    # Clean column names
    df = _clean_column_names(df)

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    logger.info(f"✅ Dataset loaded successfully")
    logger.info(f"📊 Shape: {df.shape}")
    logger.info(f"🧪 First columns: {list(df.columns[:10])}")

    return df