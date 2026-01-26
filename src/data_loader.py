import pandas as pd
from src.config import RAW_DATA_PATH


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Correct loader for NASA Exoplanet Archive exports.
    These files contain metadata lines starting with '#'.
    We skip those and load the actual table.
    """

    df = pd.read_csv(
        path,
        comment="#",          # ✅ ignore NASA metadata lines
        sep=",",              # ✅ NASA exports are normally comma-separated
        engine="python",
        on_bad_lines="skip"
    )

    print("✅ Dataset loaded successfully (NASA format)")
    print("✅ Dataset shape:", df.shape)
    print("✅ First 10 columns:", list(df.columns[:10]))

    return df
