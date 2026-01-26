import numpy as np
import pandas as pd

def find_first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_habitability_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    STRONG Milestone-1 Target:
    Uses wider & safer threshold ranges to avoid creating a single-class label.
    """
    df = df.copy()
    df["habitability"] = 0

    # Common NASA column names
    radius_candidates = ["pl_rade", "pl_radj"]
    temp_candidates = ["pl_eqt"]
    star_temp_candidates = ["st_teff"]

    r_col = find_first_existing_col(df, radius_candidates)
    t_col = find_first_existing_col(df, temp_candidates)
    st_col = find_first_existing_col(df, star_temp_candidates)

    # If key columns missing, fallback:
    # Create habitability based on available numeric columns with a simple quantile rule
    if r_col is None or t_col is None:
        numeric_df = df.select_dtypes(include=["number"]).copy()

        if numeric_df.shape[1] == 0:
            return df

        # ✅ Soft target generation using quantiles
        score = numeric_df.fillna(numeric_df.median()).sum(axis=1)
        threshold = score.quantile(0.85)  # top 15% as "habitable"
        df["habitability"] = np.where(score >= threshold, 1, 0)
        return df

    # ✅ Wider ranges (so it creates both 0 & 1)
    temp_ok = (df[t_col] >= 150) & (df[t_col] <= 400)
    radius_ok = (df[r_col] >= 0.5) & (df[r_col] <= 2.5)

    # Star temperature optional
    if st_col is not None:
        star_ok = (df[st_col] >= 3500) & (df[st_col] <= 8000)
    else:
        star_ok = True

    df["habitability"] = np.where(temp_ok & radius_ok & star_ok, 1, 0)
    return df
