import pandas as pd

def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure all column names are unique.
    Example: ['a','a','b'] -> ['a','a_1','b']
    """
    df = df.copy()
    cols = pd.Series(df.columns)

    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i == 0:
                continue
            cols[idx] = f"{dup}_{i}"

    df.columns = cols
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    return df


def force_numeric_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all possible columns into numeric safely.
    """
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def select_numeric_features(df: pd.DataFrame, target_col: str = "habitability"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    df = fix_duplicate_columns(df)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Keep numeric only
    X = X.select_dtypes(include=["number"])
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError(
            "❌ No numeric features found after preprocessing.\n"
            "Dataset loaded but no usable numeric columns exist."
        )

    return X, y
