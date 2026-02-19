"""
=====================================================
🚀 ExoHabitAI — Machine Learning Pipeline Builder
Production-ready preprocessing + modeling pipeline
=====================================================
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


# -----------------------------------------------------
# NUMERIC PIPELINE
# -----------------------------------------------------

def build_numeric_pipeline():
    """
    Handles:
    - Missing values
    - Scaling
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return numeric_pipeline


# -----------------------------------------------------
# CATEGORICAL PIPELINE (Future-proof)
# -----------------------------------------------------

def build_categorical_pipeline():
    """
    Even if current model mostly numeric,
    we keep categorical pipeline ready for upgrades.
    """

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return categorical_pipeline


# -----------------------------------------------------
# MAIN PIPELINE BUILDER
# -----------------------------------------------------

def build_baseline_pipeline(numeric_cols, categorical_cols=None):
    """
    Production-level pipeline builder.

    Parameters
    ----------
    numeric_cols : list
        Numeric feature column names

    categorical_cols : list (optional)
        Categorical feature column names

    Returns
    -------
    sklearn Pipeline
    """

    if categorical_cols is None:
        categorical_cols = []

    # -----------------------------------------
    # Column Transformer (BEST PRACTICE)
    # -----------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", build_numeric_pipeline(), numeric_cols),
            ("cat", build_categorical_pipeline(), categorical_cols),
        ],
        remainder="drop",
    )

    # -----------------------------------------
    # FINAL MODEL
    # -----------------------------------------

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=None,
    )

    # -----------------------------------------
    # FULL PIPELINE
    # -----------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline