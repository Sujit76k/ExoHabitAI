import os
import pandas as pd

def save_eda_summary(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("===== EDA SUMMARY =====")
    lines.append(f"Rows: {df.shape[0]}")
    lines.append(f"Columns: {df.shape[1]}")
    lines.append("\n--- Missing Values (Top 20) ---")
    lines.append(str(df.isna().sum().sort_values(ascending=False).head(20)))

    lines.append("\n--- Numeric Describe ---")
    lines.append(str(df.describe(include="number").T.head(20)))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
