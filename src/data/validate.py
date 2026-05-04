import pandas as pd

def validate(df: pd.DataFrame) -> None:
    """
    Print a comprehensive data validation report.

    Covers:
        - Shape (rows, columns)
        - Data types per column
        - Missing values count and percentage
        - Duplicate rows
        - Class distribution for target variable
        - outliers in numeric columns (using IQR method & isolation forest)
    """
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n── Data Types ──────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values ──────────────────────────────────────────")
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_df  = missing_df[missing_df["count"] > 0]
    if missing_df.empty:
        print("No missing values found.")
    else:
        print(missing_df.to_string())

    print(f"\n── Duplicates ──────────────────────────────────────────────")
    dup_count = df.duplicated(subset=["event_id"]).sum() if "event_id" in df.columns else "N/A"
    print(f"Duplicate event_id rows: {dup_count}")

    print("\n── Numeric Summary ─────────────────────────────────────────")
    # white_elo included here for context only — not a model input feature
    numeric_cols = [c for c in ["white_elo", "black_elo", "num_moves", "white_acl",
                    "black_acl", "white_blunders", "black_blunders",
                    "acl_gap", "game_sharpness"] if c in df.columns]
    if numeric_cols:
        print(df[numeric_cols].describe().round(2).to_string())

        print("\n── Outliers (IQR Method) ───────────────────────────────────")
        outliers_dict = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_dict[col] = outlier_mask.sum()
        
        outliers_df = pd.DataFrame(list(outliers_dict.items()), columns=["Column", "Outliers count"])
        print(outliers_df.to_string())
    print("\n── Class Distribution for Target Variable ──────────────────────────────")
    if "winner_multiclass" in df.columns:
        class_dist = df["winner_multiclass"].value_counts(normalize=True).round(4) * 100
        print("winner_multiclass distribution (%):")
        print(class_dist.to_string())
    elif "winner_binary" in df.columns:
        class_dist = df["winner_binary"].value_counts(normalize=True).round(4) * 100
        print("winner_binary distribution (%):")
        print(class_dist.to_string())
    else:
        print("No target variable found for class distribution.")
