"""
data_loader.py - Data Loading and Cleaning Module
===================================================

This module handles all data ingestion, validation, and preprocessing 
for the EV vs Petrol vehicle analysis project. It provides reusable 
functions for loading raw CSV files, standardising column names, 
handling missing values, enforcing correct data types, and extracting 
temporal features (year, decade) for downstream analysis.

Industry Practices Applied:
    - Defensive loading with encoding fallback
    - Schema validation after ingestion
    - Logging instead of silent failures
    - Idempotent cleaning pipeline
    - Type-safe conversions with explicit error handling
"""

import pandas as pd
import numpy as np
import os
import logging

# Configure a module-level logger for traceability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given filepath with encoding fallback.

    Tries UTF-8 first, then falls back to latin-1 if a UnicodeDecodeError
    is raised. Logs the shape and columns of the loaded dataframe.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe as read from the CSV.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        logger.info(f"Loaded dataset with UTF-8 encoding from {filepath}")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
        logger.warning(f"UTF-8 failed; loaded with latin-1 encoding from {filepath}")

    logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 2. COLUMN NAME STANDARDISATION
# ---------------------------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names to snake_case with no trailing whitespace.

    Transformations applied:
        - Strip leading / trailing spaces
        - Convert to lowercase
        - Replace spaces and hyphens with underscores
        - Remove any remaining special characters (except underscores)
        - Collapse consecutive underscores

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with potentially messy column names.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column names.
    """
    original_cols = list(df.columns)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)     # spaces/hyphens -> _
        .str.replace(r"[^a-z0-9_]", "", regex=True)    # drop special chars
        .str.replace(r"_+", "_", regex=True)            # collapse duplicates
        .str.strip("_")                                 # trim edge underscores
    )

    renamed = {old: new for old, new in zip(original_cols, df.columns) if old != new}
    if renamed:
        logger.info(f"Renamed columns: {renamed}")
    else:
        logger.info("Column names already clean - no renaming needed.")

    return df


# ---------------------------------------------------------------------------
# 3. MISSING VALUE HANDLING
# ---------------------------------------------------------------------------

def inspect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of missing values per column.

    The summary includes the count, percentage, and data type for every
    column that has at least one null entry.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        A summary table sorted by missing percentage (descending).
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    dtypes = df.dtypes

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "dtype": dtypes
    })
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )

    if summary.empty:
        logger.info("No missing values detected in the dataset.")
    else:
        logger.warning(
            f"Found missing values in {len(summary)} column(s). "
            f"Total null cells: {int(missing_count.sum())}"
        )
    return summary


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values using a strategy suited to each column type.

    Strategy
    --------
    - **Numeric columns**: fill with the column median (robust to outliers).
    - **Categorical / object columns**: fill with the mode (most frequent value).
    - If a column is more than 60 % null it is dropped entirely as it
      carries insufficient information.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled.
    """
    initial_nulls = df.isnull().sum().sum()

    # Drop columns that are mostly empty (> 60 % missing)
    threshold = 0.60
    high_null_cols = [
        col for col in df.columns
        if df[col].isnull().mean() > threshold
    ]
    if high_null_cols:
        logger.warning(
            f"Dropping {len(high_null_cols)} column(s) with > {threshold*100:.0f}% "
            f"missing: {high_null_cols}"
        )
        df = df.drop(columns=high_null_cols)

    # Impute remaining nulls
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Filled {col} nulls with median = {median_val}")
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Filled {col} nulls with mode = '{mode_val}'")

    final_nulls = df.isnull().sum().sum()
    logger.info(
        f"Missing values reduced from {initial_nulls} -> {final_nulls}"
    )
    return df


# ---------------------------------------------------------------------------
# 4. DATA TYPE CONVERSION
# ---------------------------------------------------------------------------

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce correct pandas dtypes across all columns.

    Conversions
    -----------
    - year                        -> int (period)
    - Sales / count columns       -> int (whole units)
    - Percentages / rates / prices -> float
    - country, region, segment,
      powertrain_type             -> category (memory-efficient)
    - is_ev_dominant              -> bool

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with corrected data types.
    """
    # --- Integer columns (counts & discrete values) ---
    int_cols = [
        "year", "ev_sales", "petrol_car_sales", "diesel_car_sales",
        "total_vehicle_sales", "charging_stations"
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # --- Float columns (rates, prices, percentages) ---
    float_cols = [
        "ev_market_share", "fast_chargers_share", "avg_ev_range_km",
        "fuel_price_usd_per_liter", "electricity_price_usd_per_kwh",
        "gdp_per_capita", "urban_population_percent",
        "co2_emissions_transport_mt", "ev_subsidy_usd",
        "emission_regulation_score", "ev_growth_rate_yoy"
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # --- Categorical columns ---
    cat_cols = ["country", "region", "vehicle_segment", "powertrain_type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Boolean column ---
    if "is_ev_dominant" in df.columns:
        df["is_ev_dominant"] = df["is_ev_dominant"].astype(bool)

    logger.info("Data type conversions applied successfully.")
    return df


# ---------------------------------------------------------------------------
# 5. TEMPORAL FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive additional time-based columns from the 'year' column.

    New Columns
    -----------
    - **decade** : str - e.g. '2010s', '2020s'. Useful for grouping
      long-range trends.
    - **period** : str - Classifies each year into one of three phases:
          • 'Early (2010-2014)' - EV infancy
          • 'Growth (2015-2019)' - EV acceleration
          • 'Mainstream (2020-2025)' - Mass adoption era

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a numeric 'year' column.

    Returns
    -------
    pd.DataFrame
        Dataframe with the new temporal columns appended.
    """
    if "year" not in df.columns:
        logger.warning("'year' column not found - skipping temporal extraction.")
        return df

    # Decade grouping
    df["decade"] = (df["year"] // 10 * 10).astype(str) + "s"

    # Business-phase classification
    conditions = [
        df["year"].between(2010, 2014),
        df["year"].between(2015, 2019),
        df["year"].between(2020, 2025),
    ]
    labels = ["Early (2010-2014)", "Growth (2015-2019)", "Mainstream (2020-2025)"]
    df["period"] = np.select(conditions, labels, default="Other")

    logger.info("Temporal features (decade, period) extracted.")
    return df


# ---------------------------------------------------------------------------
# 6. SCHEMA VALIDATION
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> bool:
    """
    Check that the dataframe contains the minimum expected columns.

    Returns True if all required columns are present, otherwise logs
    the missing ones and returns False.
    """
    required = {
        "country", "region", "year", "vehicle_segment", "powertrain_type",
        "ev_sales", "petrol_car_sales", "diesel_car_sales",
        "total_vehicle_sales", "ev_market_share"
    }
    present = set(df.columns)
    missing = required - present

    if missing:
        logger.error(f"Schema validation FAILED - missing columns: {missing}")
        return False

    logger.info("Schema validation passed - all required columns present.")
    return True


# ---------------------------------------------------------------------------
# 7. DATA QUALITY REPORT
# ---------------------------------------------------------------------------

def generate_quality_report(df: pd.DataFrame) -> dict:
    """
    Produce a concise data-quality snapshot of the cleaned dataframe.

    Returns a dictionary with:
        - total_rows, total_columns
        - missing_cells (should be 0 after cleaning)
        - duplicate_rows
        - numeric_summary  (describe() output)
        - categorical_summary (value counts for key categorical columns)
        - year_range
        - countries_covered
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "year_range": (
            int(df["year"].min()) if "year" in df.columns else None,
            int(df["year"].max()) if "year" in df.columns else None
        ),
        "countries_covered": (
            sorted(df["country"].unique().tolist())
            if "country" in df.columns else []
        ),
        "numeric_summary": df.describe().round(2),
        "categorical_summary": {
            col: df[col].value_counts().to_dict()
            for col in ["country", "region", "vehicle_segment", "powertrain_type"]
            if col in df.columns
        }
    }

    logger.info(
        f"Quality report - {report['total_rows']} rows, "
        f"{report['total_columns']} cols, "
        f"{report['missing_cells']} nulls, "
        f"{report['duplicate_rows']} duplicates."
    )
    return report


# ---------------------------------------------------------------------------
# 8. FULL CLEANING PIPELINE
# ---------------------------------------------------------------------------

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    End-to-end pipeline: load -> clean names -> handle nulls -> convert types
    -> extract temporal features -> validate.

    This is the single entry-point that downstream modules (analysis,
    visualisation, database) should call to obtain a ready-to-use dataframe.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Fully cleaned and enriched dataframe.
    """
    logger.info("=" * 60)
    logger.info("STARTING DATA LOADING & CLEANING PIPELINE")
    logger.info("=" * 60)

    # Step 1 - Load raw data
    df = load_dataset(filepath)

    # Step 2 - Standardise column names
    df = clean_column_names(df)

    # Step 3 - Inspect and handle missing values
    missing_report = inspect_missing_values(df)
    if not missing_report.empty:
        print("\nMissing Value Report (before cleaning):")
        print(missing_report.to_string())
    df = handle_missing_values(df)

    # Step 4 - Enforce data types
    df = convert_data_types(df)

    # Step 5 - Extract temporal features
    df = extract_temporal_features(df)

    # Step 6 - Validate schema
    is_valid = validate_schema(df)
    if not is_valid:
        logger.warning("Pipeline completed WITH schema warnings.")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE - data is clean and ready for analysis.")
    logger.info("=" * 60)

    return df


# ---------------------------------------------------------------------------
# Standalone execution for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Default path when run from the project root
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "ev_vs_petrol_dataset_v3.csv"

    df_clean = load_and_clean(csv_path)

    print(f"\nCleaned dataset: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
    print(f"Year range: {df_clean['year'].min()} - {df_clean['year'].max()}")
    print(f"Countries: {df_clean['country'].nunique()}")
    print(f"\nColumn dtypes:\n{df_clean.dtypes}")
    print(f"\nFirst 5 rows:\n{df_clean.head()}")
