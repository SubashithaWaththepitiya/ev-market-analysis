"""
analysis.py -- Data Analysis and Processing Module
====================================================

Provides functions for computing yearly aggregations, correlation
analysis, market-share breakdowns, and summary statistics on the
cleaned EV vs Petrol vehicle dataset.

Each function accepts a pandas DataFrame (as produced by the
data_loader pipeline) and returns either a DataFrame or a scalar
result, keeping the interface consistent for downstream consumers.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. YEARLY AVERAGES BY COUNTRY
# ---------------------------------------------------------------------------

def yearly_averages_by_country(df):
    """
    Calculate the mean EV, petrol, and diesel sales per year per country.

    Aggregates across vehicle segments (commercial, mass_market, premium)
    so that each row represents a single country-year combination with
    the average sales figure for each powertrain category.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with columns: country, year, ev_sales,
        petrol_car_sales, diesel_car_sales.

    Returns
    -------
    pd.DataFrame
        Columns: country, year, avg_ev_sales, avg_petrol_sales,
        avg_diesel_sales.
    """
    sales_cols = {
        "ev_sales": "avg_ev_sales",
        "petrol_car_sales": "avg_petrol_sales",
        "diesel_car_sales": "avg_diesel_sales",
    }

    result = (
        df.groupby(["country", "year"], observed=True)
        .agg(
            avg_ev_sales=("ev_sales", "mean"),
            avg_petrol_sales=("petrol_car_sales", "mean"),
            avg_diesel_sales=("diesel_car_sales", "mean"),
        )
        .round(2)
        .reset_index()
    )

    logger.info(
        f"Computed yearly averages: {result.shape[0]} country-year combinations"
    )
    return result


# ---------------------------------------------------------------------------
# 2. YEARLY MAXIMUM SALES BY COUNTRY
# ---------------------------------------------------------------------------

def yearly_max_sales_by_country(df):
    """
    Find the maximum EV, petrol, and diesel sales recorded in each
    country-year pair (i.e. the highest-selling vehicle segment).

    This highlights which segment drives peak volume in a given market
    and year.

    Returns
    -------
    pd.DataFrame
        Columns: country, year, max_ev_sales, max_petrol_sales,
        max_diesel_sales.
    """
    result = (
        df.groupby(["country", "year"], observed=True)
        .agg(
            max_ev_sales=("ev_sales", "max"),
            max_petrol_sales=("petrol_car_sales", "max"),
            max_diesel_sales=("diesel_car_sales", "max"),
        )
        .reset_index()
    )

    logger.info(
        f"Computed yearly max sales: {result.shape[0]} country-year combinations"
    )
    return result


# ---------------------------------------------------------------------------
# 3. TOTAL SALES BY COUNTRY AND YEAR  (aggregated across segments)
# ---------------------------------------------------------------------------

def total_sales_by_country_year(df):
    """
    Sum EV, petrol, and diesel sales across all vehicle segments for
    each country-year pair. Also re-computes total_vehicle_sales and
    the EV market share from the summed figures.

    This aggregated view is the most useful for trend analysis because
    it removes the segment-level granularity and focuses on national
    totals.
    """
    result = (
        df.groupby(["country", "year", "region"], observed=True)
        .agg(
            ev_sales=("ev_sales", "sum"),
            petrol_car_sales=("petrol_car_sales", "sum"),
            diesel_car_sales=("diesel_car_sales", "sum"),
            total_vehicle_sales=("total_vehicle_sales", "sum"),
            ev_market_share=("ev_market_share", "mean"),
            co2_emissions=("co2_emissions_transport_mt", "first"),
            charging_stations=("charging_stations", "first"),
            fast_chargers_share=("fast_chargers_share", "first"),
            gdp_per_capita=("gdp_per_capita", "first"),
        )
        .reset_index()
    )

    # Recalculate EV share from the summed totals for accuracy
    result["ev_share_computed"] = (
        result["ev_sales"] / result["total_vehicle_sales"] * 100
    ).round(2)

    logger.info(f"Aggregated total sales: {result.shape[0]} rows")
    return result


# ---------------------------------------------------------------------------
# 4. CORRELATION: EV MARKET SHARE vs CO2 EMISSIONS
# ---------------------------------------------------------------------------

def correlation_ev_share_co2(df):
    """
    Compute the Pearson and Spearman correlation coefficients between
    EV market share and transport CO2 emissions.

    Pearson captures linear relationships while Spearman captures
    monotonic (rank-based) relationships, giving a more complete
    picture.

    Returns
    -------
    dict
        Keys: 'pearson', 'spearman', each containing the correlation
        coefficient and p-value.
    """
    from scipy import stats

    ev_share = df["ev_market_share"].astype(float)
    co2 = df["co2_emissions_transport_mt"].astype(float)

    # Drop rows where either value is NaN
    mask = ev_share.notna() & co2.notna()
    ev_share = ev_share[mask]
    co2 = co2[mask]

    pearson_r, pearson_p = stats.pearsonr(ev_share, co2)
    spearman_r, spearman_p = stats.spearmanr(ev_share, co2)

    results = {
        "pearson": {"coefficient": round(pearson_r, 4), "p_value": round(pearson_p, 6)},
        "spearman": {"coefficient": round(spearman_r, 4), "p_value": round(spearman_p, 6)},
        "sample_size": int(mask.sum()),
    }

    logger.info(
        f"Correlation (EV share vs CO2): Pearson r={pearson_r:.4f}, "
        f"Spearman rho={spearman_r:.4f}"
    )
    return results


def correlation_matrix(df, columns=None):
    """
    Build a correlation matrix for the specified numeric columns.

    If no columns are provided, a sensible default set is used covering
    sales volumes, market share, infrastructure, and economic indicators.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix.
    """
    if columns is None:
        columns = [
            "ev_market_share", "co2_emissions_transport_mt",
            "charging_stations", "fast_chargers_share",
            "avg_ev_range_km", "fuel_price_usd_per_liter",
            "electricity_price_usd_per_kwh", "gdp_per_capita",
            "ev_subsidy_usd", "emission_regulation_score",
        ]

    valid_cols = [c for c in columns if c in df.columns]
    matrix = df[valid_cols].corr().round(3)

    logger.info(f"Correlation matrix computed for {len(valid_cols)} variables")
    return matrix


# ---------------------------------------------------------------------------
# 5. HIGHEST EV MARKET SHARE YEARS (country-wise)
# ---------------------------------------------------------------------------

def highest_ev_market_share_years(df):
    """
    For each country, find the year with the highest average EV market
    share and report the corresponding EV, petrol, and diesel sales in
    that year.

    Returns
    -------
    pd.DataFrame
        One row per country with columns: country, peak_year,
        peak_ev_share, ev_sales, petrol_sales, diesel_sales.
    """
    # Aggregate to country-year level first
    agg = (
        df.groupby(["country", "year"], observed=True)
        .agg(
            ev_market_share=("ev_market_share", "mean"),
            ev_sales=("ev_sales", "sum"),
            petrol_sales=("petrol_car_sales", "sum"),
            diesel_sales=("diesel_car_sales", "sum"),
        )
        .reset_index()
    )

    # Pick the row with the highest EV market share per country
    idx = agg.groupby("country", observed=True)["ev_market_share"].idxmax()
    peak_years = agg.loc[idx].copy()
    peak_years = peak_years.rename(columns={
        "year": "peak_year",
        "ev_market_share": "peak_ev_share",
    })
    peak_years["peak_ev_share"] = peak_years["peak_ev_share"].round(2)

    peak_years = peak_years.sort_values("peak_ev_share", ascending=False)
    peak_years = peak_years.reset_index(drop=True)

    logger.info(f"Identified peak EV share years for {len(peak_years)} countries")
    return peak_years


# ---------------------------------------------------------------------------
# 6. YEAR-OVER-YEAR GROWTH ANALYSIS
# ---------------------------------------------------------------------------

def yoy_growth_by_country(df):
    """
    Calculate the year-over-year percentage change in EV sales for each
    country. This is derived from the summed (segment-aggregated) sales
    to give a national-level growth picture.

    Returns
    -------
    pd.DataFrame
        Columns: country, year, ev_sales, ev_yoy_growth_pct.
    """
    totals = (
        df.groupby(["country", "year"], observed=True)["ev_sales"]
        .sum()
        .reset_index()
        .sort_values(["country", "year"])
    )

    totals["ev_yoy_growth_pct"] = (
        totals.groupby("country", observed=True)["ev_sales"]
        .pct_change() * 100
    ).round(2)

    logger.info(f"Computed YoY EV growth for {totals['country'].nunique()} countries")
    return totals


# ---------------------------------------------------------------------------
# 7. EV-DOMINANT YEARS COUNT
# ---------------------------------------------------------------------------

def ev_dominant_years_by_country(df):
    """
    Count how many year-segment combinations each country has where
    EVs are the dominant powertrain (is_ev_dominant == True).

    Also counts the distinct years in which at least one segment was
    EV-dominant.

    Returns
    -------
    pd.DataFrame
        Columns: country, dominant_records, dominant_years.
    """
    ev_dom = df[df["is_ev_dominant"] == True].copy()

    records = (
        ev_dom.groupby("country", observed=True)
        .agg(
            dominant_records=("is_ev_dominant", "size"),
            dominant_years=("year", "nunique"),
        )
        .reset_index()
        .sort_values("dominant_years", ascending=False)
    )

    # Include countries with zero dominant years
    all_countries = pd.DataFrame({"country": df["country"].unique()})
    result = all_countries.merge(records, on="country", how="left")
    result["dominant_records"] = result["dominant_records"].fillna(0).astype(int)
    result["dominant_years"] = result["dominant_years"].fillna(0).astype(int)
    result = result.sort_values("dominant_years", ascending=False).reset_index(drop=True)

    logger.info(
        f"EV-dominant analysis: {result[result['dominant_years'] > 0].shape[0]} "
        f"countries have at least one EV-dominant year"
    )
    return result


# ---------------------------------------------------------------------------
# 8. COMPOUND ANNUAL GROWTH RATE (CAGR)
# ---------------------------------------------------------------------------

def cagr_by_country(df, start_year=2015, end_year=2025):
    """
    Calculate the Compound Annual Growth Rate of EV sales for each
    country between the specified start and end years.

    CAGR = (ending_value / beginning_value)^(1/n) - 1

    Countries with zero or missing sales in the start year are excluded
    since CAGR is undefined for a zero base.

    Returns
    -------
    pd.DataFrame
        Columns: country, start_sales, end_sales, cagr_pct.
    """
    totals = (
        df.groupby(["country", "year"], observed=True)["ev_sales"]
        .sum()
        .reset_index()
    )

    start = totals[totals["year"] == start_year][["country", "ev_sales"]]
    start = start.rename(columns={"ev_sales": "start_sales"})

    end = totals[totals["year"] == end_year][["country", "ev_sales"]]
    end = end.rename(columns={"ev_sales": "end_sales"})

    merged = start.merge(end, on="country")
    # Filter out zero-start countries
    merged = merged[merged["start_sales"] > 0].copy()

    n_years = end_year - start_year
    merged["cagr_pct"] = (
        ((merged["end_sales"] / merged["start_sales"]) ** (1 / n_years) - 1) * 100
    ).round(2)

    merged = merged.sort_values("cagr_pct", ascending=False).reset_index(drop=True)

    logger.info(
        f"CAGR ({start_year}-{end_year}) computed for {len(merged)} countries"
    )
    return merged


# ---------------------------------------------------------------------------
# 9. SUMMARY STATISTICS
# ---------------------------------------------------------------------------

def generate_summary_statistics(df):
    """
    Produce a comprehensive statistical summary of the dataset.

    Returns a dictionary containing:
        - descriptive_stats: describe() output for numeric columns
        - sales_by_region: total EV/petrol/diesel sales grouped by region
        - top_ev_countries: top 10 countries by cumulative EV sales
        - market_overview: overall totals and averages
    """
    # Descriptive statistics for all numeric columns
    descriptive = df.describe().round(2)

    # Regional sales breakdown
    sales_by_region = (
        df.groupby("region", observed=True)
        .agg(
            total_ev=("ev_sales", "sum"),
            total_petrol=("petrol_car_sales", "sum"),
            total_diesel=("diesel_car_sales", "sum"),
            avg_ev_share=("ev_market_share", "mean"),
            countries=("country", "nunique"),
        )
        .round(2)
        .sort_values("total_ev", ascending=False)
        .reset_index()
    )

    # Top 10 EV countries by total cumulative sales
    top_ev = (
        df.groupby("country", observed=True)["ev_sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"ev_sales": "cumulative_ev_sales"})
    )

    # Overall market snapshot
    total_ev = int(df["ev_sales"].sum())
    total_petrol = int(df["petrol_car_sales"].sum())
    total_diesel = int(df["diesel_car_sales"].sum())
    grand_total = total_ev + total_petrol + total_diesel

    market_overview = {
        "total_ev_sales": total_ev,
        "total_petrol_sales": total_petrol,
        "total_diesel_sales": total_diesel,
        "grand_total_sales": grand_total,
        "ev_share_of_total": round(total_ev / grand_total * 100, 2) if grand_total else 0,
        "avg_ev_market_share": round(df["ev_market_share"].mean(), 2),
        "median_ev_market_share": round(df["ev_market_share"].median(), 2),
    }

    summary = {
        "descriptive_stats": descriptive,
        "sales_by_region": sales_by_region,
        "top_ev_countries": top_ev,
        "market_overview": market_overview,
    }

    logger.info("Summary statistics generated successfully")
    return summary


# ---------------------------------------------------------------------------
# 10. MARKET SHARE TREND OVER TIME
# ---------------------------------------------------------------------------

def market_share_trend(df):
    """
    Calculate the global average EV market share by year, weighted by
    total vehicle sales. This gives a more accurate picture than a
    simple average because larger markets contribute proportionally.

    Returns
    -------
    pd.DataFrame
        Columns: year, weighted_ev_share, simple_avg_share, total_ev,
        total_vehicles.
    """
    yearly = (
        df.groupby("year", observed=True)
        .agg(
            total_ev=("ev_sales", "sum"),
            total_vehicles=("total_vehicle_sales", "sum"),
            simple_avg_share=("ev_market_share", "mean"),
        )
        .reset_index()
    )

    yearly["weighted_ev_share"] = (
        yearly["total_ev"] / yearly["total_vehicles"] * 100
    ).round(3)
    yearly["simple_avg_share"] = yearly["simple_avg_share"].round(3)

    logger.info(f"Market share trend computed across {len(yearly)} years")
    return yearly
