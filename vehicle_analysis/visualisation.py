"""
visualisation.py -- Data Visualisation Module
===============================================

Publication-quality charts for EV market analysis. Each function
returns a matplotlib Figure object.

Required CW charts:
    1. Line chart      - Sales trends (country-wise)
    2. Bar chart       - EV-dominant years per country
    3. Scatter/Heatmap - Fast chargers vs EV adoption + correlation heatmap

Business decision-making charts:
    4. Stacked area    - Global powertrain mix
    5. CAGR ranking    - Growth rate comparison
    6. Market share    - Weighted vs simple trend
    7. Bubble chart    - Strategic market positioning
    8. Dual-axis       - Infrastructure vs adoption
    9. Regional bars   - Sales comparison by region
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------------------------
C = {
    "ev": "#0077B6", "petrol": "#E63946", "diesel": "#457B9D",
    "accent": "#F4A261", "teal": "#2A9D8F", "bg": "#FAFAFA",
    "grid": "#E0E0E0", "text": "#2B2D42", "muted": "#8D99AE",
    "dark": "#264653", "coral": "#E76F51", "gold": "#E9C46A",
}

REGION_COLOURS = {
    "Europe": "#0077B6", "Asia": "#E63946", "North America": "#2A9D8F",
    "South America": "#F4A261", "Oceania": "#457B9D",
}


def _style(ax, title, xlabel="", ylabel="", title_size=14):
    ax.set_title(title, fontsize=title_size, fontweight="600",
                 color=C["text"], pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=C["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=C["text"])
    ax.tick_params(colors=C["text"], labelsize=9)
    ax.set_facecolor(C["bg"])
    ax.grid(True, linestyle="--", alpha=0.4, color=C["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["grid"])
    ax.spines["bottom"].set_color(C["grid"])


def _fmt_units(x, _):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ===================================================================
# 1. LINE CHART: SALES TRENDS (CW REQUIRED)
# ===================================================================

def plot_sales_trends(df, countries=None, figsize=(18, 12)):
    """Multi-panel line chart: EV, petrol, diesel trends per country."""
    if countries is None:
        countries = ["United States", "China", "Norway", "Germany",
                     "India", "United Kingdom"]

    agg = (df.groupby(["country", "year"], observed=True)
           .agg(ev=("ev_sales", "sum"), petrol=("petrol_car_sales", "sum"),
                diesel=("diesel_car_sales", "sum"))
           .reset_index())

    n = len(countries)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for i, country in enumerate(countries):
        ax = axes[i]
        s = agg[agg["country"] == country].sort_values("year")
        if s.empty:
            continue

        yrs = s["year"].values

        # Plot with distinct markers and line weights
        ax.plot(yrs, s["ev"], color=C["ev"], lw=2.8, marker="o",
                ms=5, label="EV", zorder=3)
        ax.plot(yrs, s["petrol"], color=C["petrol"], lw=2, marker="s",
                ms=4, label="Petrol", alpha=0.85)
        ax.plot(yrs, s["diesel"], color=C["diesel"], lw=2, marker="^",
                ms=4, label="Diesel", alpha=0.85)

        # Fill under EV line for emphasis
        ax.fill_between(yrs, s["ev"], alpha=0.08, color=C["ev"])

        _style(ax, country, "Year", "Sales")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_units))

        # Annotate final EV value
        last_ev = int(s["ev"].iloc[-1])
        last_yr = int(s["year"].iloc[-1])
        ax.annotate(f"{last_ev:,}", xy=(last_yr, last_ev),
                    xytext=(6, 12), textcoords="offset points",
                    fontsize=8, color=C["ev"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C["muted"], lw=0.5))

        # Mark EV crossover point if it exists
        ev_vals = s["ev"].values
        pet_vals = s["petrol"].values
        for k in range(1, len(ev_vals)):
            if ev_vals[k] > pet_vals[k] and ev_vals[k-1] <= pet_vals[k-1]:
                ax.axvline(x=yrs[k], color=C["teal"], ls=":", alpha=0.7, lw=1.5)
                ax.text(yrs[k], ax.get_ylim()[1] * 0.95, "EV crossover",
                        fontsize=7, color=C["teal"], ha="center",
                        fontstyle="italic", rotation=90, va="top")
                break

        ax.legend(fontsize=8, loc="upper left", framealpha=0.85,
                  edgecolor=C["grid"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Vehicle Sales Trends by Powertrain Type (2010-2025)",
                 fontsize=17, fontweight="bold", color=C["text"], y=1.01)
    fig.tight_layout()
    logger.info(f"Sales trend chart created for {n} countries")
    return fig


# ===================================================================
# 2. BAR CHART: EV-DOMINANT YEARS (CW REQUIRED)
# ===================================================================

def plot_ev_dominant_years(dominant_df, figsize=(13, 7)):
    """Horizontal bar chart of EV-dominant years per country."""
    data = dominant_df.copy()
    has = data[data["dominant_years"] > 0].sort_values("dominant_years")
    no_dom = data[data["dominant_years"] == 0]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    if has.empty:
        ax.text(0.5, 0.5, "No countries have achieved EV dominance",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        return fig

    countries = has["country"].values
    years_val = has["dominant_years"].values
    records = has["dominant_records"].values

    # Gradient colour based on value
    norm = years_val / years_val.max()
    bar_colors = [plt.cm.viridis(0.25 + 0.65 * v) for v in norm]

    bars = ax.barh(countries, years_val, color=bar_colors,
                   edgecolor="white", height=0.55)

    for bar, yr, rec in zip(bars, years_val, records):
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{yr} years ({rec} segment-records)",
                va="center", fontsize=9.5, color=C["text"], fontweight="500")

    _style(ax, "Countries with EV-Dominant Years",
           xlabel="Number of Years")
    ax.set_xlim(0, years_val.max() + 3.5)

    n_dom = len(has)
    n_total = len(data)
    ax.text(0.98, 0.03,
            f"Only {n_dom} of {n_total} countries have achieved EV dominance in any segment",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontstyle="italic", color=C["muted"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["grid"], alpha=0.8))

    plt.tight_layout()
    logger.info("EV-dominant years bar chart created")
    return fig


# ===================================================================
# 3. SCATTER PLOT: FAST CHARGERS vs EV SHARE (CW REQUIRED)
# ===================================================================

def plot_fast_chargers_vs_ev_share(df, figsize=(12, 8)):
    """Scatter: fast charger share vs EV market share with regression."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    x = df["fast_chargers_share"].astype(float).values
    y = df["ev_market_share"].astype(float).values
    mask = np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]

    # Colour points by year for temporal context
    years = df.loc[mask, "year"].astype(int).values if "year" in df.columns else None
    if years is not None:
        sc = ax.scatter(x_c, y_c, c=years, cmap="plasma", alpha=0.5, s=45,
                        edgecolors="white", linewidths=0.4, zorder=2)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Year", fontsize=10)
    else:
        ax.scatter(x_c, y_c, color=C["ev"], alpha=0.4, s=45,
                   edgecolors="white", linewidths=0.4)

    # Regression line
    if len(x_c) > 2:
        coeffs = np.polyfit(x_c, y_c, 1)
        trend_x = np.linspace(x_c.min(), x_c.max(), 100)
        trend_y = np.polyval(coeffs, trend_x)
        ax.plot(trend_x, trend_y, color=C["petrol"], lw=2.5,
                ls="--", label="Linear trend", zorder=3)

        y_pred = np.polyval(coeffs, x_c)
        ss_res = np.sum((y_c - y_pred) ** 2)
        ss_tot = np.sum((y_c - np.mean(y_c)) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        corr = np.corrcoef(x_c, y_c)[0, 1]

        ax.text(0.03, 0.95,
                f"Pearson r = {corr:.3f}\nR² = {r_sq:.3f}\nn = {len(x_c):,}",
                transform=ax.transAxes, fontsize=10.5, va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white",
                          ec=C["grid"], alpha=0.92),
                fontweight="500")

    _style(ax, "Does Fast Charger Availability Drive EV Adoption?",
           "Fast Chargers Share (%)", "EV Market Share (%)")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    logger.info("Fast charger vs EV share scatter created")
    return fig


# ===================================================================
# 4. HEATMAP: CORRELATION MATRIX (CW REQUIRED)
# ===================================================================

def plot_correlation_heatmap(corr_matrix, figsize=(12, 10)):
    """Annotated correlation heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    data = corr_matrix.values
    labels = corr_matrix.columns.tolist()
    short = {
        "ev_market_share": "EV Share", "co2_emissions_transport_mt": "CO2 Emissions",
        "charging_stations": "Charging Stations", "fast_chargers_share": "Fast Chargers %",
        "avg_ev_range_km": "Avg EV Range", "fuel_price_usd_per_liter": "Fuel Price",
        "electricity_price_usd_per_kwh": "Electricity Price",
        "gdp_per_capita": "GDP per Capita", "ev_subsidy_usd": "EV Subsidy",
        "emission_regulation_score": "Regulation Score",
    }
    display_labels = [short.get(l, l) for l in labels]

    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82)
    cbar.set_label("Pearson Correlation Coefficient", fontsize=10)

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9.5)
    ax.set_yticklabels(display_labels, fontsize=9.5)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = data[i, j]
            tc = "white" if abs(v) > 0.55 else C["text"]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color=tc, fontweight="500")

    ax.set_title("Correlation Heatmap of Key Market Indicators",
                 fontsize=14, fontweight="600", color=C["text"], pad=16)
    plt.tight_layout()
    logger.info("Correlation heatmap created")
    return fig


# ===================================================================
# 5. STACKED AREA: POWERTRAIN MIX (BUSINESS)
# ===================================================================

def plot_global_powertrain_mix(df, figsize=(14, 7)):
    """Stacked area: global EV/petrol/diesel share over time."""
    yr = (df.groupby("year", observed=True)
          .agg(ev=("ev_sales", "sum"), petrol=("petrol_car_sales", "sum"),
               diesel=("diesel_car_sales", "sum")).reset_index())
    total = yr["ev"] + yr["petrol"] + yr["diesel"]
    yr["ev_pct"] = yr["ev"] / total * 100
    yr["petrol_pct"] = yr["petrol"] / total * 100
    yr["diesel_pct"] = yr["diesel"] / total * 100

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    years = yr["year"].values
    ax.stackplot(years, yr["ev_pct"], yr["petrol_pct"], yr["diesel_pct"],
                 labels=["EV", "Petrol", "Diesel"],
                 colors=[C["ev"], C["petrol"], C["diesel"]], alpha=0.85)

    _style(ax, "Global Vehicle Sales Composition (2010-2025)",
           "Year", "Share of Total Sales (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(years.min(), years.max())
    ax.legend(loc="center right", fontsize=10, framealpha=0.9)

    ev_end = yr["ev_pct"].iloc[-1]
    ax.annotate(f"EV: {ev_end:.1f}%", xy=(years[-1], ev_end / 2),
                xytext=(-55, 20), textcoords="offset points",
                fontsize=11, color=C["ev"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["ev"], lw=1.5))

    plt.tight_layout()
    logger.info("Powertrain mix chart created")
    return fig


# ===================================================================
# 6. CAGR RANKING (BUSINESS)
# ===================================================================

def plot_cagr_ranking(cagr_df, figsize=(13, 9)):
    """Horizontal bar ranking countries by EV sales CAGR."""
    data = cagr_df.sort_values("cagr_pct", ascending=True).copy()
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    countries = data["country"].values
    cagr = data["cagr_pct"].values
    colours = []
    for v in cagr:
        if v >= 60: colours.append(C["petrol"])
        elif v >= 40: colours.append(C["accent"])
        else: colours.append(C["teal"])

    bars = ax.barh(countries, cagr, color=colours, edgecolor="white", height=0.6)

    for bar, val in zip(bars, cagr):
        ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color=C["text"],
                fontweight="500")

    _style(ax, "EV Sales CAGR by Country (2015-2025)",
           xlabel="Compound Annual Growth Rate (%)")

    legend_els = [
        Patch(facecolor=C["petrol"], label="High growth (60%+)"),
        Patch(facecolor=C["accent"], label="Moderate (40-60%)"),
        Patch(facecolor=C["teal"], label="Steady (<40%)"),
    ]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    logger.info("CAGR ranking chart created")
    return fig


# ===================================================================
# 7. MARKET SHARE TREND (BUSINESS)
# ===================================================================

def plot_market_share_trend(trend_df, figsize=(13, 6)):
    """Weighted vs simple average global EV market share."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    yrs = trend_df["year"].values
    ax.plot(yrs, trend_df["weighted_ev_share"], color=C["ev"], lw=2.8,
            marker="o", ms=6, label="Weighted share (volume-adjusted)")
    ax.plot(yrs, trend_df["simple_avg_share"], color=C["accent"], lw=2.8,
            marker="s", ms=6, ls="--", label="Simple average (equal-weight)")
    ax.fill_between(yrs, trend_df["weighted_ev_share"],
                    trend_df["simple_avg_share"], alpha=0.1, color=C["ev"])

    _style(ax, "Global EV Market Share Trend (2010-2025)",
           "Year", "EV Market Share (%)")

    last = trend_df.iloc[-1]
    ax.annotate(f"Weighted: {last['weighted_ev_share']:.1f}%",
                xy=(last["year"], last["weighted_ev_share"]),
                xytext=(8, 10), textcoords="offset points",
                fontsize=9.5, fontweight="bold", color=C["ev"])
    ax.annotate(f"Simple: {last['simple_avg_share']:.1f}%",
                xy=(last["year"], last["simple_avg_share"]),
                xytext=(8, -15), textcoords="offset points",
                fontsize=9.5, fontweight="bold", color=C["accent"])

    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    plt.tight_layout()
    logger.info("Market share trend chart created")
    return fig


# ===================================================================
# 8. BUBBLE CHART: STRATEGIC MARKET POSITIONING (BUSINESS)
# ===================================================================

def plot_market_positioning(df, target_year=2025, figsize=(14, 9)):
    """
    Bubble chart: GDP per capita (x) vs EV market share (y),
    bubble size = total EV sales. This is a BCG-style strategic
    positioning map for executive presentations.
    """
    snap = (df[df["year"] == target_year]
            .groupby("country", observed=True)
            .agg(ev_share=("ev_market_share", "mean"),
                 gdp=("gdp_per_capita", "first"),
                 ev_sales=("ev_sales", "sum"),
                 region=("region", "first"))
            .reset_index())

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    # Scale bubble size
    max_sales = snap["ev_sales"].max()
    sizes = (snap["ev_sales"] / max_sales) * 2500 + 60

    for _, row in snap.iterrows():
        region = str(row["region"])
        colour = REGION_COLOURS.get(region, C["muted"])
        ax.scatter(row["gdp"], row["ev_share"], s=sizes[_],
                   color=colour, alpha=0.6, edgecolors="white", lw=1.2,
                   zorder=3)
        # Label each bubble
        ax.annotate(row["country"], xy=(row["gdp"], row["ev_share"]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7.5, color=C["text"])

    # Quadrant lines at medians
    med_gdp = snap["gdp"].median()
    med_share = snap["ev_share"].median()
    ax.axvline(med_gdp, color=C["grid"], ls=":", lw=1.2, alpha=0.7)
    ax.axhline(med_share, color=C["grid"], ls=":", lw=1.2, alpha=0.7)

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    quad_style = dict(fontsize=8.5, color=C["muted"], fontstyle="italic",
                      alpha=0.7, ha="center")
    ax.text(med_gdp * 0.5, ylim[1] * 0.92, "High Adoption\nLower GDP", **quad_style)
    ax.text(xlim[1] * 0.85, ylim[1] * 0.92, "High Adoption\nHigh GDP", **quad_style)
    ax.text(med_gdp * 0.5, ylim[0] + 1, "Low Adoption\nLower GDP", **quad_style)
    ax.text(xlim[1] * 0.85, ylim[0] + 1, "Low Adoption\nHigh GDP", **quad_style)

    _style(ax, f"Strategic Market Positioning ({target_year})",
           "GDP per Capita (USD)", "EV Market Share (%)")

    # Region legend
    region_patches = [Patch(fc=col, label=reg, alpha=0.6)
                      for reg, col in REGION_COLOURS.items()]
    ax.legend(handles=region_patches, loc="upper left", fontsize=9,
              title="Region", title_fontsize=10, framealpha=0.9)

    plt.tight_layout()
    logger.info("Market positioning bubble chart created")
    return fig


# ===================================================================
# 9. DUAL-AXIS: INFRASTRUCTURE vs ADOPTION (BUSINESS)
# ===================================================================

def plot_infrastructure_vs_adoption(df, country="China", figsize=(13, 7)):
    """
    Dual-axis chart: bar = charging stations, line = EV sales.
    Shows the co-evolution of infrastructure and adoption.
    """
    snap = (df[df["country"] == country]
            .groupby("year", observed=True)
            .agg(ev_sales=("ev_sales", "sum"),
                 stations=("charging_stations", "first"))
            .reset_index().sort_values("year"))

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    yrs = snap["year"].values

    # Bars for charging stations
    ax1.bar(yrs, snap["stations"], color=C["teal"], alpha=0.45,
            width=0.7, label="Charging Stations", zorder=2)
    ax1.set_ylabel("Charging Stations", fontsize=11, color=C["teal"])
    ax1.tick_params(axis="y", colors=C["teal"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_units))

    # Line for EV sales on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(yrs, snap["ev_sales"], color=C["ev"], lw=2.8, marker="o",
             ms=6, label="EV Sales", zorder=3)
    ax2.set_ylabel("EV Sales (units)", fontsize=11, color=C["ev"])
    ax2.tick_params(axis="y", colors=C["ev"])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_units))

    _style(ax1, f"Infrastructure Growth vs EV Adoption: {country}",
           xlabel="Year")
    ax1.spines["right"].set_visible(True)
    ax1.spines["right"].set_color(C["grid"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               fontsize=9, framealpha=0.9)

    plt.tight_layout()
    logger.info(f"Infrastructure vs adoption chart created for {country}")
    return fig


# ===================================================================
# 10. REGIONAL COMPARISON (BUSINESS)
# ===================================================================

def plot_regional_sales_comparison(df, figsize=(14, 7)):
    """Grouped bar chart comparing EV, petrol, diesel by region."""
    reg = (df.groupby("region", observed=True)
           .agg(ev=("ev_sales", "sum"), petrol=("petrol_car_sales", "sum"),
                diesel=("diesel_car_sales", "sum"))
           .reset_index()
           .sort_values("ev", ascending=False))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    regions = reg["region"].values
    x = np.arange(len(regions))
    w = 0.25

    ax.bar(x - w, reg["ev"], w, color=C["ev"], label="EV", edgecolor="white")
    ax.bar(x, reg["petrol"], w, color=C["petrol"], label="Petrol", edgecolor="white")
    ax.bar(x + w, reg["diesel"], w, color=C["diesel"], label="Diesel", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_units))

    _style(ax, "Cumulative Vehicle Sales by Region and Powertrain (2010-2025)",
           ylabel="Total Sales")
    ax.legend(fontsize=10, framealpha=0.9)

    # Annotate EV values on top of bars
    for i, val in enumerate(reg["ev"].values):
        ax.text(i - w, val + val * 0.02, f"{val/1e6:.1f}M",
                ha="center", fontsize=8, color=C["ev"], fontweight="bold")

    plt.tight_layout()
    logger.info("Regional sales comparison chart created")
    return fig
