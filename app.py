import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Import from our custom package
from vehicle_analysis.data_loader import load_and_clean
from vehicle_analysis.visualisation import (
    plot_sales_trends, 
    plot_global_powertrain_mix,
    plot_regional_sales_comparison
)

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Global EV Market Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Global EV Market Analysis Dashboard (2010–2025)")
st.markdown("""
This interactive dashboard explores the global transition from internal combustion engine (ICE) 
vehicles to Electric Vehicles (EVs). It demonstrates **Distinction-Level** integration of 
cloud computing, interactive data visualisation, and modular Python software engineering.
""")

# ---------------------------------------------------------------------------
# DATA LOADING (CACHED FOR PERFORMANCE)
# ---------------------------------------------------------------------------
@st.cache_data
def get_data():
    # Load and clean using our modular pipeline
    return load_and_clean('ev_vs_petrol_dataset_v3.csv')

try:
    df = get_data()
    st.success(f"✅ Data successfully loaded: {df.shape[0]} rows × {df.shape[1]} columns")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.header("Dashboard Filters")
selected_region = st.sidebar.multiselect(
    "Select Regions:",
    options=df['region'].unique(),
    default=df['region'].unique()
)

selected_years = st.sidebar.slider(
    "Select Year Range:",
    min_value=int(df['year'].min()),
    max_value=int(df['year'].max()),
    value=(2010, 2025)
)

# Apply filters
filtered_df = df[
    (df['region'].isin(selected_region)) & 
    (df['year'] >= selected_years[0]) & 
    (df['year'] <= selected_years[1])
]

# ---------------------------------------------------------------------------
# KEY METRICS (KPIs)
# ---------------------------------------------------------------------------
st.header("1. High-Level Market KPIs")
col1, col2, col3, col4 = st.columns(4)

latest_year = filtered_df['year'].max()
latest_data = filtered_df[filtered_df['year'] == latest_year]

with col1:
    total_evs = latest_data['ev_sales'].sum()
    st.metric(label=f"Total EV Sales ({latest_year})", value=f"{total_evs:,.0f}")

with col2:
    avg_share = latest_data['ev_market_share'].mean()
    st.metric(label=f"Avg EV Market Share ({latest_year})", value=f"{avg_share:.1f}%")

with col3:
    stations = latest_data['charging_stations'].sum()
    st.metric(label=f"Total Charging Stations ({latest_year})", value=f"{stations:,.0f}")

with col4:
    avg_co2 = latest_data['co2_emissions_transport_mt'].mean()
    st.metric(label=f"Avg Transport CO2 (Mt)", value=f"{avg_co2:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------------------------
st.header("2. Market Visualisations")

tab1, tab2, tab3 = st.tabs(["📈 Sales Trends", "🌍 Regional Comparison", "🚗 Powertrain Mix"])

with tab1:
    st.subheader("Global Vehicle Sales Breakdown")
    fig1 = plot_sales_trends(filtered_df)
    st.pyplot(fig1)

with tab2:
    st.subheader("Vehicle Sales by Region")
    fig2 = plot_regional_sales_comparison(filtered_df)
    st.pyplot(fig2)

with tab3:
    st.subheader("Global Powertrain Mix Over Time")
    fig3 = plot_global_powertrain_mix(filtered_df)
    st.pyplot(fig3)

# ---------------------------------------------------------------------------
# DATA TABLE
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Raw Data Explorer")
st.dataframe(filtered_df, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Coursework Demo**\nBuilt with Streamlit and GitHub.")
