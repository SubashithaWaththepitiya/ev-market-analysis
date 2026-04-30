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
    page_icon="EV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# CUSTOM CSS FOR A MODERN, POLISHED LOOK
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import a modern typeface */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0B1D3A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #5A6A85;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    /* KPI card styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fd 0%, #eef1f8 100%);
        border: 1px solid #e2e6ef;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        font-weight: 500;
        color: #5A6A85 !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #0B1D3A !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #0B1D3A;
        border-left: 4px solid #0077B6;
        padding-left: 14px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
        font-weight: 500;
    }

    /* Sidebar refinements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1D3A 0%, #1a2f52 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] h2 {
        color: #e0e6f0 !important;
    }

    /* Dataframe container */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.8rem;
        color: #8896AB;
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid #e2e6ef;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">Global EV Market Analysis Dashboard</p>',
            unsafe_allow_html=True)
st.markdown("""<p class="subtitle">
An interactive exploration of the worldwide transition to Electric Vehicles
across 25 countries and 16 years (2010 to 2025). Built with a modular Python
pipeline, SQLite database layer, and deployed via Streamlit Cloud.
</p>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA LOADING (CACHED FOR PERFORMANCE)
# ---------------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_and_clean('ev_vs_petrol_dataset_v3.csv')

try:
    df = get_data()
    st.success(
        f"Data loaded successfully: {df.shape[0]:,} rows and {df.shape[1]} columns"
    )
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Filters")
selected_region = st.sidebar.multiselect(
    "Regions",
    options=df['region'].unique(),
    default=df['region'].unique()
)

selected_years = st.sidebar.slider(
    "Year Range",
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

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**EV Market Analysis**  \n"
    "Modular Python coursework  \n"
    "Deployed with Streamlit Cloud"
)

# ---------------------------------------------------------------------------
# KEY METRICS (KPIs)
# ---------------------------------------------------------------------------
st.markdown('<p class="section-header">Key Performance Indicators</p>',
            unsafe_allow_html=True)

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
    st.metric(label=f"Charging Stations ({latest_year})", value=f"{stations:,.0f}")

with col4:
    avg_co2 = latest_data['co2_emissions_transport_mt'].mean()
    st.metric(label="Avg Transport CO2 (Mt)", value=f"{avg_co2:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------------------------
st.markdown('<p class="section-header">Market Visualisations</p>',
            unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Sales Trends", "Regional Comparison", "Powertrain Mix"])

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
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">Raw Data Explorer</p>',
            unsafe_allow_html=True)
st.dataframe(filtered_df, use_container_width=True, height=420)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("""
<p class="footer-text">
    Global EV Market Analysis | Modular Python Application |
    Powered by Streamlit Cloud and GitHub
</p>
""", unsafe_allow_html=True)
