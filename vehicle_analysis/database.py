"""
database.py -- SQLite Database Integration Module
====================================================

Handles schema design, data ingestion, and analytical queries for
persistent storage of the cleaned EV market dataset.

Design decisions:
    - Normalised schema: separate countries and vehicle_sales tables
      linked by foreign key, reducing redundancy.
    - Indexes on frequently queried columns (country, year, region).
    - Context-managed connections to prevent resource leaks.
    - Parameterised queries throughout to guard against injection.
    - Bulk insert with explicit transaction control for performance.
    - All query functions return pandas DataFrames for seamless
      integration with the analysis and visualisation modules.
"""

import sqlite3
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "ev_market_analysis.db"


# ---------------------------------------------------------------------------
# 1. CONNECTION MANAGEMENT
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    Context-managed wrapper around a SQLite database connection.

    Usage:
        with DatabaseManager("my_database.db") as db:
            db.create_schema()
            db.insert_data(df)
            results = db.run_query("SELECT * FROM countries")
    """

    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        # Enable foreign key enforcement (off by default in SQLite)
        self.conn.execute("PRAGMA foreign_keys = ON")
        logger.info(f"Connected to database: {self.db_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
        return False  # do not suppress exceptions

    def run_query(self, sql, params=None):
        """Execute a SELECT query and return results as a DataFrame."""
        if params is None:
            params = []
        return pd.read_sql_query(sql, self.conn, params=params)

    def execute(self, sql, params=None):
        """Execute a non-SELECT statement (INSERT, UPDATE, CREATE, etc.)."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        self.conn.commit()
        return cursor

    def executemany(self, sql, data):
        """Execute a parameterised statement for multiple rows."""
        cursor = self.conn.cursor()
        cursor.executemany(sql, data)
        self.conn.commit()
        return cursor

    # -------------------------------------------------------------------
    # 2. SCHEMA CREATION
    # -------------------------------------------------------------------

    def create_schema(self):
        """
        Create a normalised database schema with two tables:

        countries
        ---------
        country_id  INTEGER PRIMARY KEY
        name        TEXT UNIQUE
        region      TEXT

        vehicle_sales
        -------------
        id                          INTEGER PRIMARY KEY
        country_id                  INTEGER (FK -> countries)
        year                        INTEGER
        vehicle_segment             TEXT
        powertrain_type             TEXT
        ev_sales                    INTEGER
        petrol_car_sales            INTEGER
        diesel_car_sales            INTEGER
        total_vehicle_sales         INTEGER
        ev_market_share             REAL
        charging_stations           INTEGER
        fast_chargers_share         REAL
        avg_ev_range_km             REAL
        fuel_price_usd_per_liter    REAL
        electricity_price_usd_per_kwh REAL
        gdp_per_capita              REAL
        urban_population_percent    REAL
        co2_emissions_transport_mt  REAL
        ev_subsidy_usd              REAL
        emission_regulation_score   REAL
        ev_growth_rate_yoy          REAL
        is_ev_dominant              INTEGER (0 or 1)

        Indexes are created on country_id, year, region, and
        vehicle_segment for query performance.
        """
        cursor = self.conn.cursor()

        # Drop existing tables for a clean rebuild
        cursor.execute("DROP TABLE IF EXISTS vehicle_sales")
        cursor.execute("DROP TABLE IF EXISTS countries")

        # Countries dimension table
        cursor.execute("""
            CREATE TABLE countries (
                country_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                region      TEXT NOT NULL
            )
        """)

        # Vehicle sales fact table
        cursor.execute("""
            CREATE TABLE vehicle_sales (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                country_id                  INTEGER NOT NULL,
                year                        INTEGER NOT NULL,
                vehicle_segment             TEXT NOT NULL,
                powertrain_type             TEXT NOT NULL,
                ev_sales                    INTEGER,
                petrol_car_sales            INTEGER,
                diesel_car_sales            INTEGER,
                total_vehicle_sales         INTEGER,
                ev_market_share             REAL,
                charging_stations           INTEGER,
                fast_chargers_share         REAL,
                avg_ev_range_km             REAL,
                fuel_price_usd_per_liter    REAL,
                electricity_price_usd_per_kwh REAL,
                gdp_per_capita              REAL,
                urban_population_percent    REAL,
                co2_emissions_transport_mt  REAL,
                ev_subsidy_usd              REAL,
                emission_regulation_score   REAL,
                ev_growth_rate_yoy          REAL,
                is_ev_dominant              INTEGER DEFAULT 0,
                FOREIGN KEY (country_id) REFERENCES countries(country_id)
            )
        """)

        # Performance indexes
        cursor.execute(
            "CREATE INDEX idx_sales_country ON vehicle_sales(country_id)")
        cursor.execute(
            "CREATE INDEX idx_sales_year ON vehicle_sales(year)")
        cursor.execute(
            "CREATE INDEX idx_sales_segment ON vehicle_sales(vehicle_segment)")
        cursor.execute(
            "CREATE INDEX idx_sales_country_year ON vehicle_sales(country_id, year)")

        # Analytical view: joins country name into sales for convenience
        cursor.execute("DROP VIEW IF EXISTS v_sales_with_country")
        cursor.execute("""
            CREATE VIEW v_sales_with_country AS
            SELECT
                c.name AS country,
                c.region,
                vs.*
            FROM vehicle_sales vs
            JOIN countries c ON vs.country_id = c.country_id
        """)

        self.conn.commit()
        logger.info("Database schema created (2 tables, 4 indexes, 1 view)")

    # -------------------------------------------------------------------
    # 3. DATA INSERTION
    # -------------------------------------------------------------------

    def insert_data(self, df):
        """
        Populate the database from a cleaned DataFrame.

        The process runs in two phases:
        1. Extract unique country-region pairs and insert into `countries`.
        2. Map country names to their generated IDs, then bulk-insert
           all rows into `vehicle_sales`.

        Both phases use explicit transactions for atomicity.
        """
        cursor = self.conn.cursor()

        # Phase 1: Insert countries
        country_regions = (
            df[["country", "region"]]
            .drop_duplicates()
            .sort_values("country")
        )
        country_data = [
            (str(row["country"]), str(row["region"]))
            for _, row in country_regions.iterrows()
        ]
        cursor.executemany(
            "INSERT OR IGNORE INTO countries (name, region) VALUES (?, ?)",
            country_data
        )
        self.conn.commit()
        logger.info(f"Inserted {len(country_data)} countries")

        # Build a name -> id lookup
        country_map = {}
        for row in cursor.execute("SELECT country_id, name FROM countries"):
            country_map[row[1]] = row[0]

        # Phase 2: Bulk-insert vehicle sales
        sales_rows = []
        for _, row in df.iterrows():
            country_id = country_map.get(str(row["country"]))
            if country_id is None:
                continue
            sales_rows.append((
                country_id,
                int(row["year"]),
                str(row["vehicle_segment"]),
                str(row["powertrain_type"]),
                int(row["ev_sales"]),
                int(row["petrol_car_sales"]),
                int(row["diesel_car_sales"]),
                int(row["total_vehicle_sales"]),
                float(row["ev_market_share"]),
                int(row["charging_stations"]),
                float(row["fast_chargers_share"]),
                float(row["avg_ev_range_km"]),
                float(row["fuel_price_usd_per_liter"]),
                float(row["electricity_price_usd_per_kwh"]),
                float(row["gdp_per_capita"]),
                float(row["urban_population_percent"]),
                float(row["co2_emissions_transport_mt"]),
                float(row["ev_subsidy_usd"]),
                float(row["emission_regulation_score"]),
                float(row["ev_growth_rate_yoy"]),
                int(row["is_ev_dominant"]),
            ))

        cursor.executemany("""
            INSERT INTO vehicle_sales (
                country_id, year, vehicle_segment, powertrain_type,
                ev_sales, petrol_car_sales, diesel_car_sales,
                total_vehicle_sales, ev_market_share, charging_stations,
                fast_chargers_share, avg_ev_range_km,
                fuel_price_usd_per_liter, electricity_price_usd_per_kwh,
                gdp_per_capita, urban_population_percent,
                co2_emissions_transport_mt, ev_subsidy_usd,
                emission_regulation_score, ev_growth_rate_yoy,
                is_ev_dominant
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, sales_rows)

        self.conn.commit()
        logger.info(f"Inserted {len(sales_rows)} vehicle sales records")

    # -------------------------------------------------------------------
    # 4. VERIFICATION
    # -------------------------------------------------------------------

    def verify_integrity(self):
        """
        Run basic integrity checks on the populated database.

        Returns a dict with row counts and referential integrity status.
        """
        countries = self.run_query("SELECT COUNT(*) AS n FROM countries")
        sales = self.run_query("SELECT COUNT(*) AS n FROM vehicle_sales")

        # Check for orphan records (sales without a valid country)
        orphans = self.run_query("""
            SELECT COUNT(*) AS n FROM vehicle_sales
            WHERE country_id NOT IN (SELECT country_id FROM countries)
        """)

        result = {
            "countries_count": int(countries["n"].iloc[0]),
            "sales_count": int(sales["n"].iloc[0]),
            "orphan_records": int(orphans["n"].iloc[0]),
        }

        status = "PASSED" if result["orphan_records"] == 0 else "FAILED"
        logger.info(
            f"Integrity check {status}: {result['countries_count']} countries, "
            f"{result['sales_count']} sales, {result['orphan_records']} orphans"
        )
        return result


# ---------------------------------------------------------------------------
# 5. ANALYTICAL QUERIES
# ---------------------------------------------------------------------------

def query_top_ev_countries(db, n=5):
    """Top N countries by cumulative EV sales."""
    return db.run_query(f"""
        SELECT
            c.name AS country,
            c.region,
            SUM(vs.ev_sales) AS total_ev_sales,
            ROUND(AVG(vs.ev_market_share), 2) AS avg_ev_share
        FROM vehicle_sales vs
        JOIN countries c ON vs.country_id = c.country_id
        GROUP BY c.name, c.region
        ORDER BY total_ev_sales DESC
        LIMIT {n}
    """)


def query_avg_co2_by_region(db):
    """Average transport CO2 emissions by region."""
    return db.run_query("""
        SELECT
            c.region,
            COUNT(DISTINCT c.name) AS countries,
            ROUND(AVG(vs.co2_emissions_transport_mt), 2) AS avg_co2_mt,
            ROUND(MIN(vs.co2_emissions_transport_mt), 2) AS min_co2_mt,
            ROUND(MAX(vs.co2_emissions_transport_mt), 2) AS max_co2_mt
        FROM vehicle_sales vs
        JOIN countries c ON vs.country_id = c.country_id
        GROUP BY c.region
        ORDER BY avg_co2_mt DESC
    """)


def query_ev_growth_by_year(db):
    """Global EV sales and market share aggregated by year."""
    return db.run_query("""
        SELECT
            vs.year,
            SUM(vs.ev_sales) AS global_ev_sales,
            SUM(vs.total_vehicle_sales) AS global_total_sales,
            ROUND(
                CAST(SUM(vs.ev_sales) AS REAL) /
                SUM(vs.total_vehicle_sales) * 100, 2
            ) AS global_ev_share_pct
        FROM vehicle_sales vs
        GROUP BY vs.year
        ORDER BY vs.year
    """)


def query_ev_dominant_summary(db):
    """Countries and years where EV achieved market dominance."""
    return db.run_query("""
        SELECT
            c.name AS country,
            vs.year,
            vs.vehicle_segment,
            vs.ev_sales,
            vs.petrol_car_sales,
            vs.diesel_car_sales,
            vs.ev_market_share
        FROM vehicle_sales vs
        JOIN countries c ON vs.country_id = c.country_id
        WHERE vs.is_ev_dominant = 1
        ORDER BY c.name, vs.year, vs.vehicle_segment
    """)


def query_infrastructure_leaders(db, year=2025):
    """Countries with the most charging stations in a given year."""
    return db.run_query("""
        SELECT
            c.name AS country,
            c.region,
            MAX(vs.charging_stations) AS stations,
            ROUND(MAX(vs.fast_chargers_share), 1) AS fast_pct,
            ROUND(AVG(vs.ev_market_share), 2) AS ev_share
        FROM vehicle_sales vs
        JOIN countries c ON vs.country_id = c.country_id
        WHERE vs.year = ?
        GROUP BY c.name, c.region
        ORDER BY stations DESC
        LIMIT 10
    """, params=[year])


def query_subsidy_effectiveness(db):
    """Compare EV market share in countries with high vs low subsidies."""
    return db.run_query("""
        SELECT
            CASE
                WHEN vs.ev_subsidy_usd >= 5000 THEN 'High (5000+ USD)'
                WHEN vs.ev_subsidy_usd >= 2000 THEN 'Medium (2000-5000 USD)'
                WHEN vs.ev_subsidy_usd > 0 THEN 'Low (1-2000 USD)'
                ELSE 'No subsidy'
            END AS subsidy_tier,
            COUNT(*) AS records,
            ROUND(AVG(vs.ev_market_share), 2) AS avg_ev_share,
            ROUND(AVG(vs.ev_sales), 0) AS avg_ev_sales
        FROM vehicle_sales vs
        GROUP BY subsidy_tier
        ORDER BY avg_ev_share DESC
    """)


def query_country_yearly_ranking(db, year=2025):
    """Full country ranking for a specific year by EV market share."""
    return db.run_query("""
        SELECT
            c.name AS country,
            c.region,
            SUM(vs.ev_sales) AS ev_sales,
            SUM(vs.petrol_car_sales) AS petrol_sales,
            SUM(vs.diesel_car_sales) AS diesel_sales,
            ROUND(AVG(vs.ev_market_share), 2) AS ev_share,
            ROUND(AVG(vs.gdp_per_capita), 0) AS gdp_per_capita
        FROM vehicle_sales vs
        JOIN countries c ON vs.country_id = c.country_id
        WHERE vs.year = ?
        GROUP BY c.name, c.region
        ORDER BY ev_share DESC
    """, params=[year])


def query_segment_performance(db):
    """EV performance by vehicle segment across all years."""
    return db.run_query("""
        SELECT
            vs.vehicle_segment,
            SUM(vs.ev_sales) AS total_ev_sales,
            SUM(vs.total_vehicle_sales) AS total_sales,
            ROUND(
                CAST(SUM(vs.ev_sales) AS REAL) /
                SUM(vs.total_vehicle_sales) * 100, 2
            ) AS ev_penetration_pct,
            COUNT(CASE WHEN vs.is_ev_dominant = 1 THEN 1 END) AS dominant_records
        FROM vehicle_sales vs
        GROUP BY vs.vehicle_segment
        ORDER BY ev_penetration_pct DESC
    """)

