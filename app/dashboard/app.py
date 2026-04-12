"""
src/dashboard/app.py — MDK Mining AI Streamlit Dashboard

Connects to PostgreSQL via SQLAlchemy.
Tabs: Fleet Overview, Device Detail, KPI Trends, AI Insights
"""

from sqlalchemy import create_engine, text
import plotly.express as px
import streamlit as st
import pandas as pd

from app.config import settings


def get_db_connection():
    return create_engine(settings.DATABASE_URL)


def query(sql: str) -> pd.DataFrame:
    with get_db_connection().connect() as conn:
        return pd.read_sql(text(sql), conn)


st.set_page_config(page_title="MDK Mining AI", page_icon="⛏️", layout="wide")
st.title("MDK Mining AI Dashboard")

try:
    q = query("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'kpi'")
    tables_exist = q.iloc[0, 0] > 0
except:
    tables_exist = False

if not tables_exist:
    st.warning("⚠️ Ejecuta primero: python -m src.pipeline.kpi")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Fleet Overview", "Device Detail", "KPI Trends", "AI Insights"]
)

with tab1:
    st.header("Fleet Overview")

    summary = query("""
        SELECT
            COUNT(DISTINCT device_id) as total_miners,
            AVG(asic_hashrate_th) as avg_hashrate,
            AVG(chip_temperature_c) as avg_temp,
            AVG(asic_power_w) as avg_power,
            AVG(true_efficiency_jth) as avg_te,
            AVG(economic_te) as avg_ete,
            COUNT(DISTINCT CASE WHEN economic_te < 1.0 THEN device_id END) as profitable,
            COUNT(DISTINCT CASE WHEN economic_te >= 1.0 THEN device_id END) as unprofitable
        FROM kpi
        WHERE true_efficiency_jth IS NOT NULL
    """).iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Miners", f"{summary.total_miners:.0f}")
    col2.metric("Avg Hashrate", f"{summary.avg_hashrate:.1f} TH/s")
    col3.metric("Avg TE", f"{summary.avg_te:.1f} J/TH")
    col4.metric("Avg ETE", f"{summary.avg_ete:.2f}")
    col5.metric(
        "Profitable", f"{summary.profitable:.0f}", f"{summary.unprofitable:.0f} loss"
    )

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("True Efficiency by Device (Top 20)")
        te_data = query("""
            SELECT device_id, AVG(true_efficiency_jth) as te, model
            FROM kpi WHERE true_efficiency_jth IS NOT NULL
            GROUP BY device_id, model ORDER BY te ASC LIMIT 20
        """)
        fig = px.bar(
            te_data, x="device_id", y="te", color="te", color_continuous_scale="Viridis"
        )
        fig.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.subheader("Operating Modes")
        mode_data = query("""
            SELECT operating_mode, COUNT(DISTINCT device_id) as count
            FROM kpi GROUP BY operating_mode
        """)
        fig = px.pie(mode_data, names="operating_mode", values="count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)


with tab2:
    st.header("Device Detail")

    devices = query("SELECT DISTINCT device_id FROM kpi ORDER BY device_id")[
        "device_id"
    ].tolist()
    selected = st.selectbox("Select Device", devices)

    if selected:
        stats = query(f"""
            SELECT asic_hashrate_th, chip_temperature_c, asic_power_w,
                   true_efficiency_jth, economic_te, fan_speed_rpm, error_count, model
            FROM kpi WHERE device_id = '{selected}'
            ORDER BY timestamp DESC LIMIT 1
        """).iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Hashrate", f"{stats.asic_hashrate_th:.1f} TH/s")
        col2.metric("Temperature", f"{stats.chip_temperature_c:.1f}°C")
        col3.metric("Power", f"{stats.asic_power_w:.0f} W")
        col4.metric("TE", f"{stats.true_efficiency_jth:.1f} J/TH")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("ETE", f"{stats.economic_te:.2f}")
        col6.metric("Fan", f"{stats.fan_speed_rpm:.0f} RPM")
        col7.metric("Errors", f"{stats.error_count:.0f}")
        col8.metric("Model", stats.model)

        st.subheader("Time Series")
        ts_data = query(f"""
            SELECT timestamp, asic_hashrate_th, chip_temperature_c, asic_power_w
            FROM kpi WHERE device_id = '{selected}' ORDER BY timestamp
        """)

        tab_h, tab_t, tab_p = st.tabs(["Hashrate", "Temperature", "Power"])
        with tab_h:
            fig = px.line(
                ts_data,
                x="timestamp",
                y="asic_hashrate_th",
                labels={"asic_hashrate_th": "TH/s"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab_t:
            fig = px.line(
                ts_data,
                x="timestamp",
                y="chip_temperature_c",
                labels={"chip_temperature_c": "°C"},
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab_p:
            fig = px.line(
                ts_data, x="timestamp", y="asic_power_w", labels={"asic_power_w": "W"}
            )
            st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.header("KPI Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fleet Avg TE Over Time")
        trend = query("""
            SELECT DATE_TRUNC('hour', timestamp) as t, AVG(true_efficiency_jth) as avg_te
            FROM kpi WHERE true_efficiency_jth IS NOT NULL
            GROUP BY t ORDER BY t
        """)
        fig = px.line(trend, x="t", y="avg_te", labels={"avg_te": "J/TH"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Economic TE Over Time")
        ete_trend = query("""
            SELECT DATE_TRUNC('hour', timestamp) as t, AVG(economic_te) as avg_ete
            FROM kpi WHERE economic_te IS NOT NULL
            GROUP BY t ORDER BY t
        """)
        fig = px.line(ete_trend, x="t", y="avg_ete", labels={"avg_ete": "ETE"})
        st.plotly_chart(fig, use_container_width=True)


with tab4:
    st.header("AI Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Health Score Distribution")
        health = query("""
            SELECT device_id, AVG(error_count) as errors, AVG(chip_temperature_c) as temp
            FROM kpi GROUP BY device_id
        """)
        health["health_score"] = 1 - (health.errors / health.errors.max())
        fig = px.histogram(
            health, x="health_score", nbins=20, labels={"health_score": "Health Score"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Miners at Risk")
        at_risk = query("""
            SELECT device_id, model, AVG(chip_temperature_c) as temp, 
                   AVG(error_count) as errors, AVG(true_efficiency_jth) as te
            FROM kpi WHERE true_efficiency_jth IS NOT NULL
            GROUP BY device_id, model
            HAVING AVG(chip_temperature_c) > 80 OR AVG(error_count) > 50
            ORDER BY errors DESC LIMIT 20
        """)
        st.dataframe(at_risk, use_container_width=True, hide_index=True)

    st.subheader("Recommended Actions")
    actions = query("""
        SELECT device_id, model,
               CASE 
                   WHEN AVG(chip_temperature_c) > 85 THEN 'Review cooling'
                   WHEN AVG(error_count) > 30 THEN 'Check hardware'
                   ELSE 'Normal operation'
               END as action,
               AVG(chip_temperature_c) as temp
        FROM kpi GROUP BY device_id, model
        ORDER BY temp DESC LIMIT 20
    """)
    st.dataframe(actions, use_container_width=True, hide_index=True)
