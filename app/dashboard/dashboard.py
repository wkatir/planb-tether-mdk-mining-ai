"""
app/dashboard/dashboard.py - MDK Mining AI Streamlit Dashboard.

Connects to DuckDB for embedded analytics.
Tabs: Fleet Overview | Device Detail | KPI Trends | AI Insights
      | Synthetic Data Generator | AI Assistant (Gemma on NVIDIA NIM)
"""

import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(Path(_PROJECT_ROOT) / ".env")

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from app.config import settings


@st.cache_data(ttl=60)
def query(sql: str, params: tuple | None = None) -> pd.DataFrame:
    """Cached DB query. TTL=60s."""
    conn = duckdb.connect(str(settings.DUCKDB_PATH), read_only=True)
    try:
        if params:
            df = conn.execute(sql, list(params)).fetchdf()
        else:
            df = conn.execute(sql).fetchdf()
    finally:
        conn.close()
    return df


st.set_page_config(page_title="MDK Mining AI", page_icon="MDK", layout="wide")
st.title("MDK Mining AI Dashboard")

try:
    q = query(
        "SELECT COUNT(*) AS cnt FROM information_schema.tables WHERE table_name = 'kpi'"
    )
    tables_exist = bool(q.iloc[0, 0] > 0)
except Exception:
    tables_exist = False

if not tables_exist:
    st.warning(
        "KPI table is empty. Use the **Synthetic Data Generator** tab to "
        "seed the database, or run `python -m app.run_all` in a terminal."
    )

tab_overview, tab_device, tab_kpi, tab_ai_insights, tab_synth, tab_assistant = st.tabs(
    [
        "Fleet Overview",
        "Device Detail",
        "KPI Trends",
        "AI Insights",
        "Synthetic Data",
        "AI Assistant",
    ]
)

with tab_overview:
    st.header("Fleet Overview")
    if not tables_exist:
        st.info("No data yet - head to **Synthetic Data** tab.")
    else:
        summary = query("""
            SELECT
                COUNT(DISTINCT device_id) AS total_miners,
                AVG(asic_hashrate_th) AS avg_hashrate,
                AVG(chip_temperature_c) AS avg_temp,
                AVG(asic_power_w) AS avg_power,
                AVG(true_efficiency_jth) AS avg_te,
                AVG(economic_te) AS avg_ete,
                COUNT(DISTINCT CASE WHEN economic_te < 1.0 THEN device_id END) AS profitable,
                COUNT(DISTINCT CASE WHEN economic_te >= 1.0 THEN device_id END) AS unprofitable
            FROM kpi
            WHERE true_efficiency_jth IS NOT NULL
        """).iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Miners", f"{summary.total_miners:.0f}")
        col2.metric("Avg Hashrate", f"{summary.avg_hashrate:.1f} TH/s")
        col3.metric("Avg TE", f"{summary.avg_te:.1f} J/TH")
        col4.metric("Avg ETE", f"{summary.avg_ete:.2f}")
        col5.metric(
            "Profitable",
            f"{summary.profitable:.0f}",
            f"{summary.unprofitable:.0f} loss",
        )

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("True Efficiency by Device (Top 20)")
            te_data = query("""
                SELECT device_id, AVG(true_efficiency_jth) AS te, model
                FROM kpi WHERE true_efficiency_jth IS NOT NULL
                GROUP BY device_id, model ORDER BY te ASC LIMIT 20
            """)
            fig = px.bar(
                te_data,
                x="device_id",
                y="te",
                color="te",
                color_continuous_scale="Viridis",
            )
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with col_chart2:
            st.subheader("Operating Modes")
            mode_data = query("""
                SELECT operating_mode, COUNT(DISTINCT device_id) AS count
                FROM kpi GROUP BY operating_mode
            """)
            fig = px.pie(mode_data, names="operating_mode", values="count", hole=0.4)
            st.plotly_chart(fig, width="stretch")


with tab_device:
    st.header("Device Detail")
    if not tables_exist:
        st.info("No data yet.")
    else:
        devices = query("SELECT DISTINCT device_id FROM kpi ORDER BY device_id")[
            "device_id"
        ].tolist()
        selected = st.selectbox("Select Device", devices)

        if selected:
            stats = query(
                """
                SELECT asic_hashrate_th, chip_temperature_c, asic_power_w,
                       true_efficiency_jth, economic_te, fan_speed_rpm,
                       error_count, model
                FROM kpi WHERE device_id = $1
                ORDER BY timestamp DESC LIMIT 1
                """,
                params=(selected,),
            ).iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Hashrate", f"{stats.asic_hashrate_th:.1f} TH/s")
            col2.metric("Temperature", f"{stats.chip_temperature_c:.1f} C")
            col3.metric("Power", f"{stats.asic_power_w:.0f} W")
            col4.metric("TE", f"{stats.true_efficiency_jth:.1f} J/TH")

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("ETE", f"{stats.economic_te:.2f}")
            col6.metric("Fan", f"{stats.fan_speed_rpm:.0f} RPM")
            col7.metric("Errors", f"{stats.error_count:.0f}")
            col8.metric("Model", stats.model)

            st.subheader("Time Series")
            ts_data = query(
                """
                SELECT timestamp, asic_hashrate_th, chip_temperature_c, asic_power_w
                FROM kpi WHERE device_id = $1 ORDER BY timestamp
                """,
                params=(selected,),
            )

            tab_h, tab_t, tab_p = st.tabs(["Hashrate", "Temperature", "Power"])
            with tab_h:
                fig = px.line(
                    ts_data,
                    x="timestamp",
                    y="asic_hashrate_th",
                    labels={"asic_hashrate_th": "TH/s"},
                )
                st.plotly_chart(fig, width="stretch")
            with tab_t:
                fig = px.line(
                    ts_data,
                    x="timestamp",
                    y="chip_temperature_c",
                    labels={"chip_temperature_c": "C"},
                )
                st.plotly_chart(fig, width="stretch")
            with tab_p:
                fig = px.line(
                    ts_data,
                    x="timestamp",
                    y="asic_power_w",
                    labels={"asic_power_w": "W"},
                )
                st.plotly_chart(fig, width="stretch")


with tab_kpi:
    st.header("KPI Trends")
    if not tables_exist:
        st.info("No data yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Fleet Avg TE Over Time")
            trend = query("""
                SELECT DATE_TRUNC('hour', timestamp) AS t, AVG(true_efficiency_jth) AS avg_te
                FROM kpi WHERE true_efficiency_jth IS NOT NULL
                GROUP BY t ORDER BY t
            """)
            fig = px.line(trend, x="t", y="avg_te", labels={"avg_te": "J/TH"})
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.subheader("Economic TE Over Time")
            ete_trend = query("""
                SELECT DATE_TRUNC('hour', timestamp) AS t, AVG(economic_te) AS avg_ete
                FROM kpi WHERE economic_te IS NOT NULL
                GROUP BY t ORDER BY t
            """)
            fig = px.line(ete_trend, x="t", y="avg_ete", labels={"avg_ete": "ETE"})
            st.plotly_chart(fig, width="stretch")


with tab_ai_insights:
    st.header("AI Insights")
    if not tables_exist:
        st.info("No data yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Health Score Distribution")
            health = query("""
                SELECT device_id, AVG(error_count) AS errors,
                       AVG(chip_temperature_c) AS temp
                FROM kpi GROUP BY device_id
            """)
            max_err = max(float(health.errors.max() or 1.0), 1.0)
            health["health_score"] = 1 - (health.errors / max_err)
            fig = px.histogram(
                health,
                x="health_score",
                nbins=20,
                labels={"health_score": "Health Score"},
            )
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.subheader("Miners at Risk")
            at_risk = query("""
                SELECT device_id, model, AVG(chip_temperature_c) AS temp,
                       AVG(error_count) AS errors, AVG(true_efficiency_jth) AS te
                FROM kpi WHERE true_efficiency_jth IS NOT NULL
                GROUP BY device_id, model
                HAVING AVG(chip_temperature_c) > 80 OR AVG(error_count) > 50
                ORDER BY errors DESC LIMIT 20
            """)
            st.dataframe(at_risk, width="stretch", hide_index=True)

        st.subheader("Recommended Actions")
        actions = query("""
            SELECT device_id, model,
                   CASE
                       WHEN AVG(chip_temperature_c) > 85 THEN 'Review cooling'
                       WHEN AVG(error_count) > 30 THEN 'Check hardware'
                       ELSE 'Normal operation'
                   END AS action,
                   AVG(chip_temperature_c) AS temp
            FROM kpi GROUP BY device_id, model
            ORDER BY temp DESC LIMIT 20
        """)
        st.dataframe(actions, width="stretch", hide_index=True)


with tab_synth:
    st.header("Synthetic Data Generator")
    st.markdown(
        "Generate physics-grounded ASIC telemetry (voltage^2 * freq power model, "
        "thermal RC response, age-based degradation, injected thermal/hashboard/PSU "
        "failures) and push it through the full ML pipeline: ingestion -> features "
        "-> KPIs (TE / ETE / Profit Density) -> optional model training."
    )

    with st.form("synth_form"):
        col_a, col_b, col_c = st.columns(3)
        fleet_size = col_a.slider(
            "Fleet size", min_value=5, max_value=200, value=25, step=5
        )
        days = col_b.slider("Days of history", min_value=1, max_value=30, value=3)
        failure_rate = col_c.slider(
            "Failure injection rate",
            min_value=0.00,
            max_value=0.30,
            value=0.15,
            step=0.05,
            help="Share of miners with thermal / hashboard / PSU pre-failures.",
        )
        col_d, col_e = st.columns(2)
        seed = col_d.number_input("Random seed", value=42, step=1)
        train_models = col_e.checkbox(
            "Train ML models after generation",
            value=False,
            help="Slower (~1-3 min on CPU). Skips on a fresh PoC by default.",
        )
        submitted = st.form_submit_button("Generate & ingest")

    if submitted:
        from app.data.generator import SyntheticDataGenerator
        from app.pipeline.features import FeatureEngineering
        from app.pipeline.ingestion import DataIngestion
        from app.pipeline.kpi import KPIEngine

        settings.ensure_dirs()
        t0 = time.time()
        progress = st.progress(0, text="Generating telemetry...")

        gen = SyntheticDataGenerator(
            fleet_size=int(fleet_size),
            days=int(days),
            failure_rate=float(failure_rate),
            seed=int(seed),
        )
        parquet_path = gen.generate()
        progress.progress(30, text="Ingesting into DuckDB...")

        ingestion = DataIngestion()
        rows = ingestion.ingest_parquet(parquet_path)
        validation = ingestion.validate_bounds()
        ingestion.close()
        progress.progress(55, text="Feature engineering...")

        fe = FeatureEngineering()
        fe.compute_rolling_features()
        fe.compute_cross_device_features()
        fe.export_features()
        fe.close()
        progress.progress(75, text="Computing KPIs (TE, ETE, PD)...")

        kpi = KPIEngine()
        kpi.compute_te()
        kpi_summary = kpi.get_device_kpi_summary()
        kpi.close()
        progress.progress(90, text="Done.")

        if train_models:
            progress.progress(92, text="Training ML models (this is slower)...")
            from app.models.train_models import train_all

            train_all()

        elapsed = time.time() - t0
        progress.progress(100, text=f"Pipeline finished in {elapsed:.1f}s")
        st.cache_data.clear()

        st.success(
            f"Pipeline complete: {rows:,} rows ingested in {elapsed:.1f}s "
            f"(validation rate {validation['validation_rate']:.1%})."
        )
        st.caption(
            "Switch to **Fleet Overview** to inspect the newly generated fleet. "
            "All cached queries were invalidated."
        )
        st.dataframe(kpi_summary.head(10), width="stretch", hide_index=True)


with tab_assistant:
    st.header("AI Assistant (Gemma via NVIDIA NIM)")
    st.markdown(
        "Natural-language access to fleet analytics. The LLM never sees raw "
        "telemetry rows - it calls Python tools that return pre-computed "
        "KPIs and ML outputs. This is the 'ML-first, LLM-on-top' boundary."
    )

    import os

    if not os.getenv("NVIDIA_API_KEY") and not os.getenv("LLM_API_KEY"):
        st.warning(
            "`NVIDIA_API_KEY` (or `LLM_API_KEY`) is not set. Get a free key "
            "at https://build.nvidia.com and add it to `.env`. "
            "Example: `NVIDIA_API_KEY=nvapi-...`"
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about the fleet, e.g. 'worst 3 miners and why'")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from app.ai.agent import FleetAgent

                    agent = FleetAgent()
                    trace = agent.ask(user_q)
                    answer = trace.answer or "(no response)"
                    st.markdown(answer)
                    if trace.tool_calls:
                        with st.expander("Tool calls"):
                            st.json(trace.tool_calls)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as exc:  # noqa: BLE001
                    err = f"LLM call failed: {exc}"
                    st.error(err)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err}
                    )
