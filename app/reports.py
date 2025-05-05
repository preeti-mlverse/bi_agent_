# reports.py
import os
import json
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime
from utils.helpers import (
    plot_chart,
    infer_chart_type,
    export_chart_image,
    export_chart_pdf
)



def load_dashboards():
    dashboards = []
    os.makedirs("dashboards", exist_ok=True)
    for file in os.listdir("dashboards"):
        if file.endswith(".json"):
            with open(os.path.join("dashboards", file), "r") as f:
                try:
                    dashboards.append(json.load(f))
                except Exception as e:
                    st.warning(f"Error loading {file}: {e}")
    return sorted(dashboards, key=lambda d: d.get("saved_at", ""), reverse=True)


def reports_dashboard():
    st.title("📁 Saved Reports")

    dashboards = load_dashboards()
    if not dashboards:
        st.info("No dashboards saved yet.")
        return

    # Filter panel
    st.sidebar.header("🔎 Filter Reports")
    date_range = st.sidebar.date_input("Date Range", [])
    goal_filter = st.sidebar.multiselect("Goal", list(set(d.get("goal") for d in dashboards)))
    chart_filter = st.sidebar.multiselect("Chart Type", list(set(d.get("chart_type", "bar") for d in dashboards)))
    query_search = st.sidebar.text_input("Search by keyword...")

    # Apply filters
    filtered = []
    for d in dashboards:
        dt = datetime.fromisoformat(d["saved_at"])
        if date_range:
            if not (date_range[0] <= dt.date() <= date_range[-1]):
                continue
        if goal_filter and d.get("goal") not in goal_filter:
            continue
        if chart_filter and d.get("chart_type", "bar") not in chart_filter:
            continue
        if query_search and query_search.lower() not in d.get("query", "").lower():
            continue
        filtered.append(d)

    st.write(f"🗂 Showing {len(filtered)} of {len(dashboards)} reports")

    for report in filtered:
        with st.expander(f"🧾 {report['query']} ({report['goal']}) — {report['saved_at']}"):
            st.markdown(f"**Goal:** {report['goal']}")
            st.markdown(f"**Query:** `{report['query']}`")
            if report.get("insight"):
                st.markdown(f"**Insight:**\n{report['insight']}")
            if report.get("preview"):
                df = pd.DataFrame(report["preview"])
                st.dataframe(df)

                if not df.empty and len(df.columns) >= 2:
                    chart = plot_chart(df, report["query"])
                    if isinstance(chart, str) and chart.startswith("❌"):
                        st.warning(chart)
                    else:
                        unique_id = uuid.uuid4().hex  # 🔑 unique key base
                        st.plotly_chart(chart, key=f"chart_{unique_id}")

                        # ✅ Chart downloads with unique keys
                        img_png = export_chart_image(chart, "png")
                        img_jpg = export_chart_image(chart, "jpg")
                        img_pdf = export_chart_pdf(chart)

                        with open(img_png, "rb") as f:
                            st.download_button("🖼️ Download PNG", f, file_name="dashboard.png", key=f"png_{unique_id}")

                        with open(img_jpg, "rb") as f:
                            st.download_button("🖼️ Download JPG", f, file_name="dashboard.jpg", key=f"jpg_{unique_id}")

                        with open(img_pdf, "rb") as f:
                            st.download_button("📄 Download PDF", f, file_name="dashboard.pdf", key=f"pdf_{unique_id}")