"""Display the compiled research graph and run it against the latest SQL export."""
import json

import pandas as pd
import streamlit as st

from bikeshare.pipeline import OUTPUT
from bikeshare.research import graph

st.title("DC bikeshare demand")
if not (OUTPUT / "evaluation.json").exists():
    st.info("Trigger dc_bikeshare_demand in Airflow to populate this dashboard.")
    st.stop()
app = graph()
st.subheader("Adaptive workflow")
nodes = app.get_graph()
lines = ["digraph { rankdir=LR; node [shape=box];"]
for edge in nodes.edges:
    lines.append(f'"{edge.source}" -> "{edge.target}";')
st.graphviz_chart("\n".join(lines + ["}"]))
simulate = st.checkbox("Simulate a transient training failure")
if st.button("Run workflow"):
    frame = pd.read_csv(OUTPUT / "daily.csv")
    trace = st.empty()
    visited = []
    for event in app.stream({"frame": frame, "adaptive": True, "fail_first": simulate},
                            stream_mode="updates"):
        for node, update in event.items():
            visited.append(node)
            if node == "holdout":
                st.success(f"Holdout MAE: {update['result']['holdout_mae']:.1f} rentals")
        trace.write(" → ".join(visited))
report = json.loads((OUTPUT / "evaluation.json").read_text())
st.subheader("Holdout comparison")
st.dataframe(pd.DataFrame([
    {"Workflow": label, "Model": report[key]["selected"],
     "MAE (rentals)": round(report[key]["holdout_mae"], 1)}
    for key, label in [("fixed", "Fixed Airflow analysis"), ("simple", "Simple agent baseline"),
                       ("adaptive", "Adaptive agents")]
]), hide_index=True)
st.subheader("Evaluation cases")
st.dataframe(pd.DataFrame(report["cases"]), hide_index=True)
st.subheader("SQL summary")
st.dataframe(pd.read_csv(OUTPUT / "weather_summary.csv"), hide_index=True)
