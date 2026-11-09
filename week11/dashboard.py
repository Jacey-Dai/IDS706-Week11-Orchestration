"""Live view of the actual compiled LangGraph, with per-node execution state."""
import json
import sys
from pathlib import Path
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from student_loop.graph import build_graph, _initial_state, _finalize_payload

st.set_page_config(page_title="Week 11 | Fraud research", layout="wide")
st.title("Week 11 · Fraud detection research loop")
st.caption("Synthetic classroom data · PostgreSQL → data quality → validation experiments → sealed holdout")
latest = ROOT / "outputs/week11/latest.json"
if not latest.exists():
    st.info("Trigger week11_fraud_pipeline in Airflow first. Results appear when the workflow succeeds.")
    st.stop()
run_id = json.loads(latest.read_text())["run_id"]
out = latest.parent / run_id
analysis = json.loads((out / "analysis.json").read_text())
st.write(f"Pipeline run: {run_id} · database rows: {analysis['rows']:,} · model sample: {analysis['sample_rows']:,}")
app = build_graph()
canvas = st.empty()

def draw(active=None, completed=()):
    graph = app.get_graph()
    lines = ['digraph { rankdir=LR; node [shape=box style="rounded,filled" fillcolor="#eff6ff"];']
    for name in graph.nodes:
        color = '#facc15' if name == active else '#86efac' if name in completed else '#eff6ff'
        lines.append(f'"{name}" [fillcolor="{color}"];')
    for edge in graph.edges:
        lines.append(f'"{edge.source}" -> "{edge.target}";')
    canvas.graphviz_chart('\n'.join(lines + ['}']), use_container_width=True)

draw()
if st.button("Run adapted agents", type="primary"):
    state = _initial_state(dataset_path=analysis['dataset_path'], target_column='is_fraud', target_score=.8, max_cycles=3, seed=42, holdout_frac=.2, model='offline')
    completed = set()
    trace = st.empty()
    for event in app.stream(state, stream_mode="debug", config={"recursion_limit": 40}):
        if event['type'] == 'task':
            draw(event['payload']['name'], completed)
        elif event['type'] == 'task_result':
            name = event['payload']['name']
            completed.add(name)
            patch = dict(event['payload']['result'])
            state.update({k:v for k,v in patch.items() if k != 'steps'})
            state['steps'] += patch.get('steps', [])
            trace.write(f"Completed {name} · cycle {state.get('cycle', 0)}")
            draw(None, completed)
    result = _finalize_payload(state)
    (out / 'dashboard_agent_results.json').write_text(json.dumps(result, indent=2, default=str))
    st.success('Agent workflow completed; sealed holdout evaluated once.')
else:
    result = json.loads((out / 'agent_results.json').read_text())

metrics = result.get('holdout_metrics') or {}
cols = st.columns(4)
for col, name in zip(cols, ['precision', 'recall', 'f1', 'accuracy']):
    col.metric('Holdout ' + name, f"{metrics.get(name, 0):.3f}")
st.write('Confusion matrix (rows=true, columns=predicted; labels 0, 1)', metrics.get('confusion_matrix'))
st.subheader('Validation experiments')
trials = [s for s in result.get('steps', []) if s['node'] == 'ml_engineer']
st.dataframe([{'model':s.get('model_family'), **s.get('validation_metrics', {})} for s in trials], use_container_width=True)
st.subheader('Data quality and evidence')
st.json(result.get('quality'))
with st.expander('Full execution results'):
    st.json(result)
