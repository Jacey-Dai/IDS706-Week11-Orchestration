"""Part 3 dashboard: experiment routing, recovery traces, and benchmark results."""
import sys
import json
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from week11.part3.graph import build_graph, initial_state, summarize

st.set_page_config(page_title='Adaptive research | Week 11',layout='wide')
st.title('Adaptive research workflow')
st.caption('Quality checks, specialist selection, bounded retries, and validation-based threshold selection')
folder=ROOT/'outputs/week11/part3'
latest=json.loads((ROOT/'outputs/week11/latest.json').read_text())['run_id']
path=ROOT/'outputs/week11'/latest/'database_cleaned.csv'
app=build_graph()
canvas=st.empty()

def draw(active=None,completed=()):
    graph=app.get_graph()
    lines=['digraph { rankdir=LR; node [shape=box style="rounded,filled" fontsize=11];']
    for name in graph.nodes:
        color='#facc15' if name==active else '#86efac' if name in completed else '#eff6ff'
        lines.append(f'"{name}" [fillcolor="{color}"];')
    for edge in graph.edges:
        lines.append(f'"{edge.source}" -> "{edge.target}";')
    canvas.graphviz_chart('\n'.join(lines+['}']),use_container_width=True)

draw()
scenario=st.selectbox('Run scenario',['Clean data','One transient training failure'])
if st.button('Run adaptive workflow',type='primary'):
    state=initial_state(path,failure_injection=1 if scenario.startswith('One') else 0)
    completed=set()
    progress=st.empty()
    for event in app.stream(state,stream_mode='debug',config={'recursion_limit':80}):
        if event['type']=='task':
            draw(event['payload']['name'],completed)
        elif event['type']=='task_result':
            name=event['payload']['name']
            if event['payload'].get('error'):
                st.error(str(event['payload']['error']))
                st.stop()
            patch=dict(event['payload']['result'])
            state.update({k:v for k,v in patch.items() if k!='steps'})
            state['steps']+=patch.get('steps',[])
            completed.add(name)
            progress.write(f"Completed {name} · attempts {state.get('attempt',0)} · retries {state.get('retries',0)}")
            draw(None,completed)
    result=summarize(state)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'dashboard_result.json').write_text(json.dumps(result,indent=2))
    st.success('Workflow completed.')
else:
    saved=folder/'adaptive_example.json'
    result=json.loads(saved.read_text()) if saved.exists() else {}
if result:
    cols=st.columns(5)
    for col,name in zip(cols,['precision','recall','f1']):
        col.metric('Holdout '+name,f"{(result.get('holdout_metrics') or {}).get(name,0):.3f}")
    cols[3].metric('Training attempts',result.get('attempt',0))
    cols[4].metric('Retries',result.get('retries',0))
    st.write('Selected specialist:',(result.get('best') or {}).get('family'), '· validation-selected threshold:',f"{result.get('threshold',0):.2f}")
    with st.expander('Execution trace and split audit'):
        st.json({'steps':result.get('steps'),'selection_audit':result.get('selection_audit')})

st.subheader('Comparison with the two classroom baselines')
comparison=folder/'comparison.json'
if comparison.exists():
    runs=json.loads(comparison.read_text())
    table=pd.DataFrame([{'Workflow':r['workflow'],'Seed':r['seed'],'Attempts':r['attempts'],
        'Seconds':r['duration_seconds'],**{k:r['holdout_metrics'][k] for k in ['precision','recall','f1','accuracy']}} for r in runs])
    means=table.groupby('Workflow')[['precision','recall','f1','accuracy','Seconds','Attempts']].mean()
    st.dataframe(means,use_container_width=True)
    plot=means.reset_index().melt(id_vars='Workflow',value_vars=['precision','recall','f1'],var_name='Metric',value_name='Score')
    chart=alt.Chart(plot).mark_bar().encode(x=alt.X('Workflow:N',axis=alt.Axis(labelAngle=0,labelLimit=240)),y=alt.Y('Score:Q',scale=alt.Scale(domain=[0,1])),color='Metric:N',xOffset='Metric:N').properties(height=250)
    st.altair_chart(chart,use_container_width=True)
    st.caption('Means over seeds 42, 43, 44. Same rows and split indices within each seed. Runtime covers the modeling stage, excluding scheduler/container startup.')
    with st.expander('All nine benchmark runs'):
        st.dataframe(table,use_container_width=True)
else:
    st.info('Run python -m week11.part3.evaluate in the scheduler container to create comparison results.')

st.subheader('Evaluation cases')
cases_path=folder/'case_results.json'
if cases_path.exists():
    suite=json.loads(cases_path.read_text())
    st.metric('Cases passed',f"{suite['passed']} / {suite['total']}")
    st.dataframe([{'Case':c['name'],'Score':c['score'], 'Pass threshold':c['pass_threshold'],'Passed':c['passed']} for c in suite['cases']],use_container_width=True,height=425)
    with st.expander('Scoring criteria and observed behavior'):
        st.json(suite['cases'])
