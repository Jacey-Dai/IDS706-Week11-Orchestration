# Adapted Week 11 agent workflow

The node bodies and graph wiring are implemented in [graph.py](graph.py), extending the classroom reference in [mini_loop/graph.py](../mini_loop/graph.py). The new `data_quality` node validates the database-exported dataset before profiling; the loop trains and judges up to three validation experiments, then evaluates the selected model once on holdout.

Follow the [root run instructions](../README.md#run-both-workflows). The assignment's live graph is the Streamlit dashboard at http://localhost:8501. The classroom React frontend is preserved as reference and is not required by the assignment Docker stack.

```bash
docker compose -f week11/compose.yml exec scheduler python -m student_loop.run_cli --status
docker compose -f week11/compose.yml exec scheduler python -m week11.run_agents
```

The shared training/LLM helpers live in [loop_runtime.py](../loop_runtime.py). The verified runs use the classroom planning and review rules.
