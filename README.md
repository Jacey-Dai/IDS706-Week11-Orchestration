# Week 11 · Weather and bikeshare demand in Washington, DC

Jacey Dai · IDS 706 · Parts 1–3

Can airport weather measurements improve daily rental estimates over a calendar-only model? This project joins Capital Bikeshare counts with NOAA weather and tests the models on the final 147 days of the sample. The analysis uses observed weather from each day, so it measures retrospective demand estimation. A forecast would require weather information available before the rental day.

## Run the project

From the repository root, with Docker running:

```bash
docker compose -f bikeshare/compose.yml up --build -d
```

1. Open [Airflow](http://localhost:8089) and sign in with `admin` / `admin`.
2. Unpause and trigger `dc_bikeshare_demand`. Wait for all five tasks to finish.
3. Open the [dashboard](http://localhost:8502) and click **Run workflow**. Select **Simulate a transient training failure** to see the retry path.

To rerun the evaluation suite from the SQL export:

```bash
docker compose -f bikeshare/compose.yml exec scheduler python -m bikeshare.research
```

## Part 1 · Ingest, clean, and analyze

The two independently sourced datasets are [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) and [NOAA GHCN-Daily](https://www.ncei.noaa.gov/pub/data/ghcn/daily/). See [data sources and extraction notes](bikeshare/data/README.md).

The [new Airflow DAG](bikeshare/dags/demand.py) runs daily at 07:00 UTC and cleans the two sources in parallel. Each branch standardizes dates, drops exact duplicates, and rejects conflicting keys or invalid values. The load task joins one-to-one by date and replaces the PostgreSQL snapshot in a transaction. The analysis task reads the stored rows back, checks the count, and computes the wet/dry comparison in SQL. The joined dataset contains **731 days**.

## Part 2 · Adapted research loop

The [LangGraph implementation](bikeshare/research.py) prepares the joined data, plans trials, trains models, reviews validation MAE, and scores the selected model on a chronological holdout. The split uses 438 training days, 146 validation days, and 147 holdout days. After selection the model refits on the first 584 days. The live dashboard shows the actual compiled graph, node execution order, and final score.

## Part 3 · Adaptive decisions and evaluation

The reviewer tries a random forest when adding weather improves validation MAE by less than 10%. The recovery node retries one training failure and stops if the retry also fails. The [evaluation cases](bikeshare/results/evaluation.json) record inputs, expected behavior, criteria, scores, and pass thresholds. All four cases passed, including the controlled recovery.

| Workflow | Selection procedure | Holdout MAE (rentals/day) |
| --- | --- | ---: |
| Fixed Airflow baseline | One predefined calendar ridge model | 1134.5 |
| Simple classroom-style agent baseline | Try calendar ridge and weather ridge, then select by validation MAE | 880.2 |
| Adaptive workflow | Add a forest trial when the weather ridge gain falls short, with one retry available | 880.2 |

Both agent workflows selected weather ridge and reduced holdout MAE by **22.4%** compared with the calendar-only model. Weather ridge improved validation MAE by 5.0%, below the 10% threshold, so the adaptive loop also tried a random forest. Its validation MAE was higher, so the loop kept ridge. The adaptive workflow matched the simple loop's holdout error and successfully recovered from the simulated error. These results come from one chronological split. The fixed baseline uses a predefined model, and the simple agent baseline applies the classroom loop structure to this regression task.

## Execution evidence

- [Successful Airflow execution](bikeshare/results/01_airflow_success.jpg)
- [Airflow DAG graph](bikeshare/results/02_dag_graph.jpg)
- [Live agent graph and trace](bikeshare/results/03_live_graph.jpg)
- [Holdout model evaluation](bikeshare/results/04_model_evaluation.jpg)
- [SQL wet/dry summary](bikeshare/results/weather_summary.csv)
- [Full evaluation results](bikeshare/results/evaluation.json)

The [earlier classroom-data submission](week11/fraud-submission.md) is archived for reference.
