from datetime import datetime, timezone

from airflow.decorators import dag, task

from bikeshare import pipeline
from bikeshare.research import evaluate


@dag(dag_id="dc_bikeshare_demand", schedule="0 7 * * *",
     start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), catchup=False,
     max_active_runs=1, default_args={"retries": 1})
def demand():
    @task
    def clean_rentals():
        return pipeline.clean("rentals").to_dict("records")

    @task
    def clean_weather():
        return pipeline.clean("weather").to_dict("records")

    @task
    def load_postgres(rentals, weather):
        import pandas as pd

        return pipeline.load(pd.DataFrame(rentals), pd.DataFrame(weather))

    @task
    def analyze_postgres(rows):
        if pipeline.analyze() != rows:
            raise ValueError("Database readback count differs from load")

    @task
    def evaluate_models():
        evaluate()

    analyze_postgres(load_postgres(clean_rentals(), clean_weather())) >> evaluate_models()


demand()
