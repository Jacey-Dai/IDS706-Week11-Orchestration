"""Week 11: parallel source cleaning -> PostgreSQL -> analysis -> adapted agents."""
from datetime import datetime, timedelta, timezone
from airflow.decorators import dag, task

@dag(dag_id="week11_fraud_pipeline", start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
     schedule="0 7 * * *", catchup=False, max_active_runs=1,
     default_args={"retries": 1, "retry_delay": timedelta(seconds=15)}, tags=["week11", "fraud"])
def week11_fraud_pipeline():
    @task
    def clean_customers(**context):
        from week11.pipeline import clean_table
        return clean_table("customers", context["ts_nodash"])

    @task
    def clean_transactions(**context):
        from week11.pipeline import clean_table
        return clean_table("transactions", context["ts_nodash"])

    @task
    def load_postgres(customers, transactions, **context):
        from week11.pipeline import merge_load
        return merge_load(customers, transactions, context["ts_nodash"])

    @task
    def analyze_database(snapshot):
        from week11.pipeline import read_analyze
        return read_analyze(snapshot)

    @task
    def agent_workflow(analysis):
        from week11.pipeline import run_agents
        return run_agents(analysis)

    customers = clean_customers()
    transactions = clean_transactions()
    loaded = load_postgres(customers, transactions)
    agent_workflow(analyze_database(loaded))

week11_fraud_pipeline()
