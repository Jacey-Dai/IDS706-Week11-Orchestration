"""Task implementations shared by Airflow and the reproducible local runner."""
from pathlib import Path
import json
import os
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/week11"


def clean_table(kind, run_id):
    df = pd.read_csv(ROOT / f"datasets/airflow/{kind}.csv").drop_duplicates()
    key = "customer_id" if kind == "customers" else "transaction_id"
    required = [key] if kind == "customers" else [key, "customer_id", "is_fraud", "amount", "timestamp"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {kind}")
    # Exact duplicates can be removed safely. Conflicting keys require investigation.
    if df[required].isna().any().any() or df[key].duplicated().any():
        raise ValueError(f"Missing required values or conflicting identifiers in {kind}")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().replace("", pd.NA)
    if df[required].isna().any().any() or df[key].duplicated().any():
        raise ValueError(f"Invalid identifiers after normalization in {kind}")
    if kind == "transactions":
        df["amount"] = pd.to_numeric(df["amount"], errors="raise")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        if (df["amount"] < 0).any() or not df.is_fraud.isin([0, 1]).all():
            raise ValueError("Invalid transaction amount or binary label")
        # Synthetic probability can reveal how the label was generated.
        df = df.drop(columns=["fraud_probability"], errors="ignore")
    else:
        for col in ["age", "annual_income", "credit_score"]:
            df[col] = pd.to_numeric(df[col], errors="raise")
        if not df.age.between(0, 120).all():
            raise ValueError("Invalid customer age")
    out = OUT / run_id / f"{kind}_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


def merge_load(customers_path, transactions_path, run_id):
    customers, transactions = pd.read_csv(customers_path), pd.read_csv(transactions_path)
    merged = transactions.merge(customers, on="customer_id", how="left", validate="many_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("Transactions reference missing customers")
    merged = merged.drop(columns="_merge")
    # Each DAG run gets a snapshot key, avoiding collisions across retries/runs.
    merged.insert(0, "pipeline_run", run_id)
    engine = create_engine(os.environ["WEEK11_DATABASE_URL"])
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS week11_runs (run_id TEXT PRIMARY KEY, row_count INTEGER NOT NULL)"))
        exists = conn.execute(text("SELECT to_regclass('public.fraud_transactions')")).scalar()
        if exists:
            conn.execute(text("DELETE FROM fraud_transactions WHERE pipeline_run=:run"), {"run": run_id})
        merged.to_sql("fraud_transactions", conn, if_exists="append", index=False, chunksize=1000, method="multi")
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS fraud_run_transaction ON fraud_transactions (pipeline_run, transaction_id)"))
        conn.execute(text("INSERT INTO week11_runs VALUES (:run,:rows) ON CONFLICT (run_id) DO UPDATE SET row_count=EXCLUDED.row_count"), {"run": run_id, "rows": len(merged)})
    return {"run_id": run_id, "rows": len(merged)}


def read_analyze(meta):
    """Read the persisted snapshot via SQL; baseline uses the same sample as Part 2."""
    from loop_runtime import _fit_score, _heuristic_profile
    engine = create_engine(os.environ["WEEK11_DATABASE_URL"])
    with engine.connect() as conn:
        df = pd.read_sql_query(text("SELECT * FROM fraud_transactions WHERE pipeline_run=:run ORDER BY transaction_id"), conn, params={"run": meta["run_id"]})
        segments = pd.read_sql_query(text("SELECT payment_method, COUNT(*) AS transactions, AVG(is_fraud) AS fraud_rate FROM fraud_transactions WHERE pipeline_run=:run GROUP BY payment_method ORDER BY fraud_rate DESC"), conn, params={"run": meta["run_id"]})
    if len(df) != meta["rows"]:
        raise ValueError("Database roundtrip row count mismatch")
    df = df.drop(columns="pipeline_run")
    # Deterministic random subset keeps classroom experiments responsive.
    sample = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    out = OUT / meta["run_id"]
    path = out / "database_cleaned.csv"
    sample.to_csv(path, index=False)
    segments.to_csv(out / "payment_analysis.csv", index=False)
    profile = _heuristic_profile(sample, "is_fraud")
    spec = {"model_family": "logistic_regression", "hyperparameters": {"C": 1.0, "max_iter": 1000}, "drop_columns": profile["recommended_drops"], "experiment_id": "baseline"}
    baseline = _fit_score(sample, "is_fraud", spec, 42, 0.2, on_holdout=False)
    report = {**meta, "sample_rows": len(sample), "fraud_rate": float(df.is_fraud.mean()), "baseline": baseline, "dataset_path": str(path)}
    (out / "analysis.json").write_text(json.dumps(report, indent=2))
    return report


def run_agents(analysis):
    from student_loop.graph import run_simple_loop
    result = run_simple_loop(dataset_path=analysis["dataset_path"], target_column="is_fraud", target_score=0.8, max_cycles=3, seed=42, model="offline")
    out = OUT / analysis["run_id"]
    (out / "agent_results.json").write_text(json.dumps(result, indent=2, default=str))
    # Publish only after both workflows succeed.
    (OUT / "latest.json").write_text(json.dumps({"run_id": analysis["run_id"]}))
    return {"holdout_metrics": result["holdout_metrics"], "cycles": result["cycles"]}
