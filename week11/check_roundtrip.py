"""Re-run a snapshot load and prove retries do not create duplicate database rows."""
import json
from sqlalchemy import create_engine, text
from week11.pipeline import OUT, merge_load
import os

latest = json.loads((OUT / 'latest.json').read_text())['run_id']
folder = OUT / latest
meta = merge_load(str(folder/'customers_clean.csv'), str(folder/'transactions_clean.csv'), latest)
with create_engine(os.environ['WEEK11_DATABASE_URL']).connect() as conn:
    count = conn.execute(text('SELECT COUNT(*) FROM fraud_transactions WHERE pipeline_run=:run'), {'run':latest}).scalar()
assert count == meta['rows']
print(json.dumps({'run_id': latest, 'retry_row_count': count, 'idempotency': 'passed'}))
