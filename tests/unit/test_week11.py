"""Meaningful checks for source integrity, leakage control, and loop routing."""
import pandas as pd
import pytest
from student_loop.graph import build_graph, node_data_quality, route_after_judge
from week11 import pipeline


def test_quality_rejects_duplicates(tmp_path):
    path = tmp_path / 'bad.csv'
    pd.DataFrame({'transaction_id': ['a'] * 20, 'is_fraud': [0, 1] * 10}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        node_data_quality({'dataset_path': str(path), 'target_column': 'is_fraud'})


def test_sources_and_leakage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'OUT', tmp_path)
    customers = pd.read_csv(pipeline.clean_table('customers', 'test'))
    transactions = pd.read_csv(pipeline.clean_table('transactions', 'test'))
    assert customers.customer_id.is_unique
    assert transactions.transaction_id.is_unique
    assert 'fraud_probability' not in transactions
    assert set(transactions.customer_id) <= set(customers.customer_id)


def test_quality_is_first_and_router_obeys_budget():
    graph = build_graph().get_graph()
    assert any(e.source == '__start__' and e.target == 'data_quality' for e in graph.edges)
    assert route_after_judge({'verdict': {'action':'revise'}, 'cycle':1, 'max_cycles':3}) == 'memory'
    assert route_after_judge({'verdict': {'action':'revise'}, 'cycle':3, 'max_cycles':3}) == 'holdout'
    assert route_after_judge({'verdict': {'action':'accept'}, 'cycle':1, 'max_cycles':3}) == 'holdout'
