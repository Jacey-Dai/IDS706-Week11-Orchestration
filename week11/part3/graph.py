"""A bounded research graph with quality repair, specialist routing, and retries."""
from __future__ import annotations
from typing import TypedDict, Annotated
import operator
import hashlib
import time
import numpy as np
import pandas as pd
from langgraph.graph import StateGraph, START, END
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from loop_runtime import _heuristic_profile


class ResearchState(TypedDict, total=False):
    dataset_path: str
    target: str
    seed: int
    max_attempts: int
    min_val_f1: float
    failure_injection: int
    frame: object
    partitions: object
    profile: dict
    quality: dict
    repair_count: int
    quality_route: str
    family: str
    attempt: int
    retries: int
    retry_limit: int
    retrying: bool
    used_families: list
    trial: dict
    trial_error: str
    trials: list
    best: dict
    best_pipeline: object
    best_validation_probabilities: object
    review_route: str
    threshold: float
    threshold_metrics: dict
    selection_audit: dict
    holdout_metrics: dict
    holdout_calls: int
    stop_reason: str
    steps: Annotated[list, operator.add]


def step(node, **data):
    return {"node": node, **data}


def metrics(y, pred):
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def split_indices(df, target, seed):
    idx = np.arange(len(df))
    dev, hold = train_test_split(idx, test_size=.2, random_state=seed, stratify=df[target])
    train, val = train_test_split(dev, test_size=.25, random_state=seed, stratify=df.iloc[dev][target])
    return {"train": train, "validation": val, "holdout": hold}


def fingerprint(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def quality_agent(s):
    df = s.get("frame")
    if df is None:
        df = pd.read_csv(s['dataset_path'])
    target = s['target']
    error = None
    if target not in df:
        error = "missing_target"
    elif df[target].isna().any() or set(df[target].unique()) != {0, 1}:
        error = "invalid_binary_labels"
    elif df[target].value_counts().min() < 10:
        error = "insufficient_class_support"
    if error:
        return {"quality": {"status": "rejected", "error": error}, "quality_route": "stop",
                "stop_reason": error, "steps": [step('quality_agent', status='rejected', error=error)]}
    duplicates = int(df.duplicated().sum())
    if duplicates:
        route = 'repair' if s.get('repair_count', 0) < 1 else 'stop'
        return {"frame": df, "quality_route": route, "stop_reason": 'quality_repair_required',
                "quality": {"status": "repairable", "duplicate_rows": duplicates},
                "steps": [step('quality_agent', status='repairable', duplicate_rows=duplicates)]}
    if 'transaction_id' in df and (df.transaction_id.isna().any() or df.transaction_id.duplicated().any()):
        return {"quality_route": 'stop', "stop_reason": 'conflicting_transaction_ids',
                "quality": {"status": 'rejected', "error": 'conflicting_transaction_ids'},
                "steps": [step('quality_agent', status='rejected')]}
    profile = _heuristic_profile(df, target)
    return {"frame": df, "quality": {"status": 'passed', "rows": len(df)},
            "quality_route": 'plan', "profile": profile,
            "partitions": split_indices(df, target, s['seed']),
            "steps": [step('quality_agent', status='passed', rows=len(df), drops=profile['recommended_drops'])]}


def repair_agent(s):
    df = s['frame'].drop_duplicates().reset_index(drop=True)
    return {"frame": df, "repair_count": s.get('repair_count', 0) + 1,
            "steps": [step('repair_agent', action='remove_exact_duplicates', rows=len(df))]}


def planner_agent(s):
    if s.get('retrying'):
        family = s['family']
        reason = 'Retry the same specialist after a training error.'
    elif not s.get('used_families'):
        family = 'linear'
        reason = 'Start with a scaled, class-balanced linear model.'
    else:
        family = 'tree'
        reason = 'Validation F1 is below target; test a nonlinear specialist.'
    return {"family": family, "steps": [step('planner_agent', selected=family, reason=reason)]}


def train_specialist(s, family):
    attempt = s.get('attempt', 0) + 1
    node = 'linear_specialist' if family == 'linear' else 'tree_specialist'
    start = time.perf_counter()
    try:
        if attempt <= s.get('failure_injection', 0):
            raise RuntimeError('Injected training failure for recovery evaluation')
        df, target = s['frame'], s['target']
        drops = s['profile']['recommended_drops']
        features = df.drop(columns=[target, *drops], errors='ignore')
        if features.empty:
            raise ValueError('No usable model features')
        numeric = list(features.select_dtypes('number').columns)
        categorical = [c for c in features if c not in numeric]
        pre = ColumnTransformer([
            ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric),
            ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                              ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical)])
        model = (LogisticRegression(C=1., max_iter=1500, class_weight='balanced', random_state=s['seed'])
                 if family == 'linear' else RandomForestClassifier(n_estimators=120, max_depth=8,
                     min_samples_leaf=3, class_weight='balanced', random_state=s['seed'], n_jobs=1))
        pipe = Pipeline([('preprocess', pre), ('model', model)])
        parts = s['partitions']
        pipe.fit(features.iloc[parts['train']], df[target].iloc[parts['train']])
        probability = pipe.predict_proba(features.iloc[parts['validation']])[:, 1]
        score = metrics(df[target].iloc[parts['validation']], probability >= .5)
        trial = {"attempt": attempt, "family": family, "status": 'succeeded',
                 "validation_metrics": score, "duration_seconds": time.perf_counter() - start,
                 "feature_columns": list(features.columns), "dropped_columns": drops}
        patch = {"attempt": attempt, "trial": trial, "trial_error": '',
                 "used_families": list(dict.fromkeys([*s.get('used_families', []), family])),
                 "trials": [*s.get('trials', []), trial],
                 "steps": [step(node, status='succeeded', attempt=attempt, validation_f1=score['f1'])]}
        if not s.get('best') or score['f1'] > s['best']['validation_metrics']['f1']:
            patch.update(best=trial, best_pipeline=pipe, best_validation_probabilities=probability)
        return patch
    except Exception as exc:
        trial = {"attempt": attempt, "family": family, "status": 'failed', "error": str(exc)}
        return {"attempt": attempt, "trial": trial, "trial_error": str(exc),
                "trials": [*s.get('trials', []), trial],
                "steps": [step(node, status='failed', attempt=attempt, error=str(exc))]}


def linear_specialist(s):
    return train_specialist(s, 'linear')


def tree_specialist(s):
    return train_specialist(s, 'tree')


def reviewer_agent(s):
    if s.get('trial_error'):
        if s.get('retries', 0) < s['retry_limit'] and s['attempt'] < s['max_attempts']:
            route, reason = 'retry', 'Training error; retry budget remains.'
        elif s.get('best'):
            route, reason = 'threshold', 'Training budget exhausted; retain the best successful trial.'
        else:
            route, reason = 'stop', 'No successful model within the retry budget.'
    elif s['best']['validation_metrics']['f1'] >= s['min_val_f1']:
        route, reason = 'threshold', 'Validation target reached.'
    elif s['attempt'] < s['max_attempts'] and 'tree' not in s.get('used_families', []):
        route, reason = 'plan', 'Try the alternative specialist.'
    else:
        route, reason = 'threshold', 'Experiment budget reached or both specialists evaluated.'
    return {"review_route": route, "retrying": False,
            "stop_reason": 'training_failed' if route == 'stop' else reason,
            "steps": [step('reviewer_agent', route=route, reason=reason)]}


def recovery_agent(s):
    return {"retries": s.get('retries', 0) + 1, "retrying": True,
            "steps": [step('recovery_agent', action='retry_same_specialist', retry=s.get('retries', 0)+1)]}


def threshold_agent(s):
    # No holdout labels or probabilities are read by this agent.
    val = s['partitions']['validation']
    y = s['frame'][s['target']].iloc[val]
    prob = s['best_validation_probabilities']
    candidates = np.linspace(.05, .95, 19)
    threshold = float(max(candidates, key=lambda t: (f1_score(y, prob >= t, zero_division=0), t)))
    score = metrics(y, prob >= threshold)
    audit = {"selection_split": 'validation', "selection_rows": len(val),
             "selection_indices_sha256": fingerprint(val),
             "holdout_indices_sha256": fingerprint(s['partitions']['holdout']),
             "overlap_rows": int(len(set(val) & set(s['partitions']['holdout'])))}
    return {"threshold": threshold, "threshold_metrics": score, "selection_audit": audit,
            "steps": [step('threshold_agent', threshold=threshold, validation_f1=score['f1'], **audit)]}


def holdout_agent(s):
    df = s['frame']
    parts = s['partitions']
    dev = np.concatenate([parts['train'], parts['validation']])
    hold = parts['holdout']
    features = df.drop(columns=[s['target'], *s['profile']['recommended_drops']], errors='ignore')
    pipe = s['best_pipeline']
    pipe.fit(features.iloc[dev], df[s['target']].iloc[dev])
    prob = pipe.predict_proba(features.iloc[hold])[:, 1]
    score = metrics(df[s['target']].iloc[hold], prob >= s['threshold'])
    return {"holdout_metrics": {**score, "n_holdout": len(hold)},
            "holdout_calls": s.get('holdout_calls', 0)+1,
            "steps": [step('holdout_agent', status='completed', metrics=score)]}


def build_graph():
    g = StateGraph(ResearchState)
    for name, fn in [('quality_agent',quality_agent),('repair_agent',repair_agent),
                     ('planner_agent',planner_agent),('linear_specialist',linear_specialist),
                     ('tree_specialist',tree_specialist),('reviewer_agent',reviewer_agent),
                     ('recovery_agent',recovery_agent),('threshold_agent',threshold_agent),('holdout_agent',holdout_agent)]:
        g.add_node(name, fn)
    g.add_edge(START, 'quality_agent')
    g.add_conditional_edges('quality_agent', lambda s:s['quality_route'], {'repair':'repair_agent','plan':'planner_agent','stop':END})
    g.add_edge('repair_agent','quality_agent')
    g.add_conditional_edges('planner_agent', lambda s:s['family'], {'linear':'linear_specialist','tree':'tree_specialist'})
    g.add_edge('linear_specialist','reviewer_agent')
    g.add_edge('tree_specialist','reviewer_agent')
    g.add_conditional_edges('reviewer_agent', lambda s:s['review_route'],
                            {'retry':'recovery_agent','plan':'planner_agent','threshold':'threshold_agent','stop':END})
    g.add_edge('recovery_agent','planner_agent')
    g.add_edge('threshold_agent','holdout_agent')
    g.add_edge('holdout_agent',END)
    return g.compile()


def initial_state(dataset_path, seed=42, max_attempts=3, min_val_f1=.8, failure_injection=0, retry_limit=1):
    if not 1 <= max_attempts <= 6 or not 0 <= retry_limit <= 3:
        raise ValueError('Invalid experiment or retry budget')
    return {"dataset_path": str(dataset_path), "target": 'is_fraud', "seed": seed,
            "max_attempts": max_attempts, "min_val_f1": min_val_f1, "failure_injection":failure_injection,
            "retry_limit":retry_limit, "attempt":0, "retries":0, "repair_count":0,
            "trials":[], "used_families":[], "steps":[], "holdout_calls":0}


def summarize(s):
    keys = ['dataset_path','seed','quality','repair_count','attempt','retries','trials','best',
            'threshold','threshold_metrics','selection_audit','holdout_metrics','holdout_calls','stop_reason','steps']
    result = {k:s.get(k) for k in keys}
    parts = s.get('partitions')
    if parts:
        result['split_fingerprints'] = {k:fingerprint(v) for k,v in parts.items()}
    return result


def run(dataset_path, **kwargs):
    start = time.perf_counter()
    state = build_graph().invoke(initial_state(dataset_path, **kwargs), config={'recursion_limit':80})
    result = summarize(state)
    result['duration_seconds'] = time.perf_counter() - start
    return result
