"""Run explicit contract cases and compare three workflows on identical splits."""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from week11.pipeline import OUT
from week11.part3.graph import run, split_indices, fingerprint
from loop_runtime import _fit_score, _heuristic_profile
from mini_loop.graph import run_simple_loop

RESULTS = OUT / 'part3'


def check(name, input_description, expected, criteria, threshold, observed, passed):
    return {'name':name,'input':input_description,'expected_behavior':expected,
            'scoring_criteria':criteria,'pass_threshold':threshold,'observed':observed,
            'score':int(bool(passed)),'passed':bool(passed)}


def evaluate_cases(path):
    RESULTS.mkdir(parents=True, exist_ok=True)
    fixtures = RESULTS / 'fixtures'
    fixtures.mkdir(exist_ok=True)
    df = pd.read_csv(path)
    cases=[]
    def execute(name, frame=None, **kwargs):
        target=path
        if frame is not None:
            target=fixtures/f'{name}.csv'
            frame.to_csv(target,index=False)
        return run(target, **kwargs)
    normal=execute('clean')
    hold=normal.get('holdout_metrics') or {}
    cases.append(check('clean_input', 'Part 1 database-exported sample, seed 42',
        'Quality passes; selected model produces finite binary classification metrics.',
        'quality=passed, one holdout call, every scalar metric in [0,1]',1,
        {'quality':normal['quality'],'holdout_calls':normal['holdout_calls'],'metrics':hold},
        normal['quality']['status']=='passed' and normal['holdout_calls']==1 and all(np.isfinite(hold.get(k,np.nan)) and 0<=hold[k]<=1 for k in ['f1','recall','precision','accuracy'])))
    audit=normal['selection_audit']
    cases.append(check('sealed_holdout','Same clean input',
        'Threshold uses only validation rows; holdout remains separate and is evaluated once.',
        'selection_split=validation, overlap=0, validation hash matches partition, holdout_calls=1',1,
        audit,audit['selection_split']=='validation' and audit['overlap_rows']==0 and
        audit['selection_indices_sha256']==normal['split_fingerprints']['validation'] and normal['holdout_calls']==1))
    dupe=execute('exact_duplicates',pd.concat([df,df.iloc[:5]],ignore_index=True))
    cases.append(check('exact_duplicate_repair','Five exact duplicate records appended',
        'Repair once, revalidate, then train on the original row count.',
        'repair_count=1, quality.rows equals original rows, one holdout evaluation',1,
        {'repairs':dupe['repair_count'],'quality':dupe['quality'],'holdout_calls':dupe['holdout_calls']},
        dupe['repair_count']==1 and dupe['quality'].get('rows')==len(df) and dupe['holdout_calls']==1))
    invalid=df.copy();invalid.loc[0,'is_fraud']=2
    bad=execute('invalid_labels',invalid)
    cases.append(check('invalid_labels','One target label replaced by 2',
        'Reject before training and holdout.', 'attempt=0 and holdout_calls=0 and explicit rejection',1,
        {'quality':bad['quality'],'attempts':bad['attempt']},bad['quality']['status']=='rejected' and bad['attempt']==0 and bad['holdout_calls']==0))
    missing=execute('missing_target',df.drop(columns='is_fraud'))
    cases.append(check('missing_target','Target column removed','Reject with missing_target.',
        'stop_reason=missing_target, no holdout evaluation',1,missing['stop_reason'],
        missing['stop_reason']=='missing_target' and missing['holdout_calls']==0))
    leak=df.copy();leak['fraud_probability']=leak.is_fraud
    leak_result=execute('leakage_guard',leak)
    features=leak_result['best']['feature_columns']
    cases.append(check('leakage_guard','A fraud_probability column equal to the target is added',
        'Drop label-generation probability and identifier columns before all experiments.',
        'No fraud_probability or *_id column appears in any trial feature list',1,
        {'features':features,'drops':leak_result['best']['dropped_columns']},
        all('fraud_probability' not in t.get('feature_columns',[]) and not any(c.endswith('_id') for c in t.get('feature_columns',[])) for t in leak_result['trials'])))
    transient=execute('transient_failure',failure_injection=1)
    cases.append(check('transient_recovery','First specialist training attempt raises a controlled RuntimeError',
        'Visit recovery agent once, retry same specialist, and complete.',
        'retries=1, first trial failed, second succeeded with same family, one holdout',1,
        {'retries':transient['retries'],'trials':[{k:t.get(k) for k in ['attempt','family','status']} for t in transient['trials']]},
        transient['retries']==1 and transient['trials'][0]['status']=='failed' and transient['trials'][1]['status']=='succeeded' and
        transient['trials'][0]['family']==transient['trials'][1]['family'] and transient['holdout_calls']==1))
    persistent=execute('persistent_failure',failure_injection=99)
    cases.append(check('persistent_failure','Every training attempt raises a controlled RuntimeError',
        'Stop after the retry limit; do not score an unfitted model.',
        'two attempts, one retry, training_failed, zero holdout calls',1,
        {'attempts':persistent['attempt'],'retries':persistent['retries'],'stop_reason':persistent['stop_reason']},
        persistent['attempt']==2 and persistent['retries']==1 and persistent['stop_reason']=='training_failed' and persistent['holdout_calls']==0))
    budget=execute('attempt_budget',max_attempts=1,min_val_f1=1.)
    cases.append(check('attempt_budget','One-attempt budget, validation target 1.0',
        'Finalize the available model without another experiment.',
        'attempt=1, exactly one trial, one holdout call',1,
        {'attempts':budget['attempt'],'holdout_calls':budget['holdout_calls']},
        budget['attempt']==1 and len(budget['trials'])==1 and budget['holdout_calls']==1))
    adaptive=execute('specialist_switch',min_val_f1=1.)
    families=[t['family'] for t in adaptive['trials'] if t['status']=='succeeded']
    cases.append(check('specialist_switch','Clean sample with target validation F1 1.0',
        'Below-target linear model routes to the tree specialist.',
        'successful trial families are [linear, tree]',1,families,families==['linear','tree']))
    easy_x,easy_y=make_classification(n_samples=800,n_features=8,n_informative=6,n_redundant=0,
        weights=[.7,.3],class_sep=2.5,flip_y=0.,random_state=123)
    easy=pd.DataFrame(easy_x,columns=[f'feature_{i}' for i in range(8)]);easy['is_fraud']=easy_y
    easy_result=execute('known_signal',easy)
    f1=easy_result['holdout_metrics']['f1']
    cases.append(check('known_signal','Fixed synthetic fixture: 800 rows, class_sep=2.5, seed 123',
        'Recover a clear predictive signal on sealed holdout.', 'holdout F1 >= 0.75',.75,{'holdout_f1':f1},f1>=.75))
    cases[-1]['score'] = f1
    (RESULTS/'case_results.json').write_text(json.dumps({'cases':cases,'passed':sum(c['passed'] for c in cases),'total':len(cases)},indent=2))
    (RESULTS/'adaptive_example.json').write_text(json.dumps(normal,indent=2))
    (RESULTS/'recovery_example.json').write_text(json.dumps(transient,indent=2))
    return cases


def compare(path):
    df=pd.read_csv(path)
    profile=_heuristic_profile(df,'is_fraud')
    rows=[]
    for seed in [42,43,44]:
        partitions=split_indices(df,'is_fraud',seed)
        hashes={k:fingerprint(v) for k,v in partitions.items()}
        spec={'experiment_id':'fixed','model_family':'logistic_regression',
              'hyperparameters':{'C':1.,'max_iter':1000},'drop_columns':profile['recommended_drops']}
        start=time.perf_counter()
        validation=_fit_score(df,'is_fraud',spec,seed,.2,on_holdout=False)
        final=_fit_score(df,'is_fraud',spec,seed,.2,on_holdout=True)
        rows.append({'workflow':'Fixed Airflow baseline','seed':seed,'attempts':1,'retries':0,
            'duration_seconds':time.perf_counter()-start,'validation_f1':validation['validation_metrics']['f1'],
            'holdout_metrics':final,'split_fingerprints':hashes})
        start=time.perf_counter()
        simple=run_simple_loop(dataset_path=str(path),target_column='is_fraud',target_score=.8,max_cycles=3,seed=seed,model='offline')
        trials=[s for s in simple['steps'] if s['node']=='ml_engineer']
        rows.append({'workflow':'Classroom simple agents','seed':seed,'attempts':len(trials),'retries':sum(bool(t.get('repaired')) for t in trials),
            'duration_seconds':time.perf_counter()-start,'validation_f1':simple['best_val_f1'],
            'holdout_metrics':simple['holdout_metrics'],'split_fingerprints':hashes})
        adaptive=run(path,seed=seed)
        assert adaptive['split_fingerprints']==hashes
        rows.append({'workflow':'Adaptive specialists','seed':seed,'attempts':adaptive['attempt'],'retries':adaptive['retries'],
            'duration_seconds':adaptive['duration_seconds'],'validation_f1':adaptive['threshold_metrics']['f1'],
            'holdout_metrics':adaptive['holdout_metrics'],'split_fingerprints':hashes,'threshold':adaptive['threshold']})
    (RESULTS/'comparison.json').write_text(json.dumps(rows,indent=2))
    table=pd.DataFrame([{'workflow':r['workflow'],'seed':r['seed'],'attempts':r['attempts'],'seconds':r['duration_seconds'],
                       **{k:r['holdout_metrics'][k] for k in ['precision','recall','f1','accuracy']}} for r in rows])
    table.to_csv(RESULTS/'comparison.csv',index=False)
    summary=table.groupby('workflow').agg({'f1':['mean','std'],'precision':'mean','recall':'mean','accuracy':'mean','seconds':'mean','attempts':'mean'})
    summary.to_csv(RESULTS/'comparison_summary.csv')
    return rows


def main():
    latest=json.loads((OUT/'latest.json').read_text())['run_id']
    path=OUT/latest/'database_cleaned.csv'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        cases=evaluate_cases(path)
        rows=compare(path)
    (RESULTS/'warnings.json').write_text(json.dumps([{'category':w.category.__name__,'message':str(w.message)} for w in caught],indent=2))
    summary={'passed':sum(c['passed'] for c in cases),'total':len(cases),'comparison_runs':len(rows),
             'warning_count':len(caught),'input':str(path)}
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if not all(c['passed'] for c in cases):
        raise SystemExit('Evaluation cases failed; inspect case_results.json')


if __name__=='__main__':
    main()
