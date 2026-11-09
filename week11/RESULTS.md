# Verified results - November 9, 2026

The assignment DAG completed successfully under Docker Airflow 2.10.4 with PostgreSQL 16. Both `scheduled__2026-11-08T07:00:00+00:00` and `assignment_verified` succeeded. The scheduled run's logical date denotes its interval start; its actual execution was November 9. The manual run completed all five tasks, with the two source-cleaning tasks in parallel. Its artifact directory is `outputs/week11/20261109T184725/`.

- Persisted merged transactions: **10,000**.
- Full-table fraud prevalence: **3.06%**.
- Deterministic model sample: **2,000** rows (60 fraud cases).
- Split: 1,200 training, 400 validation, 400 holdout rows.
- Baseline validation F1: **0.047619**.
- Agent loop: **3 cycles**, stopped at the budget; target F1 0.8 was **not reached**.

| Trial | Model | C | Validation F1 |
| --- | --- | --- | --- |
| 1 | logistic_regression | 0.25 | 0.056604 |
| 2 | logistic_regression | 0.5 | 0.056604 |
| 3 | logistic_regression | 1.0 | 0.057143 |

Trial 3 was selected using validation F1 only, then refit on 1,600 development rows. Final sealed holdout (400 rows, including 12 fraud cases):

| Metric | Value |
| --- | --- |
| Precision | 0.031414 |
| Recall | 0.500000 |
| F1 | 0.059113 |
| Accuracy | 0.522500 |

Confusion matrix: TN=203, FP=185, FN=6, TP=6. The model detected half the fraud cases, but generated many false positives. A classifier predicting every transaction as nonfraud would achieve 97% holdout accuracy and zero fraud recall/F1, illustrating why accuracy alone is inadequate. These results do **not** establish a useful deployed fraud detector.

The standalone Part 2 rerun reproduced the same metrics. Scikit-learn emitted convergence warnings because the unscaled logistic models reached their iteration limit; the successful workflow execution does not imply optimizer convergence. The synthetic data, small number of positive cases, unscaled classroom logistic baseline, and limited C search constrain performance. Future experiments could standardize numeric features, compare tree models, and select a probability threshold on validation data. Those changes were not run here. Reusing customers across random partitions also means results do not measure generalization to entirely new customers.

SQL payment-method analysis on the full table found observed fraud rates of UPI 3.83%, Credit Card 3.02%, Wallet 2.80%, and Debit Card 2.57%. These are descriptive synthetic-data associations, not causal effects.

## Verification

- 9 focused tests passed inside the pinned Docker environment.
- Airflow reported no DAG import errors.
- Reloading the manual run snapshot preserved exactly 10,000 rows: retry idempotency passed.
- The dashboard's **Run adapted agents** completed all nodes and saved `dashboard_agent_results.json`; its holdout results agree with the DAG-run output.
- Four original UI screenshots are included under `outputs/week11/screenshots/`.

No LLM/API calls were made for these runs; the classroom heuristic agents executed within LangGraph. No optional Part 3 baseline comparison/evaluation suite is claimed. Canvas submission is separate from publishing this repository.
