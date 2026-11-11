# Part 3 results

The suite ran inside the same Docker environment as Parts 1 and 2, using the 2,000-row PostgreSQL export from `20261109T184725`. Evaluation completed on November 9, 2026.

## Coordination and intermediate checks

All **11 of 11 cases passed**. The clean-data run trained the linear and tree specialists, then finalized the best validation model. The duplicate scenario removed exactly five repeated rows, revalidated the data, and completed. The transient-failure scenario used three attempts: one failed linear attempt, one successful linear retry, and one tree experiment. Persistent failures stopped after two attempts and one retry, with no holdout scoring. Invalid labels and a missing target stopped before training.

The threshold agent recorded validation-only selection, zero validation/holdout overlap, and a partition fingerprint matching the validation split. Each successful execution recorded exactly one holdout call. All experiment feature lists excluded `fraud_probability` and identifier columns. The known-signal fixture achieved holdout F1 **0.8913**, above its predeclared 0.75 pass threshold.

These cases verify input handling, routing, budget enforcement, recovery, and final metric validity. They do not establish performance under every possible failure condition. Runtime errors were injected deliberately for the recovery cases.

## Comparison with classroom baselines

Means over three split seeds; F1 variation is the sample standard deviation across seeds.

| Workflow | Precision | Recall | F1 mean ± SD | Accuracy | Training attempts | Modeling seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed Airflow baseline | 0.0249 | 0.3889 | 0.0468 ± 0.0150 | 0.5258 | 1.0 | 1.853 |
| Classroom simple agents | 0.0260 | 0.4167 | 0.0489 ± 0.0091 | 0.5142 | 3.0 | 1.246 |
| Adaptive specialists | 0.0288 | 0.3056 | 0.0511 ± 0.0196 | 0.6792 | 2.0 | 0.501 |

All workflows received the same sample and partition indices within each seed. The timing covers training, selection, and final scoring only; it excludes scheduler startup, ingestion, and database transfer. The fixed Airflow baseline runs its analysis task's modeling procedure directly; the simple-agent baseline runs the classroom `mini_loop` graph.

The adaptive mean F1 was slightly higher, but the difference was small compared with variation across splits. Its average recall was lower than both baselines. Seed 42 illustrates the trade-off: the adaptive model's holdout F1 was 0.0377, below the fixed model's 0.0609 and simple agents' 0.0591. The adaptive procedure is therefore not uniformly better. Overall fraud prediction remains weak on this sample.

| Seed | Workflow | Holdout F1 | Holdout recall |
| --- | --- | --- | --- |
| 42 | Fixed Airflow baseline | 0.060914 | 0.500000 |
| 42 | Classroom simple agents | 0.059113 | 0.500000 |
| 42 | Adaptive specialists | 0.037736 | 0.083333 |
| 43 | Fixed Airflow baseline | 0.048309 | 0.416667 |
| 43 | Classroom simple agents | 0.045872 | 0.416667 |
| 43 | Adaptive specialists | 0.073620 | 0.500000 |
| 44 | Fixed Airflow baseline | 0.031088 | 0.250000 |
| 44 | Classroom simple agents | 0.041667 | 0.333333 |
| 44 | Adaptive specialists | 0.041885 | 0.333333 |

## Interpretation

The measured coordination benefit is that failures and quality issues take explicit recovery or rejection paths, while ordinary runs use two model specialists and stop within budget. The clean adaptive run selected a scaled logistic model and a validation threshold of 0.40; after refitting on development data, its seed-42 holdout matrix was TN=348, FP=40, FN=11, TP=1.

The experiment changes several things at once: numeric scaling, model-family choice, routing, and threshold selection. It cannot attribute the small mean F1 difference to adaptive coordination alone. With only 12 positive holdout cases per split, metrics are sensitive to individual predictions. No significance claim is made.

Captured convergence warnings came from the unscaled classroom/fixed logistic procedures; they are preserved in `warnings.json`. The adaptive numeric pipeline standardizes features and uses a larger iteration budget. The original baseline configuration remains unchanged for comparison.

## Reproduce and inspect

```bash
docker compose -f week11/compose.yml exec scheduler python -m week11.part3.evaluate
```

- `outputs/week11/part3/case_results.json`: inputs, expected outcomes, scoring, thresholds, and observations.
- `outputs/week11/part3/comparison.json`: all nine benchmark runs and split fingerprints.
- `outputs/week11/part3/adaptive_example.json`: normal execution trace.
- `outputs/week11/part3/recovery_example.json`: controlled-failure recovery trace.
- `outputs/week11/part3/screenshots/`: live graph, comparison, and evaluation evidence.

The dashboard is available at http://localhost:8501/Adaptive_Research after starting the existing Compose stack.
