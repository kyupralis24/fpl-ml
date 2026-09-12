# FPL-ML

FPL-ML predicts each player's points for the next Fantasy Premier League gameweek and builds a legal 15-player squad, starting XI, and captain recommendation.

The pipeline is deliberately **deadline-safe**: a row containing information through GW *t* is trained against points in GW *t + 1*. Goals, minutes, bonus, and other outcomes from the gameweek being predicted are never model inputs.

## What the pipeline does

1. Fetches completed-GW player points and a schedule snapshot from the official FPL API.
2. Rebuilds per-player lagged form: rolling points/minutes/goals, volatility, and an EMA.
3. Creates a next-GW target (`next_total_points`) and joins target-GW fixture count, home fixtures, and FPL fixture difficulty.
4. Trains LightGBM (or a Random Forest fallback) on labelled historical snapshots.
5. Fetches the upcoming GW's fixture context and produces non-negative player point predictions.
6. Optimizes a £100m, max-three-per-team squad with a valid starting XI and captain.
7. Evaluates completed predictions using MAE, R², and rank correlation.

## Installation

```bash
pip install -r requirements.txt
```

## Weekly workflow

After a GW is complete, the normal workflow is one command:

```bash
python src/run_weekly.py
```

It detects the latest finished FPL GW, evaluates that GW if a prediction exists, then fetches data, rebuilds features, trains, predicts, and selects the next-GW squad, XI, and captain.

To choose the completed GW explicitly:

```bash
python src/run_weekly.py --completed-gw 3
```

Use `--validate` to also run the chronological backtest or `--no-optimize` to produce predictions without a squad recommendation.

The individual commands below remain available for debugging or a partial rerun. The example uses GW3 as the completed week and creates recommendations for GW4.

```bash
# 1. Save GW3 results and its schedule snapshot.
python src/ingest/fetch_gw.py --gw 3

# 2. Rebuild features from every raw GW file.
python src/features/update_features_weekly.py --gw 3

# 3. Train only on rows whose GW-(t+1) points are known.
python src/models/train_model_weekly.py --target_gw 4

# 4. Predict GW4. This requests upcoming GW4 fixture context from FPL.
python src/models/predict_next_gw.py --target_gw 4

# 5. Select a 15-player squad, a valid XI, and a captain.
python src/optimization/select_squad.py --pred data/predictions/predictions_gw4.csv
```

This produces:

- `data/raw/current/gw3_player_stats.csv` and `gw3_fixtures.csv`
- `data/processed/features.csv`
- `models/next_gw_model.pkl`
- `data/predictions/predictions_gw4.csv`
- `data/predictions/optimal_squad_gw4.csv`

After GW4 completes, evaluate the prediction made for it with one command:

```bash
python src/evaluate_weekly.py
```

It fetches official results, calculates the evaluation metrics, and refreshes the performance chart. Use `--gw 4` to choose a GW explicitly. Then repeat the main workflow to prepare GW5. At least two completed consecutive gameweeks are required before a next-GW training label exists.

## Validation

Use a chronological backtest regularly; it retrains only on information that would have been available at each historical deadline.

```bash
python src/evaluate/walk_forward_validate.py --start_gw 4
python src/visualize/model_performance.py
```

`walk_forward_metrics.csv` contains MAE, RMSE, and Spearman rank correlation per gameweek. Rank correlation is particularly useful because FPL decisions depend on ordering players, not just matching every individual score.

## Project layout

```text
data/raw/current/       completed-GW player stats and fixture snapshots
data/processed/         leakage-safe training dataset
data/predictions/       player predictions and optimized squads
data/evaluation/        per-GW joins and walk-forward metrics
models/                 deployed next-GW model artifact
src/                    ingestion, features, models, evaluation, optimization
```

## Current scope

The optimizer handles squad composition, budget, team limits, starting formation, and captaincy. It does not yet account for an existing squad, free transfers, hits, price changes, bench order, vice captain, or chips. Those are the next major planning upgrades.
