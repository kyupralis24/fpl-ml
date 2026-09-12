# Weekly FPL update runbook

Run this after the latest gameweek is fully complete and before the next FPL deadline. Replace `N` with the completed GW and `NEXT` with `N + 1`.

For example: after GW3, use `N=3` and `NEXT=4`.

## The easy weekly command

```bash
python src/run_weekly.py
```

This detects the latest completed GW automatically, then runs the full sequence below. It evaluates that GW when its saved prediction exists, fetches final official points, rebuilds features, trains, predicts the next GW, and selects the squad, XI, and captain.

If FPL has not yet marked the GW as finished, specify it yourself:

```bash
python src/run_weekly.py --completed-gw N
```

Optional flags:

- `--validate` also runs the chronological walk-forward backtest.
- `--no-optimize` produces player predictions but skips the squad recommendation.

The remaining sections document the individual steps for troubleshooting or partial reruns.

## 1. Ingest official completed-GW data

```bash
python src/ingest/fetch_gw.py --gw N
```

What it does:

- Retrieves official player totals for GW `N` from FPL.
- Saves `data/raw/current/gwN_player_stats.csv`.
- Saves `data/raw/gwN_actual_points.csv` for later evaluation.
- Saves `data/raw/current/gwN_fixtures.csv`, a compact fixture snapshot that records fixture count, home fixtures, and FDR per team.

Do this only once the GW is finished; live totals can still change while matches are in progress.

## 2. Rebuild historical features and next-GW labels

```bash
python src/features/update_features_weekly.py --gw N
```

What it does:

- Reads every saved `gw*_player_stats.csv` file, so features are reproducible from raw data.
- Builds only lagged player-form features: rolling points/minutes/goals, rolling volatility, and EMA form.
- Creates `next_total_points`: GW `t + 1` points assigned to the GW `t` snapshot.
- Attaches the *target* GW's fixture count, home-fixture count, and FDR.
- Writes `data/processed/features.csv`.

Important: same-GW goals, minutes, bonus, and other match outcomes are not model features. This prevents the model from seeing information unavailable at the prediction deadline.

## 3. Train the next-GW model

```bash
python src/models/train_model_weekly.py --target_gw NEXT
```

What it does:

- Trains only on labelled rows from GWs before `NEXT`.
- Uses LightGBM when available, otherwise a Random Forest.
- Records the exact feature columns with the model in `models/next_gw_model.pkl`.

At least two consecutive completed gameweeks are required before there are labelled training rows. More historical seasons will make early-season predictions substantially more reliable.

## 4. Generate target-GW predictions

```bash
python src/models/predict_next_gw.py --target_gw NEXT
```

What it does:

- Starts from each player's latest completed-GW form snapshot.
- Requests the upcoming GW's fixture list from the FPL API.
- Adds target fixture count, home-fixture count, and FDR, including blanks and doubles.
- Writes `data/predictions/predictions_gwNEXT.csv`.

Run this close enough to the deadline that you have current fixture information, but allow time to review the result. It does not yet model late injury/news updates.

## 5. Optimize the squad and starting XI

```bash
python src/optimization/select_squad.py --pred data/predictions/predictions_gwNEXT.csv
```

What it does:

- Selects 15 players under the £100m budget, position quotas, and three-player-per-team limit.
- Selects a legal 11-player formation from that squad.
- Selects a captain from the starting XI.
- Writes `data/predictions/optimal_squad_gwNEXT.csv`.

This is a fresh-squad optimiser. It does not yet consider your existing team, bank, free transfers, transfer hits, chips, bench order, or vice captain.

## 6. Evaluate once the predicted GW finishes

After GW `NEXT` completes, run:

```bash
python src/evaluate_weekly.py
```

This detects the latest finished GW, fetches official results, reports MAE/R²/Spearman rank correlation, and refreshes the performance chart. Use `python src/evaluate_weekly.py --gw NEXT` if you need a specific GW.

This joins predictions to official points by stable FPL player ID and reports MAE, R², and Spearman rank correlation. It saves the joined data to `data/evaluation/eval_gwNEXT.csv`.

## 7. Run periodic chronological validation

Run this every few weeks, or after a modelling change:

```bash
python src/evaluate/walk_forward_validate.py --start_gw 4
python src/visualize/model_performance.py
```

The walk-forward backtest retrains for each historical GW using only previously available data. Use it to compare changes against the current model rather than trusting a random train/test split.

## Compact weekly command sequence

```bash
python src/run_weekly.py
```
