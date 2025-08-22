# Fantasy Premier League Prediction Workflow ⚽📊

This repo helps automate weekly FPL data ingestion, model training, predictions, and optimal squad selection.  
Follow the steps below each gameweek.

---

## Weekly Workflow Checklist ✅

Run the following sequence **every gameweek**.  
Replace `2` / `3` with the actual gameweek numbers you are working with.

### Steps (run one after another):

```bash
# 1) Fetch real results of previous GW (example: GW2)
python src/ingest/fetch_gw.py --gw 2

# 2) Update features with those results
python src/features/update_features_weekly.py --gw 2

# 3) Retrain model using data up to that GW (train on GW < 3)
python src/models/train_model_weekly.py --target_gw 3

# 4) Predict performance for the upcoming GW (GW3)
python src/models/predict_next_gw.py --target_gw 3

# 5) Optimize and select best squad for the upcoming GW
python src/optimization/select_squad.py --pred data/predictions/predictions_gw3.csv