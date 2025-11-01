# Fantasy Premier League Prediction Workflow ⚽📊

This repo helps automate weekly FPL data ingestion, model training, predictions, and optimal squad selection.  
Follow the steps below each gameweek.

---

## Weekly Workflow Checklist ✅

Run the following sequence **every gameweek**.  
Replace `2` / `3` with the actual gameweek numbers you are working with.

### Steps (run one after another):

```bash
# 1️⃣ Fetch real data for GW2
python src/ingest/fetch_gw.py --gw 2

# 2️⃣ Update enhanced features (rolling averages, EMA)
python src/features/update_features_weekly.py --gw 2

# 3️⃣ Retrain stacked model using data up to GW2
python src/models/train_model_weekly.py --target_gw 3

# 4️⃣ Predict GW3 player points
python src/models/predict_next_gw.py --target_gw 3

# 5️⃣ Optimize and select best squad for GW3
python src/optimization/select_squad.py --pred data/predictions/predictions_gw3.csv

# 6️⃣ After GW3 completes: evaluate model accuracy
python src/evaluate/evaluate_model_weekly.py --gw 3

# 7️⃣ Visualize long-term model performance
python src/visualize/model_performance.py