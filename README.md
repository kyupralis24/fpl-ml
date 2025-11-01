
# FPL-ML: Fantasy Premier League Machine Learning Pipeline

A machine learning and optimization pipeline for **Fantasy Premier League (FPL)** that predicts weekly player performance and recommends the **optimal squad** for each gameweek.

This project ingests real FPL data, engineers features, trains predictive models, and selects squads based on expected performance — automating the weekly decision-making process.

---

📂 Project Structure

fpl-ml/
│
├── data/                        # Data storage
│   ├── raw/                     # Raw FPL data
│   ├── processed/               # Processed data & features
│   ├── predictions/             # Weekly model predictions
│   ├── evaluation/              # Model evaluation results
│   └── squads/                  # Saved optimal squads
│
├── src/                         # Source code
│   ├── ingest/                  # Data ingestion scripts
│   │   └── fetch_gw.py
│   ├── features/                # Feature engineering
│   │   └── update_features_weekly.py
│   ├── models/                  # Model training & prediction
│   │   ├── train_model_weekly.py
│   │   └── predict_next_gw.py
│   ├── evaluate/                # Model evaluation
│   │   └── evaluate_model_weekly.py
│   └── optimization/            # Squad optimization
│       └── select_squad.py
│
├── notebooks/                   # Exploratory analysis
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

---

## ⚙️ Pipeline Workflow

Each gameweek follows the same steps:

1. **Ingest Data** – Fetch the real FPL results of the previous GW  
2. **Update Features** – Update the dataset with the new results  
3. **Train Model** – Retrain ML model using data up to the current GW  
4. **Predict Next GW** – Predict player scores for the next GW  
5. **Optimize Squad** – Select the best 15-player squad for the upcoming GW  
6. **Evaluate Model** – Compare predictions with actual results to track model performance  

---

## 🚀 Weekly Workflow Example (Gameweek 3)

Run these commands one after another to produce the **GW3 optimal squad**.  
(Replace `2`/`3` with your real gameweek numbers.)

```bash
# 1) Fetch GW2 real results
python src/ingest/fetch_gw.py --gw 2

# 2) Update features using GW2
python src/features/update_features_weekly.py --gw 2

# 3) Retrain model using data up to GW2 (train on GW < 3)
python src/models/train_model_weekly.py --target_gw 3

# 4) Predict GW3
python src/models/predict_next_gw.py --target_gw 3

# 5) Optimize/select squad for GW3
python src/optimization/select_squad.py --pred data/predictions/predictions_gw3.csv

# 6) (After GW3 is complete) Evaluate model performance
python src/evaluate/evaluate_model_weekly.py --gw 3

👉 After this, you will have:
	•	Predictions file: data/predictions/predictions_gw3.csv
	•	Optimal Squad file: data/predictions/optimal_squad.csv (or saved per GW if extended)
	•	Evaluation results: data/evaluation/eval_gw3.csv (after GW3 completes)

⸻

🧠 Machine Learning Details
	•	Features Engineered:
	•	Player past performance (form, minutes, goals, assists, clean sheets)
	•	Team strength indicators
	•	Opponent difficulty
	•	Rolling averages (3, 5 gameweek windows) and exponential moving averages
	•	Standard deviation of points for volatility assessment
	•	Models:
	•	Stacking Ensemble (LightGBM + Random Forest + KNN with Ridge meta-learner)
	•	Single model fallback (LightGBM or Random Forest)
	•	Weekly retraining for adaptive learning
	•	Target Variable: Expected FPL points in next GW
	•	Evaluation Metrics: MAE (Mean Absolute Error) and R² Score

⸻

🏆 Optimization (Squad Selection)
	•	Constraints:
	•	15 players (2 GKs, 5 DEF, 5 MID, 3 FWD)
	•	Max 3 players per real team
	•	Budget cap (100.0 FPL budget)
	•	Method: Integer Linear Programming (ILP) to maximize total predicted points.

⸻

📊 Outputs
	•	Predicted scores per player per GW (data/predictions/predictions_gwX.csv)
	•	Optimal squad for the upcoming GW (data/predictions/optimal_squad.csv)
	•	Model evaluation results (data/evaluation/eval_gwX.csv) - compares predictions vs actual points
	•	Trained models (models/LightGBM_model.pkl, models/stacked_model.pkl)

Example (GW3 optimal squad):

Position	Player	Team	Predicted Points
GK	Player A	TOT	4.3
DEF	Player B	MCI	6.1
DEF	Player C	CHE	5.4
…	…	…	…
FWD	Player O	ARS	7.8


⸻

🛠️ Installation

# Clone repo
git clone https://github.com/kyupralis24/fpl-ml.git
cd fpl-ml

# Install dependencies
pip install -r requirements.txt


⸻

✅ Weekly Checklist

At the end of every GW:
	1.	Fetch the last GW results (`fetch_gw.py`)
	2.	Update features (`update_features_weekly.py`)
	3.	Retrain the model (target = next GW) (`train_model_weekly.py`)
	4.	Predict next GW player scores (`predict_next_gw.py`)
	5.	Run squad optimization (`select_squad.py`)
	6.	(After the GW completes) Evaluate model performance (`evaluate_model_weekly.py`)

⸻

📌 Future Improvements
	•	Save optimal squads per GW automatically (optimal_squad_gwX.csv)
	•	Add transfer logic across weeks
	•	Explore deep learning models for prediction
	•	Incorporate expected goals (xG/xA) data

⸻

👨‍💻 Author

Project maintained by Viom Kapur.
