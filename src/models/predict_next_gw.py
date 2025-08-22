# src/models/predict_next_gw.py

import pandas as pd
import joblib

FEATURES_PATH = "data/processed/features.csv"
MODEL_PATH = "models/LightGBM_model.pkl"  # or whichever model you chose

def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(FEATURES_PATH)

    # Identify current gameweek
    current_max = df['round'].max()
    print(f"Current last GW: {current_max}")

    # Select data for the current gameweek
    upcoming_df = df[df['round'] == current_max]

    # Select numeric features for prediction
    feature_cols = [c for c in upcoming_df.select_dtypes(include=[float, int]).columns 
                    if c not in ['total_points', 'round']]

    # Predict
    preds = model.predict(upcoming_df[feature_cols])

    # Create output dataframe with player info and price
    output = upcoming_df[['name', 'team', 'position', 'value']].copy()
    output['pred_points'] = preds

    # Sort by predicted points
    output = output.sort_values('pred_points', ascending=False).reset_index(drop=True)

    # Save predictions
    output_path = f"data/predictions/predictions_gw{current_max+1}.csv"
    output.to_csv(output_path, index=False)

    print(output.head(15))
    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    main()