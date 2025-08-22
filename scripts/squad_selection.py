import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('models/model.pkl')

# Try loading the scaler if it exists, else create a new one (fit on incoming data)
try:
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    print("[INFO] scaler.pkl not found. Scaling will be fit on current data instead.")
    scaler = None

# Load upcoming gameweek predictions
predictions = pd.read_csv('data/predicted_next_gw.csv')

# If scaling is required for features
feature_cols = ['form', 'points_per_game', 'selected_by_percent', 'now_cost']  # example
if scaler:
    predictions[feature_cols] = scaler.transform(predictions[feature_cols])
else:
    # Fit scaling on the current data to keep numbers in range
    temp_scaler = StandardScaler()
    predictions[feature_cols] = temp_scaler.fit_transform(predictions[feature_cols])

# Sort players by predicted points
predictions_sorted = predictions.sort_values(by='predicted_points', ascending=False)

# Select top players for the squad
# Example: Top 11 players regardless of position (you can make position-specific rules)
selected_squad = predictions_sorted.head(11)

# Save the selected squad
selected_squad.to_csv('data/selected_squad.csv', index=False)
print("[INFO] Squad selection completed. Saved to data/selected_squad.csv")