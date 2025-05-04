import json
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate(model_path, data_path):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    features = ['mileage', 'age', 'brand_encoded', 'transmission_encoded']
    X = df[features]
    y = df['price']
    
    preds = model.predict(X)
    metrics = {
        "MAE": mean_absolute_error(y, preds),
        "R2": r2_score(y, preds)
    }
    
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    evaluate("models/model.joblib", "data/processed/cleaned.csv")