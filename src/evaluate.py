import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import json

def evaluate(model_path, data_path, metrics_path):
    # Загрузка модели и данных
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    # Подготовка данных
    X = df[['New Price ($)']]
    y = df['Used Price ($)']
    
    # Предсказания и метрики
    preds = model.predict(X)
    metrics = {
        "mae": mean_absolute_error(y, preds),
        "r2": r2_score(y, preds),
        "samples": len(X)
    }
    
    # Сохранение метрик
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    print("Evaluation metrics saved to", metrics_path)

if __name__ == "__main__":
    evaluate(
        "models/model.joblib",
        "data/processed/cleaned.csv",
        "metrics/evaluation.json"
    )