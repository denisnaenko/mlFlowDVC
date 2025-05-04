import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import json
from pathlib import Path

def train_model(input_path, model_path, metrics_path):
    try:
        # Проверка существования файла
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Файл {input_path} не найден")
        
        # Загрузка с проверкой
        df = pd.read_csv(input_path)
        print(f"Загружено {len(df)} строк. Пример:")
        print(df[['Month/Year', 'New Price ($)', 'Used Price ($)']].head())
        
        # Проверка данных
        if len(df) == 0:
            raise ValueError("Файл не содержит данных")
        
        # Подготовка данных
        X = df[['New Price ($)']].values
        y = df['Used Price ($)'].values
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Нет данных для обучения")
        
        # Обучение
        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            min_samples_leaf=3  # Защита от пустых данных
        )
        model.fit(X, y)
        print("Модель успешно обучена")
        
        # Сохранение
        Path(model_path).parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        
        metrics = {
            "samples": len(X),
            "status": "success",
            "features": list(df.columns)
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
        return True
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        with open(metrics_path, 'w') as f:
            json.dump({"error": str(e)}, f)
        return False

if __name__ == "__main__":
    success = train_model(
        "data/processed/cleaned.csv",
        "models/model.joblib",
        "metrics/scores.json"
    )
    exit(0 if success else 1)