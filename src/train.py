import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import yaml

# Загрузка параметров
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

def train_model(input_path, output_path):
    df = pd.read_csv(input_path)
    features = ['mileage', 'age', 'brand_encoded', 'transmission_encoded']
    target = 'price'
    
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestRegressor(
        n_estimators=params['train']['n_estimators'],
        max_depth=params['train']['max_depth']
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, output_path)

if __name__ == "__main__":
    train_model("data/processed/cleaned.csv", "models/model.joblib")