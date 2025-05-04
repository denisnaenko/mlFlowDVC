import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Очистка
    df = df.dropna(subset=['price', 'mileage'])
    df = df[df['price'] > 0]
    
    # Генерация признаков
    df['age'] = 2023 - df['year']
    df['brand'] = df['model'].apply(lambda x: x.split()[0])
    
    # Кодировка категориальных признаков
    le = LabelEncoder()
    df['brand_encoded'] = le.fit_transform(df['brand'])
    df['transmission_encoded'] = le.fit_transform(df['transmission'])
    
    # Сохранение
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data(
        "data/raw/automobile_prices_economics_2019_2023.csv",
        "data/processed/cleaned.csv"
    )