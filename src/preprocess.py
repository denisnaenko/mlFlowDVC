import pandas as pd
from pathlib import Path

def preprocess_data(input_path, output_path):
    try:
        # Загрузка с обработкой пропущенных значений
        df = pd.read_csv(
            input_path,
            thousands=',',
            na_values=['', 'NA', 'N/A', 'NaN']
        )
        
        # Проверка и очистка данных
        df = df.dropna(subset=['Used Price ($)'])
        df = df[df['Used Price ($)'] > 0]
        
        # Сохранение
        Path(output_path).parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Обработано {len(df)} строк. Пример данных:")
        print(df.head())
        return True
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return False

if __name__ == "__main__":
    success = preprocess_data(
        "data/raw/automobile_prices_economics_2019_2023.csv",
        "data/processed/cleaned.csv"
    )
    exit(0 if success else 1)