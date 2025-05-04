# mlFlowDVC
Создание pipeline обучения модели с использованием DVC.

## Структура проекта
```
.
├── data/
│ ├── raw/ # Исходные данные
│ └── processed/ # Очищенные данные
├── models/ # Обученные модели
├── metrics/ # Метрики качества
├── src/ # Исходный код
│ ├── preprocess.py
│ ├── train.py
│ └── evaluate.py
├── dvc.yaml # Конфигурация DVC pipeline
├── params.yaml
└── requirements.txt # Зависимости
```

## Требования

- Python 3.8+
- DVC 2.0+
- Pandas, Scikit-learn

Установите зависимости:
```bash
pip install -r requirements.txt
```

## Данные

Исходный датасет: https://www.kaggle.com/datasets/abduulwasay/auto-prices-and-economic-trends-20192023?select=automobile_prices_economics_2019_2023.csv

## Запуск pipeline

1. Склонируйте репозиторий:
```bash
git clone <your-repo-url>
cd <project-folder>
```

2. Инициализируйте DVC:
```bash
dvc init
```

3. Запустите полный pipeline:
```bash
dvc repro
```

### Этапы pipeline
1. Подготовка данных:
- Очистка от пропущенных значений
- Преобразование форматов
- Генерация признаков

```bash
dvc repro prepare
```

2. Обучение модели:
- RandomForestRegressor
- Настройка гиперпараметров
```bash
dvc repro train
```

3. Оценка качества:
- Расчет MAE, R2-score
```bash
dvc repro evaluate
```

## Проверка результатов
Просмотр метрик:
```bash
dvc metrics show
```


