import os
import pandas as pd
from src.data.loader import load_and_preprocess, FEATURES

# Путь к демонстрационному датасету
SAMPLE_DATA_PATH = 'data/sample_car_prices.csv'

def test_sample_file_exists():
    """Проверяем, что демонстрационный файл физически существует"""
    assert os.path.exists(SAMPLE_DATA_PATH), f"Файл {SAMPLE_DATA_PATH} не найден!"

def test_data_loader_columns():
    """Sanity-check: проверяем корректность колонок после предобработки"""
    X, y = load_and_preprocess(SAMPLE_DATA_PATH, nrows=10)
    
    # Проверяем, что на выходе ровно те фичи, которые мы ожидаем в модели
    assert list(X.columns) == FEATURES
    # Проверяем, что таргет отделился корректно
    assert len(y) == 10
    assert y.name == 'sellingprice'

def test_data_loader_no_nan_values():
    """Sanity-check: проверяем, что препроцессинг заполнил все NaN значения"""
    X, _ = load_and_preprocess(SAMPLE_DATA_PATH, nrows=10)
    
    # После работы loader пропусков быть не должно
    assert X.isnull().sum().sum() == 0, "В обработанных данных остались пропущенные значения (NaN)!"