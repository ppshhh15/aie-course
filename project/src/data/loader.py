import pandas as pd
import yaml

# Загружаем конфигурацию
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Список признаков для модели
FEATURES = ['year', 'make', 'model', 'trim', 'body', 'transmission', 'state', 'condition', 'odometer', 'color', 'interior']
CAT_FEATURES = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior']
NUM_FEATURES = ['year', 'condition', 'odometer']

def load_and_preprocess(path: str, nrows: int = None):
    """
    Загрузка данных и заполнение пропущенных значений.
    """
    if nrows is None:
        # Если количество строк не передано явно, берем его из конфига
        nrows = config["train_params"]["rows_to_read"]

    df = pd.read_csv(path, nrows=nrows).dropna(subset=['sellingprice'])
    
    # Заполнение категориальных пропусков
    df[CAT_FEATURES] = df[CAT_FEATURES].fillna('Unknown').astype(str)
    
    # Заполнение численных пропусков медианой
    for col in NUM_FEATURES:
        df[col] = df[col].fillna(df[col].median())
        
    return df[FEATURES], df['sellingprice']