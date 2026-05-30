import os
import logging
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# Импортируем из загрузчика
from src.data.loader import load_and_preprocess, CAT_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загружаем конфигурацию
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Достаем параметры из конфигурационного файла
DATA_PATH = config["paths"]["raw_data_path"]
SAMPLE_DATA_PATH = config["paths"]["sample_data_path"]
MODEL_SAVE_PATH = config["paths"]["model_save_path"]

ROWS_TO_READ = config["train_params"]["rows_to_read"]
ITERATIONS = config["train_params"]["iterations"]
LEARNING_RATE = config["train_params"]["learning_rate"]
TEST_SIZE = config["train_params"]["test_size"]
RANDOM_STATE = config["train_params"]["random_state"]

def main():
    # 1. Загрузка данных (если нет основного файла, берем sample)
    try:
        X, y = load_and_preprocess(DATA_PATH, nrows=ROWS_TO_READ)
    except FileNotFoundError:
        logging.warning(f"Файл {DATA_PATH} не найден. Запускаем на демонстрационном sample...")
        X, y = load_and_preprocess(SAMPLE_DATA_PATH, nrows=100)
    
    # 2. Сплит данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # 3. Обучение модели с гиперпараметрами из конфига
    logging.info("Старт обучения CatBoost (параметры считываются из config.yaml)...")
    model = CatBoostRegressor(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        cat_features=CAT_FEATURES,
        verbose=100
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # 4. Оценка качества
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    logging.info(f"Обучение завершено. Метрики: MAE = ${mae:.2f} | R2 = {r2:.4f}")
    
    # 5. Сохранение артефакта
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save_model(MODEL_SAVE_PATH)
    logging.info(f"Модель успешно сохранена в {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()