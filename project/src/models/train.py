import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# Импортируем функцию загрузки и список категориальных фич из нашего модуля data.loader
from src.data.loader import load_and_preprocess, CAT_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = 'data/raw/car_prices.csv'
MODEL_SAVE_PATH = 'artifacts/catboost_model.cbm'

def main():
    # 1. Загрузка данных (если нет основного файла, попробуем загрузить sample)
    try:
        X, y = load_and_preprocess(DATA_PATH)
    except FileNotFoundError:
        logging.warning(f"Файл {DATA_PATH} не найден. Пробуем запустить на демонстрационном sample...")
        X, y = load_and_preprocess('data/sample_car_prices.csv', nrows=100)
    
    # 2. Сплит данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Инициализация и обучение
    logging.info("Старт обучения CatBoost...")
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
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
    os.makedirs('artifacts', exist_ok=True)
    model.save_model(MODEL_SAVE_PATH)
    logging.info(f"Модель успешно сохранена в {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()