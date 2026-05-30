import os
import logging
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'artifacts/catboost_model.cbm'

# 1. Описываем Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код здесь выполняется ПРИ СТАРТЕ приложения
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Файл модели {MODEL_PATH} не найден! Сначала запустите обучение.")
        raise FileNotFoundError("Модель не найдена.")
    
    logging.info("Загрузка модели CatBoost через Lifespan...")
    # Сохраняем модель в state приложения (это стандарт FastAPI)
    app.state.model = CatBoostRegressor()
    app.state.model.load_model(MODEL_PATH)
    logging.info("Модель успешно загружена и готова к работе.")
    
    yield # В этой точке приложение начинает принимать запросы
    
    # Код здесь выполняется ПРИ ОСТАНОВКЕ приложения
    logging.info("Остановка приложения, освобождение ресурсов...")
    app.state.model = None


# 2. Инициализируем FastAPI и передаем ему наш lifespan handler
app = FastAPI(
    title="Used Car Price Predictor API",
    description="API для оценки стоимости подержанных автомобилей на основе CatBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Описываем входные данные по стандарту Pydantic v2
class CarInput(BaseModel):
    year: int = Field(2015, description="Год выпуска", examples=[2015])
    make: str = Field("Ford", description="Производитель", examples=["Ford"])
    model: str = Field("Fusion", description="Модель", examples=["Fusion"])
    trim: str = Field("SE", description="Комплектация", examples=["SE"])
    body: str = Field("Sedan", description="Тип кузова", examples=["Sedan"])
    transmission: str = Field("automatic", description="Коробка передач", examples=["automatic"])
    state: str = Field("ca", description="Штат регистрации", examples=["ca"])
    condition: float = Field(4.5, description="Состояние машины (от 1 до 5)", examples=[4.5])
    odometer: float = Field(60000.0, description="Пробег (в милях)", examples=[60000.0])
    color: str = Field("black", description="Цвет кузова", examples=["black"])
    interior: str = Field("black", description="Цвет салона", examples=["black"])

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "Used Car Price Predictor API is running. Go to /docs for Swagger UI."
    }

@app.post("/predict")
def predict_price(car: CarInput):
    # Достаем модель из state приложения
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена на сервере.")
    
    try:
        input_data = pd.DataFrame([car.model_dump()])
        
        # Делаем предсказание через сохраненную в state модель
        predicted_value = app.state.model.predict(input_data)[0]
        
        return {
            "predicted_price_usd": round(float(predicted_value), 2)
        }
    except Exception as e:
        logging.error(f"Ошибка во время инференса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при обработке запроса: {str(e)}")