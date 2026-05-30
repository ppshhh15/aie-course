import os
import logging
from contextlib import asynccontextmanager
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загружаем конфигурацию
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["paths"]["model_save_path"]
API_TITLE = config["api"]["title"]
API_DESC = config["api"]["description"]
API_VERSION = config["api"]["version"]

# Управляем жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Файл модели {MODEL_PATH} не найден! Сначала запустите обучение.")
        raise FileNotFoundError("Модель не найдена.")
    
    logging.info(f"Загрузка модели CatBoost из {MODEL_PATH}...")
    app.state.model = CatBoostRegressor()
    app.state.model.load_model(MODEL_PATH)
    logging.info("Модель успешно загружена.")
    
    yield # Сервис начинает принимать запросы
    
    logging.info("Остановка приложения, освобождение ресурсов...")
    app.state.model = None


# Инициализируем FastAPI с параметрами из конфига
app = FastAPI(
    title=API_TITLE,
    description=API_DESC,
    version=API_VERSION,
    lifespan=lifespan
)

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
        "message": f"{API_TITLE} is running. Go to /docs for Swagger UI."
    }

@app.post("/predict")
def predict_price(car: CarInput):
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена на сервере.")
    
    try:
        input_data = pd.DataFrame([car.model_dump()])
        predicted_value = app.state.model.predict(input_data)[0]
        
        return {
            "predicted_price_usd": round(float(predicted_value), 2)
        }
    except Exception as e:
        logging.error(f"Ошибка во время инференса: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка сервера при обработке запроса: {str(e)}")