# Используем официальный легковесный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем системные утилиты, необходимые для сборки некоторых C++ библиотек
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip и устанавливаем библиотеки
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем код приложения, конфигурации, артефакты модели и тестовые данные
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/ ./artifacts/
COPY data/sample_car_prices.csv ./data/sample_car_prices.csv

# Открываем порт 8000 для связи с внешним миром
EXPOSE 8000

# Команда для запуска нашего FastAPI приложения
CMD ["uvicorn", "src.service.app:app", "--host", "0.0.0.0", "--port", "8000"]