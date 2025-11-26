"""
Головний модуль FastAPI-додатку.

Обов'язки:
- створити FastAPI-застосунок;
- при старті завантажити модель й ініціалізувати InferenceEngine;
- надати ендпоїнти:
    - GET /healthz   - живий процес
    - GET /readyz    - готовність моделі
    - POST /predict  - передбачення
"""

from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.inference import InferenceEngine
from app.model_loader import load_model_instance
from app.schemas import (
    HealthStatusResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessStatusResponse,
)

# Завантажуємо змінні з .env, якщо файл існує
load_dotenv()

app = FastAPI(
    title="Traffic GNN Inference API",
    version="0.1.0",
    description="API для інференсу (поки що з dummy-моделлю).",
)

# Дозволяємо CORS для зручності локальної розробки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # у проді це варто обмежити
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальний engine (буде ініціалізований при старті)
engine: Optional[InferenceEngine] = None


@app.on_event("startup")
def on_startup() -> None:
    """
    Хук, який викликається при старті FastAPI-додатку.

    Тут:
    - читаємо INFERENCE_DEVICE (cpu/cuda);
    - створюємо модель через load_model_instance();
    - обгортаємо її в InferenceEngine.
    """
    global engine

    device = os.getenv("INFERENCE_DEVICE", "cpu")

    # Логічно: спочатку створити модель...
    model = load_model_instance()

    # ...а потім обгорнути в inference engine
    engine = InferenceEngine(model=model, device=device)

    # (опційно) Можна зробити простий warm-up, але для dummy-моделі це не обов'язково
    # Наприклад:
    # _ = engine.predict(node_features=[[0.0]], edge_index=None)


@app.get("/healthz", response_model=HealthStatusResponse)
def healthcheck() -> HealthStatusResponse:
    """
    Простіший healthcheck:
    - якщо процес живий, повертає статус "ok".
    - не перевіряє модель, S3, БД тощо.
    """
    return HealthStatusResponse(status="ok")


@app.get("/readyz", response_model=ReadinessStatusResponse)
def readiness() -> ReadinessStatusResponse:
    """
    Перевірка готовності сервісу до обслуговування запитів.

    Логіка:
    - якщо engine ініціалізований (не None), вважаємо модель завантаженою.
    """
    model_loaded = engine is not None
    status = "ready" if model_loaded else "not_ready"
    return ReadinessStatusResponse(status=status, model_loaded=model_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Основний ендпоїнт для передбачення.

    Приймає PredictionRequest, передає дані в InferenceEngine,
    повертає список прогнозів (по одному на вузол).
    """
    if engine is None:
        # Якщо модель не встигла завантажитися або сталася помилка
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start_time = time.perf_counter()
    predictions = engine.predict(
        node_features=request.node_features,
        edge_index=request.edge_index,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # TODO: у майбутньому можна логувати latency, розмір графа тощо
    # print(f"/predict handled in {elapsed_ms:.2f} ms")

    return PredictionResponse(predictions=predictions)