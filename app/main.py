"""
Головний модуль FastAPI-додатку.

Обов'язки:
- створити FastAPI-застосунок;
- при старті завантажити модель й ініціалізувати InferenceEngine;
- надати ендпоїнти:
    - GET /healthz   - живий процес
    - GET /readyz    - готовність моделі
    - POST /predict  - передбачення
    - GET /metrics   - технічні метрики Prometheus
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (
    Counter,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from app.inference import InferenceEngine
from app.model_loader import load_model_instance
from app.schemas import (
    HealthStatusResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessStatusResponse,
)

# ============ Налаштування логування ============

logger = logging.getLogger("traffic_gnn_inference")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ============ Prometheus-метрики ============

REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Кількість запитів до сервісу інференсу",
)

REQUEST_EXCEPTIONS_TOTAL = Counter(
    "inference_exceptions_total",
    "Кількість помилок під час інференсу",
)

REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Тривалість обробки запиту до моделі",
)

# Завантажуємо змінні з .env, якщо файл існує
load_dotenv()

app = FastAPI(
    title="Traffic GNN Inference API",
    version="0.1.0",
    description="API для інференсу гібридної моделі дорожнього трафіку.",
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
    logger.info("INFERENCE_DEVICE=%s", device)

    # Логічно: спочатку створити модель...
    model = load_model_instance()

    # ...а потім обгорнути в inference engine
    engine = InferenceEngine(model=model, device=device)

    logger.info("InferenceEngine успішно ініціалізовано.")


# ================= Health / Readiness =================


@app.get("/healthz", response_model=HealthStatusResponse)
def healthcheck() -> HealthStatusResponse:
    """
    Проста перевірка "живий / не живий" (liveness).
    Якщо процес FastAPI працює, повертаємо status="ok".
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


# ================= Prometheus /metrics =================


@app.get("/metrics")
def metrics() -> Response:
    """
    Ендпоінт для експорту технічних метрик Prometheus.

    Метрики:
    - inference_requests_total
    - inference_exceptions_total
    - inference_latency_seconds
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ================= Основний ендпоінт передбачення =================


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Основний ендпоїнт для передбачення.

    Приймає PredictionRequest, передає дані в InferenceEngine,
    повертає список прогнозів (по одному на вузол).
    Також:
    - рахує кількість запитів/помилок;
    - вимірює латентність інференсу.
    """
    global engine

    if engine is None:
        # Якщо модель не встигла завантажитися або сталася помилка при старті
        raise HTTPException(status_code=503, detail="Model is not loaded")

    REQUESTS_TOTAL.inc()
    start_time = time.perf_counter()

    try:
        predictions = engine.predict(
            node_features=request.node_features,
            edge_index=request.edge_index,
        )

        elapsed_sec = time.perf_counter() - start_time
        REQUEST_LATENCY.observe(elapsed_sec)

        logger.info(
            "inference_success",
            extra={
                "latency_ms": elapsed_sec * 1000.0,
                "nodes": len(request.node_features),
            },
        )

        return PredictionResponse(predictions=predictions)

    except Exception as exc:
        REQUEST_EXCEPTIONS_TOTAL.inc()
        logger.exception("Помилка під час інференсу: %s", exc)
        raise HTTPException(status_code=500, detail="Inference failed")