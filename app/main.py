import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import load_model_instance
from app.inference import InferenceEngine
from app.schemas import (
    HealthStatusResponse,
    ReadinessStatusResponse,
    PredictionRequest,
    PredictionResponse
)

logger = logging.getLogger("TrafficGNNApp")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="Traffic GNN Inference Service")

# Дозволені джерела для CORS (за замовчуванням - всі)
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = None
engine = None
model_is_ready = False

@app.on_event("startup")
async def on_startup():
    """Ініціалізація сервісу: завантаження моделі та warmup"""
    global model, engine, model_is_ready
    try:
        model = load_model_instance()
        engine = InferenceEngine(model)
        engine.warmup()
        model_is_ready = True
        logger.info("Модель завантажено та ініціалізовано")
    except Exception as error:
        logger.exception("Помилка при ініціалізації моделі: %s", error)

@app.get("/healthz", response_model=HealthStatusResponse)
def health_check():
    """Перевірка доступності сервісу"""
    return {"status": "ok"}

@app.get("/readyz", response_model=ReadinessStatusResponse)
def readiness_check():
    """Перевірка готовності моделі до інференсу"""
    if not model_is_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ready"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Основна точка інференсу для моделі"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine unavailable")
    start_time = time.time()
    try:
        predictions = engine.predict(request)
        latency = (time.time() - start_time) * 1000
        return PredictionResponse(
            predictions=predictions,
            model_version=engine.model_version,
            latency_ms=round(latency, 2)
        )
    except Exception as error:
        logger.exception("Помилка під час інференсу")
        raise HTTPException(status_code=500, detail=str(error))
