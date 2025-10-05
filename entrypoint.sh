#!/bin/bash
set -euo pipefail

# Якщо модель відсутня, завантажуємо вручну
if [ ! -f "/opt/model/model.pt" ] && [ -n "${MODEL_ARTIFACT_URI:-}" ]; then
  echo "Завантаження моделі з ${MODEL_ARTIFACT_URI}"
  python -c "from app.model_loader import завантажити_модель; завантажити_модель()"
fi

exec uvicorn app.main:додаток --host 0.0.0.0 --port ${APP_PORT} --workers ${UVICORN_WORKERS:-1}