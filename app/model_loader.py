# app/model_loader.py

"""
Завантаження PyTorch-моделі для сервісу інференсу.

Основні задачі:
- Прочитати налаштування з env (клас моделі, шлях до файлу з вагами, розмірності).
- Створити інстанс моделі TrafficGraphNeuralNetwork з правильними параметрами.
- Підвантажити state_dict з файлу (наприклад, models/traffic_gnn_metrla.pt).
- Використовується у FastAPI через глобальний state (app.state.model).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from typing import Type

import torch
from torch import nn

# === 1. Конфіг за замовчуванням / через env =================================

# Клас моделі: повний шлях "модуль.клас"
MODEL_CLASS_PATH = os.getenv(
    "MODEL_CLASS_PATH",
    "models.traffic_gnn.TrafficGraphNeuralNetwork",
)

# Файл з вагами моделі: ТУТ ПІДКЛЮЧАЄМО METR-LA
MODEL_ARTIFACT_URI = os.getenv(
    "MODEL_ARTIFACT_URI",
    "models/traffic_gnn_metrla.pt",  # <- дефолт тепер METR-LA
)

# Параметри архітектури мають збігатися з training/train_metr_la.py
MODEL_INPUT_FEATURES = int(os.getenv("MODEL_INPUT_FEATURES", "1"))
MODEL_HIDDEN_UNITS = int(os.getenv("MODEL_HIDDEN_UNITS", "32"))
MODEL_OUTPUT_FEATURES = int(os.getenv("MODEL_OUTPUT_FEATURES", "1"))

# CPU/CPU-GPU – поки що тримаємо все на CPU для простоти
DEVICE = torch.device(os.getenv("DEVICE", "cpu"))


@dataclass
class ModelConfig:
    """Проста конфігурація для ініціалізації моделі."""

    input_features: int
    hidden_units: int
    output_features: int
    artifact_uri: str
    device: torch.device


def import_model_class(class_path: str) -> Type[nn.Module]:
    """
    Імпортує клас моделі за рядком виду "module.submodule.ClassName".

    Наприклад:
        "models.traffic_gnn.TrafficGraphNeuralNetwork"
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    if not issubclass(model_class, nn.Module):
        raise TypeError(f"{class_path} не є підкласом torch.nn.Module")
    return model_class


def load_model_instance() -> nn.Module:
    """
    Створює інстанс моделі та підвантажує ваги з файлу.

    Викликається на startup у FastAPI (див. app/main.py).
    """

    config = ModelConfig(
        input_features=MODEL_INPUT_FEATURES,
        hidden_units=MODEL_HIDDEN_UNITS,
        output_features=MODEL_OUTPUT_FEATURES,
        artifact_uri=MODEL_ARTIFACT_URI,
        device=DEVICE,
    )

    print("[MODEL LOADER] Використовуємо клас:", MODEL_CLASS_PATH)
    print("[MODEL LOADER] Завантажуємо ваги з:", config.artifact_uri)
    print(
        f"[MODEL LOADER] input_features={config.input_features}, "
        f"hidden_units={config.hidden_units}, "
        f"output_features={config.output_features}"
    )

    # 1. Імпортуємо клас моделі
    model_class = import_model_class(MODEL_CLASS_PATH)

    # 2. Створюємо інстанс з потрібними параметрами
    model: nn.Module = model_class(
        input_features=config.input_features,
        hidden_units=config.hidden_units,
        output_features=config.output_features,
    )

    # 3. Підвантажуємо state_dict
    if not os.path.isfile(config.artifact_uri):
        raise FileNotFoundError(
            f"[MODEL LOADER] Файл з вагами не знайдено: {config.artifact_uri}"
        )

    state_dict = torch.load(config.artifact_uri, map_location=config.device)
    model.load_state_dict(state_dict)

    # 4. Переводимо модель у eval-режим і на потрібний девайс
    model.to(config.device)
    model.eval()

    print("[MODEL LOADER] Модель успішно завантажена та готова до інференсу.")
    return model