# app/model_loader.py

import importlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _import_model_class(model_class_path: str) -> type[nn.Module]:
    module_name, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    if not issubclass(model_class, nn.Module):
        raise TypeError(f"Class {model_class_path} is not a torch.nn.Module subclass")
    return model_class


def load_model_instance() -> nn.Module:
    """
    Створює та повертає екземпляр моделі для інференсу.

    Логіка:
    1. Читаємо MODEL_CLASS_PATH з env (або беремо DummyTrafficModel за замовчуванням).
    2. Якщо це TrafficGraphNeuralNetwork – передаємо параметри input_features, hidden_units, output_features.
       Їх беремо з env (MODEL_INPUT_FEATURES, MODEL_HIDDEN_UNITS, MODEL_OUTPUT_FEATURES) або ставимо дефолти.
    3. Якщо вказано MODEL_ARTIFACT_URI і файл існує – підвантажуємо state_dict.
    """
    model_class_path = os.getenv(
        "MODEL_CLASS_PATH",
        "models.dummy_model.DummyTrafficModel",
    )

    model_class = _import_model_class(model_class_path)

    # --- 2. Створення екземпляра моделі з урахуванням параметрів -----------------
    if model_class_path.endswith(".TrafficGraphNeuralNetwork"):
        # Беремо конфіг з env або ставимо дефолти під наш toy-експеримент
        input_features = int(os.getenv("MODEL_INPUT_FEATURES", "1"))
        hidden_units = int(os.getenv("MODEL_HIDDEN_UNITS", "16"))
        output_features = int(os.getenv("MODEL_OUTPUT_FEATURES", "1"))

        model: nn.Module = model_class(  # type: ignore[call-arg]
            input_features=input_features,
            hidden_units=hidden_units,
            output_features=output_features,
        )
    else:
        # Для DummyTrafficModel та інших простих моделей конструктор без аргументів
        model = model_class()  # type: ignore[call-arg]

    # --- 3. Завантаження ваг за потреби -----------------------------------------
    artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
    if artifact_uri:
        artifact_path = Path(artifact_uri)
        if artifact_path.is_file():
            state: Any = torch.load(artifact_path, map_location="cpu")
            if isinstance(state, dict):
                model.load_state_dict(state)
            else:
                raise ValueError(
                    f"Unexpected content in model artifact: {artifact_uri}. "
                    f"Expected state_dict (dict), got {type(state)}"
                )

    return model