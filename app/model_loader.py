"""
Модуль, відповідальний за створення екземпляра моделі для інференсу.

На даному етапі:
- ми завжди створюємо DummyTrafficModel, якщо не вказано інший клас у MODEL_CLASS_PATH;
- за наявності MODEL_ARTIFACT_URI пробуємо завантажити state_dict з файлу;
- все налаштовано так, щоб працювало навіть без будь-яких ваг (тільки dummy).

У майбутньому тут можна:
- реалізувати завантаження з S3/MinIO;
- розділити логіку для різних типів моделей (GCN, GAT, Transformer тощо).
"""

import importlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _import_model_class(model_class_path: str) -> type[nn.Module]:
    """
    Імпортує клас моделі за повним шляхом "package.module.ClassName".

    :param model_class_path: рядок формату "models.dummy_model.DummyTrafficModel"
    :return: клас, що наслідує nn.Module
    """
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
    2. Створюємо екземпляр класу без аргументів (або можна додати конфіг у майбутньому).
    3. Якщо задано MODEL_ARTIFACT_URI і файл існує – пробуємо підвантажити state_dict.
    """
    # 1. Клас моделі
    model_class_path = os.getenv(
        "MODEL_CLASS_PATH",
        "models.dummy_model.DummyTrafficModel",
    )
    model_class = _import_model_class(model_class_path)

    # 2. Створення екземпляра моделі (поки без параметрів конфігурації)
    #    У майбутньому тут можна вчитувати розмірність ознак, кількість шарів тощо.
    model: nn.Module = model_class()  # type: ignore[call-arg]

    # 3. Спроба завантажити ваги з файлу (якщо шлях заданий)
    artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
    if artifact_uri:
        artifact_path = Path(artifact_uri)
        if artifact_path.is_file():
            state: Any = torch.load(artifact_path, map_location="cpu")
            # Очікуємо, що state - це state_dict
            if isinstance(state, dict):
                model.load_state_dict(state)
            else:
                raise ValueError(
                    f"Unexpected content in model artifact: {artifact_uri}. "
                    f"Expected state_dict (dict), got {type(state)}"
                )

    return model