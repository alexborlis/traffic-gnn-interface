from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """
    Схема вхідного запиту до /predict.

    node_features:
        Список вузлів, кожен вузол - це список числових ознак.
        Наприклад:
        [
          [10.0],   # вузол 0
          [20.0],   # вузол 1
          ...
        ]

    edge_index:
        Опис графа у форматі COO:
        [
          [0, 1, 1],  # from-список
          [1, 0, 2]   # to-список
        ]
        Може бути опущений (None) для dummy-моделі.
    """

    node_features: List[List[float]]
    edge_index: Optional[List[List[int]]] = None


class PredictionResponse(BaseModel):
    """
    Схема вихідної відповіді від /predict.
    """

    predictions: List[float]


class HealthStatusResponse(BaseModel):
    """
    Відповідь для /healthz.
    """

    status: str


class ReadinessStatusResponse(BaseModel):
    """
    Відповідь для /readyz.
    """

    status: str
    model_loaded: bool