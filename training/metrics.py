# training/metrics.py

"""
Модуль з реалізацією базових метрик регресії:
- MAE (mean absolute error)
- RMSE (root mean squared error)
- MAPE (mean absolute percentage error)

Ці метрики будуть використовуватись як для toy-оцінювання, так і для
майбутніх експериментів на реальних даних.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Обчислює середню абсолютну похибку (MAE).

    :param y_true: тензор істинних значень
    :param y_pred: тензор передбачених значень
    :return: скалярний тензор з MAE
    """
    return (y_true - y_pred).abs().mean()


def rmse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Обчислює корінь середньоквадратичної похибки (RMSE).

    :param y_true: тензор істинних значень
    :param y_pred: тензор передбачених значень
    :return: скалярний тензор з RMSE
    """
    return torch.sqrt(((y_true - y_pred) ** 2).mean())


def mape(y_true: Tensor, y_pred: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Обчислює середню абсолютну відносну похибку (MAPE) у відсотках.

    Особливості:
    - додаємо eps у знаменник, щоб уникнути ділення на нуль;
    - результат множимо на 100, щоб отримати значення у %.

    :param y_true: тензор істинних значень
    :param y_pred: тензор передбачених значень
    :param eps: мале число для стабільності ділення
    :return: скалярний тензор з MAPE (%)
    """
    denom = y_true.abs().clamp_min(eps)
    return ((y_true - y_pred).abs() / denom).mean() * 100.0


def regression_metrics(y_true: Tensor, y_pred: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Комплексна функція, яка повертає одразу (MAE, RMSE, MAPE).

    :param y_true: істинні значення
    :param y_pred: передбачені значення
    :return: кортеж (mae_value, rmse_value, mape_value)
    """
    return mae(y_true, y_pred), rmse(y_true, y_pred), mape(y_true, y_pred)