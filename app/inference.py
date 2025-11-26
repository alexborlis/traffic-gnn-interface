"""
Модуль для інференсу (InferenceEngine).

Інкапсулює:
- модель (torch.nn.Module),
- пристрій виконання (cpu / cuda),
- перетворення сирих вхідних даних з API у тензори,
- виконання model.forward(...) з вимкненими градієнтами.

На цьому етапі:
- ми не використовуємо batching, лише один запит = один граф;
- edge_index може бути None (для dummy-моделі це ок).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import nn
from torch import Tensor


class InferenceEngine:
    """
    Клас, який інкапсулює модель та логіку передбачення.

    Приклад використання:
        engine = InferenceEngine(model, device="cpu")
        predictions = engine.predict(node_features=[[1.0], [2.0]], edge_index=None)
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """
        :param model: torch-модель, що наслідує nn.Module
        :param device: 'cpu' або 'cuda' (якщо доступно)
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        # Переводимо модель в режим інференсу (відключає dropout, batchnorm у train-режимі)
        self.model.eval()

    def predict(
        self,
        node_features: List[List[float]],
        edge_index: Optional[List[List[int]]] = None,
    ) -> List[float]:
        """
        Основний метод передбачення.

        :param node_features: список вузлів, кожен вузол - список числових ознак
                              розмірності [num_nodes][num_features]
        :param edge_index: список ребер графа у COO-форматі:
                           [[from_1, from_2, ...], [to_1, to_2, ...]]
                           Може бути None (для dummy-моделі)
        :return: список передбачень (по одному числу на кожен вузол)
        """
        # Перетворюємо вхід на тензор
        # Якщо node_features порожній - створюємо тензор з нульовою кількістю вузлів
        if node_features:
            x: Tensor = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        else:
            x = torch.empty((0, 1), dtype=torch.float32, device=self.device)

        # Edge_index теж перетворюємо на тензор або залишаємо None
        edge_tensor: Tensor | None
        if edge_index and len(edge_index) == 2:
            edge_tensor = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        else:
            edge_tensor = None

        # Інференс без градієнтів
        with torch.no_grad():
            output: Tensor = self.model(x, edge_tensor)

        # Повертаємо Python-список float-ів
        return output.detach().cpu().tolist()