# models/traffic_gnn.py

"""
Спрощена реалізація графової нейронної мережі без залежності від torch_geometric.

Ідея:
- Замість GCNConv з torch_geometric ми реалізуємо дуже простий "graph convolution" вручну:
  x' = A_hat @ (x @ W)
  де A_hat - матриця суміжності графа з доданими self-loops.

Обмеження:
- Реалізація орієнтована на невеликі графи (для toy-навчання та відладки пайплайна).
- Немає нормалізації ступенів (degree normalization), але для синтетичного прикладу цього достатньо.

Ця модель:
- приймає node_features [num_nodes, input_features],
- edge_index [2, num_edges] у COO-форматі,
- повертає прогнози [num_nodes, output_features].
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch import Tensor


def build_adjacency(edge_index: Tensor, num_nodes: int, device: torch.device) -> Tensor:
    """
    Будує просту матрицю суміжності A розмірності [num_nodes, num_nodes]
    з додаванням self-loops на діагоналі.

    :param edge_index: тензор [2, num_edges] з індексами (from, to)
    :param num_nodes: кількість вузлів у графі
    :param device: пристрій, на якому має бути матриця
    :return: A_hat [num_nodes, num_nodes]
    """
    # Ініціалізуємо нульову матрицю суміжності
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    if edge_index.numel() > 0:
        # edge_index[0] - from, edge_index[1] - to
        src = edge_index[0]
        dst = edge_index[1]
        adj[src, dst] = 1.0
        # Якщо хочемо зробити граф неорієнтованим - можна додати ще adj[dst, src] = 1.0

    # Додаємо self-loops: кожен вузол зв'язаний сам із собою
    idx = torch.arange(num_nodes, device=device)
    adj[idx, idx] = 1.0

    return adj


class SimpleGraphConvLayer(nn.Module):
    """
    Один шар графової згортки:
        x' = A_hat @ (x @ W) + b
    без нормалізації ступенів вузлів.

    Для toy-прикладу цього цілком достатньо, щоб:
    - продемонструвати роботу "графового" шару;
    - навчити модель на простому датасеті.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))

        # Ініціалізація ваг (xavier для кращої збіжності)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        :param x: [num_nodes, input_dim]
        :param edge_index: [2, num_edges]
        :return: [num_nodes, output_dim]
        """
        num_nodes = x.size(0)
        device = x.device

        # Будуємо матрицю суміжності
        adj = build_adjacency(edge_index=edge_index, num_nodes=num_nodes, device=device)

        # Спочатку застосовуємо лінійне перетворення x @ W
        xw = x @ self.weight  # [num_nodes, output_dim]

        # Потім розповсюджуємо інформацію через A_hat
        out = adj @ xw + self.bias  # [num_nodes, output_dim]

        return out


class TrafficGraphNeuralNetwork(nn.Module):
    """
    Спрощена GNN-модель для регресії по вузлах.

    Архітектура:
    - GraphConvLayer(input_features -> hidden_units) + ReLU
    - GraphConvLayer(hidden_units -> output_features)

    Використовується як:
        model = TrafficGraphNeuralNetwork(input_features=1, hidden_units=16, output_features=1)
        y_pred = model(x, edge_index)
    """

    def __init__(self, input_features: int, hidden_units: int, output_features: int) -> None:
        """
        :param input_features: кількість ознак на вузол
        :param hidden_units: розмір прихованого шару
        :param output_features: кількість вихідних ознак на вузол (для регресії зазвичай 1)
        """
        super().__init__()
        self.conv1 = SimpleGraphConvLayer(input_dim=input_features, output_dim=hidden_units)
        self.conv2 = SimpleGraphConvLayer(input_dim=hidden_units, output_dim=output_features)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        :param x: [num_nodes, input_features]
        :param edge_index: [2, num_edges]
        :return: [num_nodes, output_features]
        """
        # Перший графовий шар + ReLU
        h = self.conv1(x, edge_index)
        h = self.activation(h)

        # Другий графовий шар (без додаткової нелінійності на виході)
        out = self.conv2(h, edge_index)

        return out