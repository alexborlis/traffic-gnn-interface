# models/traffic_gnn_transformer.py

"""
Гібридна модель прогнозування трафіку:
поєднання просторової GNN та Transformer-блоку.

Ідея:
- GNN частина агрегує інформацію по графу дорожньої мережі
  (сусіди кожного вузла впливають на його представлення).
- Transformer частина виконує self-attention між усіма вузлами,
  тобто "дивиться" на весь граф як на послідовність ембедінгів вузлів
  і вчиться виділяти найбільш релевантні зв’язки.

Зверни увагу:
- Ми працюємо з одним часовим зрізом за раз: x має форму [N, F].
  Це добре лягає на поточний MetrLaDataset, де кожен елемент — один момент часу.
- Послідовність для Transformer — це "послідовність вузлів графа":
  sequence length = N, embedding dim = hidden_units.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn


def build_adjacency(
    edge_index: Tensor,
    num_nodes: int,
    device: torch.device | None = None,
) -> Tensor:
    """
    Будує нормалізовану матрицю суміжності A розміру [N, N]
    на основі списку ребер edge_index форми [2, E].

    Логіка:
    - Для кожного ребра (src -> dst) ставимо 1 в A[src, dst].
    - Робимо граф неорієнтованим: дублюємо (dst -> src).
    - Додаємо петлі (self-loops) на діагональ.
    - Робимо рядкову нормалізацію: кожен рядок ділиться на суму
      степенів (щоб уникнути вибуху значень при множенні).

    Це спрощений варіант нормалізації, але цілком підходить
    для базової GNN-архітектури в дипломному проєкті.
    """
    if device is None:
        device = edge_index.device

    # Очікуємо edge_index форми [2, E]
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Очікується edge_index форми [2, E], отримано {edge_index.shape}")

    src = edge_index[0]  # [E]
    dst = edge_index[1]  # [E]

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    # Записуємо ребра (src -> dst) і (dst -> src), щоб отримати неорієнтований граф
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0

    # Додаємо self-loops
    adj += torch.eye(num_nodes, device=device)

    # Рядкова нормалізація: A_norm = D^{-1} A
    deg = adj.sum(dim=1, keepdim=True)  # [N, 1]
    adj_norm = adj / (deg + 1e-6)

    return adj_norm


class GraphConvolution(nn.Module):
    """
    Проста графова згортка:
        h = A_norm @ X @ W + b

    де:
        A_norm — нормалізована матриця суміжності [N, N],
        X      — ознаки вузлів [N, F_in],
        W, b   — параметри лінійного перетворення.

    Таким чином, на кожному кроці кожен вузол отримує
    зважену суміш ознак своїх сусідів.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        :param x:   Tensor [N, F_in]  — ознаки вузлів
        :param adj: Tensor [N, N]     — нормалізована матриця суміжності
        :return:    Tensor [N, F_out] — оновлені ознаки вузлів
        """
        # Агрегуємо інформацію від сусідів
        h = torch.matmul(adj, x)  # [N, F_in]
        # Лінійна трансформація
        h = self.linear(h)        # [N, F_out]
        return h


class HybridTrafficGNNTransformer(nn.Module):
    """
    Гібридна модель:
    - GNN-частина (2 графові згортки) будує просторові ембедінги вузлів;
    - TransformerEncoder додає глобальну (повнозв’язну) self-attention
      між усіма вузлами графа;
    - MLP-голова проєктує ембедінги у скалярні прогнози (наприклад, швидкість).

    Інтерфейс forward:
        x:          [N, F_in]
        edge_index: [2, E]

    Вихід:
        y_hat:      [N, F_out]
    """

    def __init__(
        self,
        input_features: int,
        hidden_units: int,
        output_features: int,
        num_transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_features = input_features
        self.hidden_units = hidden_units
        self.output_features = output_features

        # --- GNN-блок: дві послідовні графові згортки ----------------------
        self.gnn_conv1 = GraphConvolution(input_features, hidden_units)
        self.gnn_conv2 = GraphConvolution(hidden_units, hidden_units)
        self.gnn_activation = nn.ReLU()

        # --- Transformer-блок по вузлах графа -------------------------------
        # Використовуємо вузли як "послідовність", embedding_dim = hidden_units.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout,
            batch_first=True,  # форма [batch, seq, feat]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # --- MLP-голова для остаточного прогнозу ----------------------------
        self.head = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        :param x:          Tensor [N, F_in]   — ознаки вузлів
        :param edge_index: Tensor [2, E]      — індекси ребер графа
        :return:           Tensor [N, F_out]  — прогноз для кожного вузла
        """
        if x.dim() != 2:
            raise ValueError(f"Очікується x форми [N, F], отримано {x.shape}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Очікується edge_index форми [2, E], отримано {edge_index.shape}")

        device = x.device
        num_nodes = x.size(0)

        # 1. Будуємо нормалізовану матрицю суміжності для GNN
        adj = build_adjacency(edge_index=edge_index, num_nodes=num_nodes, device=device)  # [N, N]

        # 2. GNN-частина: дві графові згортки
        h = self.gnn_conv1(x, adj)               # [N, hidden_units]
        h = self.gnn_activation(h)
        h = self.gnn_conv2(h, adj)               # [N, hidden_units]
        h = self.gnn_activation(h)               # [N, hidden_units]

        # 3. Transformer по вузлах:
        #    додаємо batch-вимір: [1, N, hidden_units]
        h_seq = h.unsqueeze(0)                   # [1, N, hidden_units]

        #    self-attention між усіма вузлами графа
        h_transformed = self.transformer_encoder(h_seq)  # [1, N, hidden_units]

        #    прибираємо batch-вимір назад: [N, hidden_units]
        h_final = h_transformed.squeeze(0)

        # 4. MLP-голова: поелементний прогноз для кожного вузла
        out = self.head(h_final)                 # [N, output_features]

        return out