"""
models/traffic_gnn.py

Містить дві моделі:
- TrafficGraphNeuralNetwork: базова GNN-модель для прогнозу швидкості по вузлах графа.
- HybridTrafficGraphNeuralNetwork: гібридна модель (GNN + Transformer Encoder),
  яка спочатку отримує просторові ознаки через GNN, а потім моделює
  залежності між вузлами через трансформер.

Обидві моделі мають однаковий forward-інтерфейс:
    forward(x, edge_index) -> Tensor [N, 1]

де:
    x          – Tensor [N, input_dim] (для METR-LA це [N, 1])
    edge_index – Tensor [2, E] зі списком ребер (джерело, призначення)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Допоміжні функції
# -----------------------------------------------------------------------------


def build_adjacency(edge_index: Tensor, num_nodes: int, device: torch.device) -> Tensor:
    """
    Будує ненормовану матрицю суміжності A розміром [N, N] з edge_index.

    edge_index: Tensor форми [2, E],
        edge_index[0] = список src (індекси вузлів-джерел),
        edge_index[1] = список dst (індекси вузлів-призначень).

    Повертає:
        A (Tensor [N, N]) – 0/1 матриця суміжності.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index очікується форми [2, E], а отримано {tuple(edge_index.shape)}")

    src = edge_index[0]
    dst = edge_index[1]

    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[src, dst] = 1.0
    # За бажанням можна зробити граф неорієнтованим:
    # adj[dst, src] = 1.0
    return adj


def normalize_adjacency(adj: Tensor) -> Tensor:
    """
    Нормалізація A з додаванням самозвʼязків:
        A_hat = D^{-1/2} (A + I) D^{-1/2}
    """
    num_nodes = adj.size(0)
    device = adj.device

    eye = torch.eye(num_nodes, device=device)
    adj_hat = adj + eye

    deg = adj_hat.sum(dim=1)  # [N]
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj_hat @ D_inv_sqrt  # [N, N]


# -----------------------------------------------------------------------------
# GCN-подібний шар
# -----------------------------------------------------------------------------


class GraphConvLayer(nn.Module):
    """
    Простіший варіант GCN-шару:
        H_out = ReLU(Â H_in W + b)

    де Â – нормалізована матриця суміжності.
    """

    def __init__(self, in_dim: int, out_dim: int, activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        """
        x:        [N, F_in]
        adj_norm: [N, N] – нормалізована матриця суміжності
        """
        h = adj_norm @ x  # [N, F_in]
        h = self.linear(h)  # [N, F_out]
        if self.activation:
            h = F.relu(h)
        return h


# -----------------------------------------------------------------------------
# Базова GNN-модель
# -----------------------------------------------------------------------------


class TrafficGraphNeuralNetwork(nn.Module):
    """
    Базова GNN-модель для прогнозу швидкості на кожному вузлі графа.

    Параметри:
        input_dim:   розмірність вхідних ознак вузла (для METR-LA – 1: швидкість)
        hidden_dim:  розмірність прихованого простору
        output_dim:  розмірність виходу (1 – скаляр: прогноз швидкості)
        num_layers:  кількість GNN-шарів (мінімум 1)
        dropout:     dropout між шарами
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.1,
        **_: dict,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers має бути >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        layers = []

        # Перший шар: input_dim -> hidden_dim
        layers.append(GraphConvLayer(input_dim, hidden_dim, activation=True))

        # Проміжні шари hidden_dim -> hidden_dim
        for _i in range(num_layers - 2):
            layers.append(GraphConvLayer(hidden_dim, hidden_dim, activation=True))

        # Останній GNN-шар (якщо шарів більше 1)
        if num_layers > 1:
            layers.append(GraphConvLayer(hidden_dim, hidden_dim, activation=True))

        self.gnn_layers = nn.ModuleList(layers)

        # Вихідний лінійний шар до scalar
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        x:          [N, input_dim]
        edge_index: [2, E]

        Повертає:
            y_hat: [N, output_dim] (для нас – [N, 1])
        """
        device = x.device
        num_nodes = x.size(0)

        adj = build_adjacency(edge_index=edge_index, num_nodes=num_nodes, device=device)
        adj_norm = normalize_adjacency(adj)  # [N, N]

        h = x  # [N, input_dim]
        for layer in self.gnn_layers:
            h = layer(h, adj_norm)  # [N, hidden_dim]
            h = self.dropout(h)

        out = self.output_layer(h)  # [N, output_dim]
        return out


# -----------------------------------------------------------------------------
# Гібридна модель: GNN + Transformer Encoder
# -----------------------------------------------------------------------------


class HybridTrafficGraphNeuralNetwork(nn.Module):
    """
    Гібридна модель для прогнозу трафіку: GNN + Transformer Encoder.

    Схема:
        1) GNN-шари витягують просторові ознаки вузлів (embeddings).
        2) Transformer Encoder моделює довготривалі залежності між вузлами
           як над послідовністю вузлів.
        3) Лінійний шар проєктує ембедінги в скаляр швидкості.

    Параметри (мінімальний набір, який ми точно використовуємо):
        input_dim:        розмірність вхідних ознак вузла (1 для METR-LA)
        hidden_dim:       розмірність GNN-ембедінгів / dim трансформера
        output_dim:       розмірність виходу (1 – швидкість)
        num_gnn_layers:   кількість GNN-шарів
        num_heads:        кількість голів у multi-head attention
        dropout:          dropout і в GNN, і в трансформері

    Інші параметри, які могли опинитись у JSON-конфігу, безпечно
    приймаються через **kwargs, щоб не ламати сумісність.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        transformer_num_layers: int = 2,
        **_: dict,
    ):
        super().__init__()

        if num_gnn_layers < 1:
            raise ValueError("num_gnn_layers має бути >= 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_heads
        self.dropout_layer = nn.Dropout(dropout)

        # ----- GNN-блок -----
        gnn_layers = []

        # Перший шар: input_dim -> hidden_dim
        gnn_layers.append(GraphConvLayer(input_dim, hidden_dim, activation=True))

        # Проміжні GNN-шари hidden_dim -> hidden_dim
        for _i in range(num_gnn_layers - 1):
            gnn_layers.append(GraphConvLayer(hidden_dim, hidden_dim, activation=True))

        self.gnn_layers = nn.ModuleList(gnn_layers)

        # ----- Transformer Encoder поверх вузлів -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,  # [batch, seq_len, dim]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )

        # ----- Вихідний шар -----
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        x:          [N, input_dim]
        edge_index: [2, E]

        Повертає:
            y_hat: [N, output_dim] (для нас – [N, 1])
        """
        device = x.device
        num_nodes = x.size(0)

        # --- GNN частина ---
        adj = build_adjacency(edge_index=edge_index, num_nodes=num_nodes, device=device)
        adj_norm = normalize_adjacency(adj)  # [N, N]

        h = x  # [N, input_dim]
        for layer in self.gnn_layers:
            h = layer(h, adj_norm)  # [N, hidden_dim]
            h = self.dropout_layer(h)

        # --- Transformer частина ---
        # Розглядаємо вузли як послідовність, batch=1:
        # h_seq: [1, N, hidden_dim]
        h_seq = h.unsqueeze(0)
        h_enc = self.transformer(h_seq)  # [1, N, hidden_dim]
        h_enc = self.dropout_layer(h_enc)

        h_final = h_enc.squeeze(0)  # [N, hidden_dim]

        out = self.output_layer(h_final)  # [N, output_dim]
        return out