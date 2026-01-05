import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go

# ----------------- Базові шляхи -----------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "metr_la.pt"
MODEL_PATH = ROOT / "models" / "traffic_gnn_metrla_hybrid.pt"

PLOTS_DIR = ROOT / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_TS = PLOTS_DIR / "metr_la_hybrid_sensor0_timeseries.html"
OUT_SCATTER = PLOTS_DIR / "metr_la_hybrid_sensor0_scatter.html"
OUT_HIST = PLOTS_DIR / "metr_la_hybrid_sensor0_abs_error_hist.html"


# ----------------- Допоміжні GNN-класи -----------------


class GraphConvolution(nn.Module):
    """
    Проста графова згортка:
    h' = ReLU(Â x W), де Â — нормалізована adjacency matrix з self-loop.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        x: [N, F_in]
        edge_index: [2, E] (джерело, приймач)
        """
        device = x.device
        row, col = edge_index  # [E], [E]

        # A (без ваг) + self-loops
        adj = torch.zeros((num_nodes, num_nodes), device=device)
        adj[row, col] = 1.0
        adj[col, row] = 1.0  # двонапрямний граф
        adj = adj + torch.eye(num_nodes, device=device)

        # Нормалізація по ступеню: A_hat = A / deg
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        adj_norm = adj / deg

        # Агрегація та лінійне перетворення
        h = adj_norm @ x  # [N, F_in]
        h = self.linear(h)  # [N, F_out]
        h = F.relu(h)
        return h


class HybridTrafficGraphNeuralNetwork(nn.Module):
    """
    Гібридна модель: 2 GNN-конволюції + TransformerEncoder + MLP head.

    Імена шарів підібрані так, щоб відповідати state_dict:
    - gnn_conv1.linear.weight / bias
    - gnn_conv2.linear.weight / bias
    - transformer_encoder.layers.0/1.***
    - head.0.weight, head.0.bias, head.2.weight, head.2.bias
    """

    def __init__(
        self,
        in_features: int,
        gnn_hidden_dim: int,
        gnn_output_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ff_dim: int,
        head_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gnn_conv1 = GraphConvolution(in_features, gnn_hidden_dim)
        self.gnn_conv2 = GraphConvolution(gnn_hidden_dim, gnn_output_dim)

        # Припускаємо, що gnn_output_dim == d_model (як у більшості гібридних моделей)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,  # щоб працювати з [batch, seq, feat]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # MLP-head: Linear(d_model -> head_hidden_dim) -> ReLU -> Linear(head_hidden_dim -> 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden_dim),
            nn.ReLU(),
            nn.Linear(head_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, F_in]
        edge_index: [2, E]
        Повертає: [N, 1]
        """
        num_nodes = x.size(0)

        h = self.gnn_conv1(x, edge_index, num_nodes)  # [N, hidden]
        h = self.gnn_conv2(h, edge_index, num_nodes)  # [N, d_model]

        # інтерпретуємо вузли як послідовність довжини N
        h_seq = h.unsqueeze(0)  # [1, N, d_model]
        h_enc = self.transformer_encoder(h_seq)  # [1, N, d_model]

        out = self.head(h_enc)  # [1, N, 1]
        out = out.squeeze(0)  # [N, 1]
        return out


# ----------------- Завантаження даних та моделі -----------------


def load_payload():
    payload = torch.load(DATA_PATH, map_location="cpu")
    return payload


def _infer_model_dims_from_state_dict(state_dict: dict):
    """
    Витягуємо розміри шарів із saved state_dict, щоб не залежати від JSON-конфіга.
    """
    in_features = state_dict["gnn_conv1.linear.weight"].shape[1]
    gnn_hidden_dim = state_dict["gnn_conv1.linear.weight"].shape[0]
    gnn_output_dim = state_dict["gnn_conv2.linear.weight"].shape[0]

    d_model = state_dict["transformer_encoder.layers.0.self_attn.in_proj_weight"].shape[1]
    ff_dim = state_dict["transformer_encoder.layers.0.linear1.weight"].shape[0]

    # рахуємо кількість Transformer-шарів
    num_layers = 0
    for key in state_dict.keys():
        if key.startswith("transformer_encoder.layers."):
            parts = key.split(".")
            # transformer_encoder.layers.{idx}.xx
            if len(parts) > 2 and parts[2].isdigit():
                idx = int(parts[2])
                num_layers = max(num_layers, idx + 1)

    # прихований розмір в head
    head_hidden_dim = state_dict["head.0.weight"].shape[0]

    # кількість голів в MultiHeadAttention вибираємо будь-яку, що ділить d_model (наприклад, 4)
    n_heads = 4
    if d_model % 8 == 0:
        n_heads = 8
    elif d_model % 4 == 0:
        n_heads = 4
    elif d_model % 2 == 0:
        n_heads = 2

    return {
        "in_features": in_features,
        "gnn_hidden_dim": gnn_hidden_dim,
        "gnn_output_dim": gnn_output_dim,
        "d_model": d_model,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "head_hidden_dim": head_hidden_dim,
        "n_heads": n_heads,
    }


def load_model(device: torch.device) -> HybridTrafficGraphNeuralNetwork:
    """
    Завантажуємо state_dict, інферимо розміри, створюємо модель з правильною архітектурою
    і підвантажуємо ваги.
    """
    state_dict = torch.load(MODEL_PATH, map_location=device)

    dims = _infer_model_dims_from_state_dict(state_dict)

    model = HybridTrafficGraphNeuralNetwork(
        in_features=dims["in_features"],
        gnn_hidden_dim=dims["gnn_hidden_dim"],
        gnn_output_dim=dims["gnn_output_dim"],
        d_model=dims["d_model"],
        n_heads=dims["n_heads"],
        num_layers=dims["num_layers"],
        ff_dim=dims["ff_dim"],
        head_hidden_dim=dims["head_hidden_dim"],
        dropout=0.1,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----------------- Побудова часових рядів -----------------


def _get_scalar_for_sensor(value, sensor_index: int) -> float:
    """
    mean / std можуть бути або скалярами, або тензорами/масивами по сенсорам.
    Нормалізуємо доступ до одного сенсора.
    """
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return float(value.item())
        return float(value[sensor_index].item())
    if isinstance(value, (list, tuple)):
        return float(value[sensor_index])
    # скаляр
    return float(value)


def build_timeseries_for_sensor(
    sensor_index: int = 0,
    max_points: int = 288,  # ~2 дні при інтервалі 5 хв
) -> Tuple[List[int], List[float], List[float]]:
    device = torch.device("cpu")

    payload = load_payload()
    X = payload["X"]  # [T_eff, N, 1]
    y = payload["y"]  # [T_eff, N]
    edge_index = payload["edge_index"].long().to(device)
    train_ratio = float(payload["train_ratio"])
    val_ratio = float(payload["val_ratio"])

    mean_all = payload["mean"]
    std_all = payload["std"]

    T_eff, num_nodes, _ = X.shape
    assert sensor_index < num_nodes, "sensor_index виходить за межі кількості сенсорів"

    mean_sensor = _get_scalar_for_sensor(mean_all, sensor_index)
    std_sensor = _get_scalar_for_sensor(std_all, sensor_index)

    # Відтворюємо той самий train/val/test split
    t_train = int(T_eff * train_ratio)
    t_val = int(T_eff * val_ratio)
    t_test_start = t_train + t_val
    t_test_end = T_eff

    # Візьмемо підмножину тестового спліту
    indices = list(range(t_test_start, t_test_end))
    if max_points is not None:
        indices = indices[:max_points]

    model = load_model(device)

    # Використовуємо індекс часу для timestamp-дат,
    time_indices: List[int] = []
    y_true_denorm: List[float] = []
    y_pred_denorm: List[float] = []

    with torch.no_grad():
        for local_idx, t in enumerate(indices):
            x_t = X[t].to(device)  # [N, 1]
            y_t = y[t]  # [N]

            # модель очікує [N, F_in]
            x_t_2d = x_t.view(num_nodes, -1)

            pred_t = model(x_t_2d, edge_index).squeeze(-1).cpu()  # [N]

            y_true_val = float(y_t[sensor_index])

            # Значення вже в км/год, додаткова "денормалізація" не потрібна
            y_true_denorm.append(y_true_val)
            y_pred_denorm.append(float(pred_t[sensor_index]))

            # Індекс часу в межах тестового відрізку: 0, 1, 2, ...
            time_indices.append(local_idx)

    return time_indices, y_true_denorm, y_pred_denorm


# ----------------- Побудова графіків (Plotly) -----------------


def plot_timeseries(x, y_true, y_pred, sensor_name: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_true,
            mode="lines+markers",
            name="Фактична швидкість",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines+markers",
            name="Прогноз моделі",
        )
    )

    fig.update_layout(
        title=f"METR-LA HYBRID — часовий ряд для сенсора {sensor_name}",
        xaxis_title="Індекс часу (5-хв інтервали)",
        yaxis_title="Швидкість, км/год",
        legend=dict(x=0.01, y=0.99),
    )

    fig.write_html(str(OUT_TS), include_plotlyjs="cdn")
    print(f"[PLOT TS] Графік збережено до {OUT_TS}")


def plot_scatter(y_true, y_pred, sensor_name: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Спостереження",
            opacity=0.7,
        )
    )

    # Лінія y = x для орієнтиру
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))

    fig.add_shape(
        type="line",
        x0=min_v,
        y0=min_v,
        x1=max_v,
        y1=max_v,
        line=dict(dash="dash"),
    )

    fig.update_layout(
        title=f"METR-LA HYBRID — scatter: факт vs прогноз для сенсора {sensor_name}",
        xaxis_title="Фактична швидкість, км/год",
        yaxis_title="Прогноз моделі, км/год",
    )

    fig.write_html(str(OUT_SCATTER), include_plotlyjs="cdn")
    print(f"[PLOT SCATTER] Графік збережено до {OUT_SCATTER}")


def plot_abs_error_hist(y_true, y_pred, sensor_name: str):
    abs_errors = [abs(p - t) for p, t in zip(y_pred, y_true)]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=abs_errors,
            nbinsx=40,
            name="|помилка|",
        )
    )

    fig.update_layout(
        title=f"METR-LA HYBRID — розподіл абсолютних похибок для сенсора {sensor_name}",
        xaxis_title="|помилка|, км/год",
        yaxis_title="Кількість спостережень",
    )

    fig.write_html(str(OUT_HIST), include_plotlyjs="cdn")
    print(f"[PLOT HIST] Графік збережено до {OUT_HIST}")


# ----------------- main -----------------


def main():
    payload = torch.load(DATA_PATH, map_location="cpu")
    sensor_ids = payload["sensor_ids"]  # список ID сенсорів (рядки)
    sensor_index = 0  # можна змінити на будь-який індекс
    sensor_name = str(sensor_ids[sensor_index])

    print(f"[INFO] Обрано сенсор index={sensor_index}, id={sensor_name}")

    x, y_true, y_pred = build_timeseries_for_sensor(sensor_index=sensor_index)

    plot_timeseries(x, y_true, y_pred, sensor_name)
    plot_scatter(y_true, y_pred, sensor_name)
    plot_abs_error_hist(y_true, y_pred, sensor_name)


if __name__ == "__main__":
    main()