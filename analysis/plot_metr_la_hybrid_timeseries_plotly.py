import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import json
import torch
import plotly.graph_objects as go

from models.traffic_gnn import TrafficGraphNeuralNetwork


DATA_PATH = ROOT / "data" / "processed" / "metr_la.pt"
MODEL_PATH = ROOT / "models" / "traffic_gnn_metrla_hybrid.pt"
CONFIG_PATH = ROOT / "models" / "traffic_gnn_metrla_hybrid_config.json"

PLOTS_DIR = ROOT / "analysis" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_TS = PLOTS_DIR / "metr_la_hybrid_sensor0_timeseries.html"
OUT_SCATTER = PLOTS_DIR / "metr_la_hybrid_sensor0_scatter.html"
OUT_HIST = PLOTS_DIR / "metr_la_hybrid_sensor0_abs_error_hist.html"


# ----------------- Завантаження даних та моделі -----------------


def load_payload():
    payload = torch.load(DATA_PATH, map_location="cpu")
    return payload


def load_model(device: torch.device) -> HybridTrafficGraphNeuralNetwork:
    # Конфіг гібридної моделі з JSON
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = HybridTrafficGraphNeuralNetwork(**cfg)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----------------- Побудова часових рядів -----------------


def build_timeseries_for_sensor(
    sensor_index: int = 0,
    max_points: int = 288,  # ~2 дні при інтервалі 5 хв
):
    device = torch.device("cpu")

    payload = load_payload()
    X = payload["X"]            # [T_eff, N, 1]
    y = payload["y"]            # [T_eff, N]
    edge_index = payload["edge_index"].long().to(device)
    timestamps = payload["timestamps"]  # список строк / таймстемпів
    train_ratio = float(payload["train_ratio"])
    val_ratio = float(payload["val_ratio"])

    mean = float(payload["mean"])
    std = float(payload["std"])

    T_eff, num_nodes, _ = X.shape
    assert sensor_index < num_nodes, "sensor_index виходить за межі кількості сенсорів"

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

    times_str = []
    y_true_denorm = []
    y_pred_denorm = []

    with torch.no_grad():
        for t in indices:
            x_t = X[t].to(device)        # [N, 1]
            y_t = y[t]                  # [N]

            pred_t = model(x_t, edge_index).squeeze(-1).cpu()  # [N]

            y_true_val = float(y_t[sensor_index])
            y_pred_val = float(pred_t[sensor_index])

            # Денормалізація назад до км/год
            y_true_denorm.append(y_true_val * std + mean)
            y_pred_denorm.append(y_pred_val * std + mean)

            ts = timestamps[t]
            times_str.append(str(ts))

    return times_str, y_true_denorm, y_pred_denorm


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
        xaxis_title="Час",
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


def main():
    payload = torch.load(DATA_PATH, map_location="cpu")
    sensor_ids = payload["sensor_ids"]  # список ID сенсорів (рядки)
    sensor_index = 0                    # можна змінити на будь-який індекс
    sensor_name = str(sensor_ids[sensor_index])

    print(f"[INFO] Обрано сенсор index={sensor_index}, id={sensor_name}")

    x, y_true, y_pred = build_timeseries_for_sensor(sensor_index=sensor_index)

    plot_timeseries(x, y_true, y_pred, sensor_name)
    plot_scatter(y_true, y_pred, sensor_name)
    plot_abs_error_hist(y_true, y_pred, sensor_name)


if __name__ == "__main__":
    main()