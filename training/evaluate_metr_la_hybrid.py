# training/evaluate_metr_la_hybrid.py

"""
Оцінювання гібридної моделі HybridTrafficGNNTransformer
на тестовій частині датасету METR-LA.

Передумови:
- Файл з даними: data/processed/metr_la.pt
  (згенерований через training/etl_metr_la.py).
- Файл з моделлю: models/traffic_gnn_metrla_hybrid.pt
  (натренований через training/train_metr_la_hybrid.py).

Результат:
- Вивід у консоль підсумкових метрик (MSE, MAE, RMSE, MAPE) на TEST-спліті.
"""
from __future__ import annotations

import json

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from models.traffic_gnn_transformer import HybridTrafficGNNTransformer
from training.metr_la_dataset import MetrLaDataset, MetrLaDatasetConfig
from training.metrics import regression_metrics


def evaluate_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Обчислює середні значення loss, MAE, RMSE, MAPE на заданому датасеті.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x_t, edge_index, y_t = batch
            # Після DataLoader з batch_size=1:
            # x_t:        [1, N, 1]
            # edge_index: [1, 2, E]
            # y_t:        [1, N]
            x_t = x_t.squeeze(0).to(device)               # [N, 1]
            y_t = y_t.squeeze(0).to(device)               # [N]
            edge_index = edge_index.squeeze(0).to(device) # [2, E]

            # Модель повертає [N, 1] → стискаємо останній вимір
            y_hat: Tensor = model(x_t, edge_index).squeeze(-1)  # [N]

            loss: Tensor = loss_fn(y_hat, y_t)

            mae_val, rmse_val, mape_val = regression_metrics(y_t, y_hat)

            total_loss += float(loss.item())
            total_mae += float(mae_val.item())
            total_rmse += float(rmse_val.item())
            total_mape += float(mape_val.item())
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_rmse = total_rmse / num_batches
    avg_mape = total_mape / num_batches

    return avg_loss, avg_mae, avg_rmse, avg_mape


def main() -> None:
    device = torch.device("cpu")

    data_path = Path("data/processed/metr_la.pt")
    model_path = Path("models") / "traffic_gnn_metrla_hybrid.pt"

    if not data_path.is_file():
        raise FileNotFoundError(f"Не знайдено датасет METR-LA: {data_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Не знайдено збережену гібридну модель: {model_path}")

    print("[EVAL METR-LA HYBRID] Створюємо test-датасет ...")
    test_dataset = MetrLaDataset(
        MetrLaDatasetConfig(data_path=data_path, split="test")
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(
        f"[EVAL METR-LA HYBRID] Test samples: {len(test_dataset)}, "
        f"Nodes per sample: {test_dataset.N}"
    )

    # Параметри мають збігатися з training/train_metr_la_hybrid.py
    input_features = 1
    hidden_units = 64
    output_features = 1
    num_transformer_layers = 2
    num_heads = 4
    dropout = 0.1

    model = HybridTrafficGNNTransformer(
        input_features=input_features,
        hidden_units=hidden_units,
        output_features=output_features,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    print(f"[EVAL METR-LA HYBRID] Завантажуємо ваги моделі з {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    loss_fn = nn.MSELoss()

    print("[EVAL METR-LA HYBRID] Обчислюємо метрики на тестовому наборі ...")
    test_loss, test_mae, test_rmse, test_mape = evaluate_on_loader(
        model, test_loader, loss_fn, device
    )

    print("\n=== METR-LA HYBRID: підсумкові метрики на TEST-спліті ===")
    print(f"MSE (loss): {test_loss:.4f}")
    print(f"MAE:        {test_mae:.4f}")
    print(f"RMSE:       {test_rmse:.4f}")
    print(f"MAPE:       {test_mape:.2f} %")
    print("=========================================================")

    metrics = {
        "dataset": "METR-LA",
        "model_type": "HybridTrafficGNNTransformer",
        "loss_mse": float(test_loss),
        "mae": float(test_mae),
        "rmse": float(test_rmse),
        "mape": float(test_mape),
    }

    metrics_path = model_path.parent / "metrics_metrla_hybrid.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[EVAL METR-LA HYBRID] Метрики збережено до {metrics_path}")

if __name__ == "__main__":
    main()