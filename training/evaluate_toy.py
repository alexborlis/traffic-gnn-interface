# training/evaluate_toy.py

"""
Скрипт для оцінювання якості натренованої GNN-моделі на toy-датаcеті.

Він:
- вантажить TrafficGraphNeuralNetwork і ваги з models/traffic_gnn_toy.pt;
- проходить по всьому ToyTrafficDataset;
- обчислює MAE, RMSE, MAPE по всіх вузлах усіх зразків;
- друкує результати у зручному форматі для подальшої вставки в дипломну роботу.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.traffic_gnn import TrafficGraphNeuralNetwork
from training.toy_dataset import ToyTrafficDataset
from training.metrics import regression_metrics


def main() -> None:
    # ==== 1. Базові налаштування ===========================================
    device = torch.device("cpu")

    # Шлях до файлу з вагами моделі (повинен збігатися з train_toy.py)
    artifact_path = Path("models") / "traffic_gnn_toy.pt"
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Не знайдено файл з вагами моделі: {artifact_path.resolve()}. "
            f"Спочатку запустіть training/train_toy.py"
        )

    # ==== 2. Датасет і DataLoader ==========================================
    dataset = ToyTrafficDataset(num_samples=512, seed=42)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ==== 3. Створення моделі та завантаження ваг ==========================
    model = TrafficGraphNeuralNetwork(
        input_features=1,
        hidden_units=16,
        output_features=1,
    ).to(device)

    state_dict = torch.load(artifact_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ==== 4. Акумуляція всіх передбачень та істинних значень ===============
    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            x, edge_index, y = batch
            # Прибираємо batch-вимір
            x = x.squeeze(0).to(device)              # [num_nodes, 1]
            edge_index = edge_index.squeeze(0).to(device)  # [2, num_edges]
            y = y.squeeze(0).to(device)              # [num_nodes]

            # Передбачення моделі: [num_nodes, 1] -> [num_nodes]
            y_hat = model(x, edge_index).squeeze(-1)

            all_y_true.append(y)
            all_y_pred.append(y_hat)

    # Об'єднуємо усі зразки в один великий тензор
    y_true_full = torch.cat(all_y_true, dim=0)  # [num_samples * num_nodes]
    y_pred_full = torch.cat(all_y_pred, dim=0)  # [num_samples * num_nodes]

    # ==== 5. Обчислення метрик ==============================================
    mae_value, rmse_value, mape_value = regression_metrics(y_true_full, y_pred_full)

    # ==== 6. Вивід результатів ==============================================
    print("=== Evaluation on ToyTrafficDataset ===")
    print(f"Samples (nodes total): {y_true_full.numel()}")
    print(f"MAE  (mean absolute error):       {mae_value.item():.6f}")
    print(f"RMSE (root mean squared error):   {rmse_value.item():.6f}")
    print(f"MAPE (mean absolute percentage):  {mape_value.item():.2f} %")


if __name__ == "__main__":
    main()