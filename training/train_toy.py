# training/train_toy.py

"""
Перший простий train-loop для GNN-моделі на toy-датаcеті.

Що робить:
- створює ToyTrafficDataset;
- створює модель TrafficGraphNeuralNetwork;
- тренує кілька епох (наприклад, 100) із MSE-loss;
- зберігає натреновані ваги до файлу models/traffic_gnn_toy.pt.

Цей скрипт потрібен, щоб:
- перевірити, що модель взагалі вміє тренуватися;
- отримати перший артефакт моделі, який потім можна підключити до API.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.toy_dataset import ToyTrafficDataset
from models.traffic_gnn import TrafficGraphNeuralNetwork


def main() -> None:
    # ==== 1. Базові налаштування ===========================================
    device = torch.device("cpu")  # поки що CPU, далі можна буде додати cuda
    num_epochs = 100
    learning_rate = 1e-2

    # ==== 2. Датасет і DataLoader ==========================================
    dataset = ToyTrafficDataset(num_samples=512, seed=42)

    # Для простоти використовуємо batch_size=1 (1 граф / batch)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # ==== 3. Модель ========================================================
    # Наш TrafficGraphNeuralNetwork очікує:
    #   - input_features: кількість ознак на вузол (1 у нашому toy-датаcеті)
    #   - hidden_units: розмір прихованого шару
    #   - output_features: скільки значень прогнозуємо на вузол (1)
    model = TrafficGraphNeuralNetwork(
        input_features=1,
        hidden_units=16,
        output_features=1,
    ).to(device)

    # ==== 4. Функція втрат і оптимізатор ===================================
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ==== 5. Цикл навчання =================================================
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # batch = (x, edge_index, y)
            x, edge_index, y = batch
            # Після DataLoader з batch_size=1 форми такі:
            # x: [1, num_nodes, num_features]
            # edge_index: [1, 2, num_edges]
            # y: [1, num_nodes]

            # Прибираємо batch-вимір:
            x = x.squeeze(0).to(device)  # [num_nodes, num_features] = [3, 1]
            edge_index = edge_index.squeeze(0).to(device)  # [2, num_edges]
            y = y.squeeze(0).to(device)  # [num_nodes] = [3]

            optimizer.zero_grad()

            # TrafficGraphNeuralNetwork повертає [num_nodes, output_features]
            y_pred = model(x, edge_index).squeeze(-1)  # [3]

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")

    # ==== 6. Збереження ваг моделі =========================================
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = models_dir / "traffic_gnn_toy.pt"
    torch.save(model.state_dict(), artifact_path)

    print(f"Модель збережено до: {artifact_path.resolve()}")


if __name__ == "__main__":
    main()