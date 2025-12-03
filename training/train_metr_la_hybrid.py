# training/train_metr_la_hybrid.py

"""
Навчання гібридної моделі HybridTrafficGNNTransformer на METR-LA.

Ця модель поєднує:
- GNN-шари для просторової агрегації по графу сенсорів;
- TransformerEncoder для глобального self-attention між вузлами;
- MLP-голову для перетворення ембедінгів у прогнози швидкості.

Сценарій аналогічний train_metr_la.py, але модель — інша.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from models.traffic_gnn_transformer import HybridTrafficGNNTransformer
from training.metr_la_dataset import MetrLaDataset, MetrLaDatasetConfig
from training.metrics import regression_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Одна епоха навчання гібридної моделі.

    Повертає:
        avg_loss, avg_mae, avg_rmse, avg_mape
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    num_batches = 0

    for batch in dataloader:
        x_t, edge_index, y_t = batch
        # Після DataLoader з batch_size=1:
        # x_t:        [1, N, 1]
        # edge_index: [1, 2, E]
        # y_t:        [1, N]
        x_t = x_t.squeeze(0).to(device)               # [N, 1]
        y_t = y_t.squeeze(0).to(device)               # [N]
        edge_index = edge_index.squeeze(0).to(device) # [2, E]

        optimizer.zero_grad()

        # Гібридна модель повертає [N, 1]
        y_hat: Tensor = model(x_t, edge_index).squeeze(-1)  # [N]

        loss: Tensor = loss_fn(y_hat, y_t)
        loss.backward()
        optimizer.step()

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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """
    Оцінка моделі на валідаційному датасеті.

    Повертає:
        avg_loss, avg_mae, avg_rmse, avg_mape
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
            x_t = x_t.squeeze(0).to(device)               # [N, 1]
            y_t = y_t.squeeze(0).to(device)               # [N]
            edge_index = edge_index.squeeze(0).to(device) # [2, E]

            y_hat: Tensor = model(x_t, edge_index).squeeze(-1)

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
    # === 1. Базові налаштування =============================================
    device = torch.device("cpu")

    data_path = Path("data/processed/metr_la.pt")
    model_output_path = Path("models") / "traffic_gnn_metrla_hybrid.pt"
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Гіперпараметри гібридної моделі
    input_features = 1
    hidden_units = 64           # трохи більше, ніж у базовій GNN
    output_features = 1
    num_transformer_layers = 2
    num_heads = 4
    dropout = 0.1

    num_epochs = 5
    learning_rate = 1e-3
    weight_decay = 1e-5
    batch_size = 1

    print("[TRAIN METR-LA HYBRID] Створюємо датасети ...")
    train_dataset = MetrLaDataset(
        MetrLaDatasetConfig(data_path=data_path, split="train")
    )
    val_dataset = MetrLaDataset(
        MetrLaDatasetConfig(data_path=data_path, split="val")
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"[TRAIN METR-LA HYBRID] Train samples: {len(train_dataset)}, "
        f"Val samples: {len(val_dataset)}, "
        f"Nodes per sample: {train_dataset.N}"
    )

    # === 2. Ініціалізація гібридної моделі ==================================
    model = HybridTrafficGNNTransformer(
        input_features=input_features,
        hidden_units=hidden_units,
        output_features=output_features,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss()

    # === 3. Цикл навчання по епохах ==========================================
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n[TRAIN METR-LA HYBRID] Epoch {epoch:03d}/{num_epochs}")

        train_loss, train_mae, train_rmse, train_mape = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_mae, val_rmse, val_mape = evaluate(
            model, val_loader, loss_fn, device
        )

        print(
            f"[TRAIN METR-LA HYBRID] Train: "
            f"loss={train_loss:.4f}, MAE={train_mae:.4f}, "
            f"RMSE={train_rmse:.4f}, MAPE={train_mape:.2f}%"
        )
        print(
            f"[TRAIN METR-LA HYBRID] Val:   "
            f"loss={val_loss:.4f}, MAE={val_mae:.4f}, "
            f"RMSE={val_rmse:.4f}, MAPE={val_mape:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_output_path)
            print(
                f"[TRAIN METR-LA HYBRID] Краща модель оновлена, "
                f"збережено до {model_output_path}"
            )

    print("\n[TRAIN METR-LA HYBRID] Навчання завершено.")
    print(f"[TRAIN METR-LA HYBRID] Найкращий val loss: {best_val_loss:.4f}")
    print(f"[TRAIN METR-LA HYBRID] Фінальна модель: {model_output_path}")


if __name__ == "__main__":
    main()