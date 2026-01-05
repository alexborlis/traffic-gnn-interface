# analysis/plot_metr_la_results.py

"""
Побудова графіків для порівняння базової GNN та гібридної моделі
HybridTrafficGNNTransformer на датасеті METR-LA.

- Графік 1: MAE (Val) по епохах для обох моделей.
- Графік 2: стовпчикова діаграма MAE/RMSE на TEST-спліті.

Числа для GNN/Hybrid взято з фактичних логів навчання та оцінювання,
які вже були отримані при запуску train_metr_la.py, train_metr_la_hybrid.py
та evaluate_*.
"""

import matplotlib.pyplot as plt


def plot_val_mae_curves() -> None:
    """
    Малює криві MAE на валідаційному наборі для базової GNN і Hybrid.
    """

    # Епохи (1..5)
    epochs = [1, 2, 3, 4, 5]

    # З НАШИХ ЛОГІВ (GNN):
    # Epoch 001: Val MAE=10.9096
    # Epoch 002: Val MAE=10.9102
    # Epoch 003: Val MAE=10.9944
    # Epoch 004: Val MAE=11.6807
    # Epoch 005: Val MAE=11.1993
    val_mae_gnn = [10.9096, 10.9102, 10.9944, 11.6807, 11.1993]

    # З НАШИХ ЛОГІВ (HYBRID):
    # Epoch 001: Val MAE=9.8258
    # Epoch 002: Val MAE=9.2861
    # Epoch 003: Val MAE=9.9329
    # Epoch 004: Val MAE=9.3830
    # Epoch 005: Val MAE=10.4533
    val_mae_hybrid = [9.8258, 9.2861, 9.9329, 9.3830, 10.4533]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_mae_gnn, marker="o", label="GNN (Val MAE)")
    plt.plot(epochs, val_mae_hybrid, marker="o", label="Hybrid GNN+Transformer (Val MAE)")

    plt.xlabel("Epoch")
    plt.ylabel("MAE на валідації (mph)")
    plt.title("Порівняння MAE на валідації: GNN vs Hybrid GNN+Transformer (METR-LA)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/metr_la_val_mae_gnn_vs_hybrid.png", dpi=200)
    plt.close()


def plot_test_metrics_bar() -> None:
    """
    Малює стовпчикову діаграму MAE та RMSE на TEST-спліті
    для базової GNN та Hybrid-моделі.

    """

    # видає evaluate_metr_la.py: поки що плот поламаний
    mae_gnn_test = 12.4110
    rmse_gnn_test = 16.0130

    mae_hybrid_test = 11.2217
    rmse_hybrid_test = 14.1633

    models = ["GNN (base)", "Hybrid GNN+Transformer"]
    mae_values = [mae_gnn_test, mae_hybrid_test]
    rmse_values = [rmse_gnn_test, rmse_hybrid_test]

    x = range(len(models))

    plt.figure(figsize=(8, 5))

    # ширина стовпчика
    width = 0.35

    # MAE зліва, RMSE справа
    plt.bar([i - width / 2 for i in x], mae_values, width=width, label="MAE")
    plt.bar([i + width / 2 for i in x], rmse_values, width=width, label="RMSE")

    plt.xticks(list(x), models)
    plt.ylabel("Помилка (mph)")
    plt.title("MAE/RMSE на TEST-спліті: базова GNN vs Hybrid GNN+Transformer (METR-LA)")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("figures/metr_la_test_mae_rmse_gnn_vs_hybrid.png", dpi=200)
    plt.close()


def main() -> None:
    import os

    os.makedirs("figures", exist_ok=True)

    plot_val_mae_curves()
    plot_test_metrics_bar()
    print("Графіки збережено до папки figures/")


if __name__ == "__main__":
    main()