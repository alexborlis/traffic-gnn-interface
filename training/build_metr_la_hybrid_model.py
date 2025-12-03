"""
training/build_metr_la_hybrid_model.py

Допоміжний скрипт для формування гібридної моделі
HybridTrafficGraphNeuralNetwork для задачі METR-LA.

Його задача:
  1) Створити модель з фіксованими гіперпараметрами.
  2) Порахувати кількість параметрів.
  3) Зберегти початковий (ненавчений) стан моделі на диск
     — для репродуктивності та документування експерименту.
"""

from pathlib import Path

import torch

from models.traffic_gnn import HybridTrafficGraphNeuralNetwork


def build_hybrid_model() -> HybridTrafficGraphNeuralNetwork:
    """
    Створює екземпляр гібридної моделі з фіксованими гіперпараметрами.

    УВАГА:
    - Назви параметрів і значення повинні бути узгоджені з тими,
      що використовуються у скрипті training.train_metr_la_hybrid.
    - Якщо ти там змінюватимеш розмірність/шари — ОБОВʼЯЗКОВО
      онови ці значення тут, щоб опис моделі в дипломі був консистентним.
    """

    # TODO: якщо у training.train_metr_la_hybrid використовується інший
    # набір гіперпараметрів — підстав сюди ті ж значення.
    model = HybridTrafficGraphNeuralNetwork(
        input_features=1,          # для METR-LA ми використовуємо 1 ознаку (швидкість)
        hidden_units=64,          # розмір прихованого шару GNN/MLP
        output_features=1,        # прогнозована величина (швидкість)
        num_transformer_layers=2, # кількість шарів Transformer
        num_heads=4,              # кількість голів у multi-head attention
        dropout=0.1,              # ймовірність dropout
    )

    return model


def main() -> None:
    """
    Точка входу для CLI-режиму.

    Робить наступне:
      1. Створює модель.
      2. Виводить коротку інформацію про модель:
         - кількість параметрів,
         - пристрій (CPU/GPU).
      3. Зберігає початковий стан моделі до файлу
         models/traffic_gnn_metrla_hybrid_init.pt
    """
    device = torch.device("cpu")  # На цьому етапі нам достатньо CPU
    model = build_hybrid_model().to(device)

    # Рахуємо кількість параметрів
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Готуємо директорію для збереження ваг
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Файл з початковим станом моделі
    init_weights_path = models_dir / "traffic_gnn_metrla_hybrid_init.pt"
    torch.save(model.state_dict(), init_weights_path)

    print("\n[BUILD MODEL] HybridTrafficGraphNeuralNetwork для METR-LA створено.")
    print(f"[BUILD MODEL] Пристрій: {device}")
    print(f"[BUILD MODEL] Загальна кількість параметрів:   {total_params}")
    print(f"[BUILD MODEL] Тренованих параметрів:            {trainable_params}")
    print(f"[BUILD MODEL] Початкові (ненавчені) ваги збережено до: {init_weights_path}\n")


if __name__ == "__main__":
    main()