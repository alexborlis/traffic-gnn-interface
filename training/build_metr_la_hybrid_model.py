# training/build_metr_la_hybrid_model.py

"""
Скрипт 2/5: формування конфігурації гібридної моделі GNN + Transformer для METR-LA.

Ідея:
- Ми не створюємо сам PyTorch-модуль (це робить train-скрипт).
- Тут ми фіксуємо архітектуру та гіперпараметри в окремому JSON-файлі,
  щоб:
  * мати "артефакт" для диплому (можна вставити в текст/додаток);
  * у майбутньому, за бажання, читати цю конфігурацію з сервісного коду.

ВАЖЛИВО:
Параметри мають узгоджуватися з тим, що використовується у training.train_metr_la_hybrid.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    # Каталог для моделей (гарантуємо, що існує)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Шлях до конфіг-файлу моделі
    config_path = models_dir / "traffic_gnn_metrla_hybrid_config.json"

    # Базова конфігурація гібридної моделі.
    # ЦІ ЗНАЧЕННЯ ПОВИННІ ВІДПОВІДАТИ training.train_metr_la_hybrid
    config = {
        "model_name": "TrafficGraphNeuralNetworkHybrid",  # умовна назва архітектури
        "input_features": 1,          # одна ознака: швидкість/потік для сенсора
        "hidden_units": 64,           # розмір прихованого простору GNN/MLP
        "output_features": 1,         # прогнозуємо 1 значення на вузол
        "num_transformer_layers": 2,  # кількість шарів Transformer-частини
        "num_heads": 4,               # кількість attention heads
        "dropout": 0.1,               # дропаут у прихованих шарах
        "horizon": 12,                # горизонт прогнозу H (як у etl_metr_la)
        "num_nodes": 207,             # кількість сенсорів у METR-LA
    }

    # Записуємо конфіг у JSON, з відступами для зручного читання
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"[BUILD METR-LA HYBRID MODEL] Конфігурацію моделі збережено до {config_path}"
    )


if __name__ == "__main__":
    main()