# pipeline/run_metr_la_hybrid_pipeline.py

"""
Головний скрипт пайплайна METR-LA HYBRID (5 кроків).

Кроки:
1) ETL:
   - читання сирого CSV METR-LA;
   - нормалізація, формування тензорів X, y, edge_index;
   - збереження до data/processed/metr_la.pt.

2) BUILD MODEL CONFIG:
   - формування та збереження JSON-конфігурації гібридної моделі
     (TrafficGraphNeuralNetworkHybrid) у models/traffic_gnn_metrla_hybrid_config.json.

3) TRAIN:
   - запуск training.train_metr_la_hybrid;
   - навчання моделі та збереження ваг до models/traffic_gnn_metrla_hybrid.pt.

4) EVAL:
   - запуск training.evaluate_metr_la_hybrid;
   - підрахунок MSE, MAE, RMSE, MAPE на TEST-спліті, вивід у консоль.

5) PLOT:
   - запуск analysis/plot_metr_la_hybrid_results_plotly.py;
   - повторне оцінювання всередині, парсинг метрик та побудова bar-chart (Plotly).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Використовуємо той самий інтерпретатор, з якого запущений скрипт
PYTHON_BIN = sys.executable


def run_step(title: str, cmd: list[str]) -> None:
    """
    Допоміжна функція для запуску одного кроку пайплайна.

    :param title: Людиночитна назва кроку (для логів).
    :param cmd:   Команда (список аргументів), яку треба виконати через subprocess.run.
    """
    print("\n" + "=" * 80)
    print(f"=== {title} ===")
    print("Команда:", " ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"❌ Крок '{title}' завершився з помилкою (exit code={result.returncode}). "
            "Перевір логи вище."
        )

    print(f"✅ Крок '{title}' виконано успішно.")


def main() -> None:
    # Корінь проєкту (.. відносно файлу пайплайна)
    project_root = Path(__file__).resolve().parent.parent

    # Шляхи до даних
    data_raw = project_root / "data" / "raw" / "METR-LA.csv"
    data_processed = project_root / "data" / "processed" / "metr_la.pt"

    # 1/5: ETL
    run_step(
        "1/5 Підготовка та нормалізація даних METR-LA (ETL)",
        [
            PYTHON_BIN,
            "-m",
            "training.etl_metr_la",
            "--input-csv",
            str(data_raw),
            "--output-pt",
            str(data_processed),
            "--horizon",
            "12",
        ],
    )

    # 2/5: BUILD MODEL CONFIG
    run_step(
        "2/5 Формування конфігурації гібридної моделі (GNN + Transformer)",
        [
            PYTHON_BIN,
            "-m",
            "training.build_metr_la_hybrid_model",
        ],
    )

    # 3/5: TRAIN
    run_step(
        "3/5 Навчання гібридної моделі на METR-LA",
        [
            PYTHON_BIN,
            "-m",
            "training.train_metr_la_hybrid",
        ],
    )

    # 4/5: EVAL
    run_step(
        "4/5 Оцінювання якості гібридної моделі на тестовому наборі",
        [
            PYTHON_BIN,
            "-m",
            "training.evaluate_metr_la_hybrid",
        ],
    )

    # 5/5: PLOT
    run_step(
        "5/5 Побудова графіків метрик (Plotly)",
        [
            PYTHON_BIN,
            "analysis/plot_metr_la_hybrid_results_plotly.py",
        ],
    )

    print("\n🎉 Усі 5 кроків пайплайна METR-LA HYBRID виконано успішно.")


if __name__ == "__main__":
    main()