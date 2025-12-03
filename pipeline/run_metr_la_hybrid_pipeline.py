"""
pipeline/run_metr_la_hybrid_pipeline.py

Фінальний сценарій запуску повного експерименту METR-LA HYBRID. ВИКОРИСТАЄМО В ФІНАЛЬНОМУ ПЕЙПЛАЙНІ!

Послідовність кроків:
  1) Обробка / ETL вихідних даних METR-LA -> data/processed/metr_la.pt
     (викликає training.etl_metr_la)

  2) Формування гібридної моделі:
     - створення HybridTrafficGraphNeuralNetwork
     - підрахунок кількості параметрів
     - збереження початкового стану ваг
     (викликає training.build_metr_la_hybrid_model)

  3) Навчання гібридної моделі на підготовлених даних
     (викликає training.train_metr_la_hybrid)

  4) Аналіз якості моделі на TEST-спліті METR-LA
     (викликає training.evaluate_metr_la_hybrid)

  5) Побудова графіків з результатами експерименту
     (викликає analysis/plot_metr_la_results_plotly.py)

УВАГА:
- Скрипт передбачає, що:
  * запущений з кореня проєкту (де лежать папки training, analysis, data, models),
  * активоване віртуальне середовище (.venv),
  * файл data/raw/METR-LA.csv вже завантажений,
  * всі залежності (torch, torch_geometric, plotly, fastapi тощо) встановлені.
"""

from pathlib import Path
import subprocess
import sys


# Корінь проєкту (наприклад: /Users/alexborlis/PycharmProjects/traffic-gnn-interface)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Поточний інтерпретатор Python (з активованого .venv)
PYTHON_BIN = sys.executable


def run_step(description: str, args: list[str]) -> None:
    """
    Допоміжна функція для запуску окремого кроку пайплайна.

    :param description: опис кроку (для логів у консолі).
    :param args: Список аргументів командного рядка, який буде передано
                 до subprocess.run, наприклад:
                 [PYTHON_BIN, "-m", "training.etl_metr_la", "--input-csv", ...]
    """
    print(f"\n=== {description} ===")
    print("Команда:", " ".join(str(a) for a in args))

    # Виконуємо підпроцес у корені проєкту, щоб імпорти працювали коректно
    result = subprocess.run(args, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        # Якщо крок впав — зупиняємо весь пайплайн з явною помилкою
        raise SystemExit(
            f"❌ Крок '{description}' завершився з помилкою (exit code={result.returncode})."
        )

    print(f"✅ Крок '{description}' виконано успішно.")


def main() -> None:
    """
    Основна функція, яка послідовно виконує всі 5 кроків.
    """

    # 1) ETL: підготовка даних METR-LA
    run_step(
        "1/5 Підготовка та нормалізація даних METR-LA (ETL)",
        [
            PYTHON_BIN,
            "-m",
            "training.etl_metr_la",
            "--input-csv",
            "data/raw/METR-LA.csv",
            "--output-pt",
            "data/processed/metr_la.pt",
            "--horizon",
            "12",  # той самий горизонт, який ти вже використовував
        ],
    )

    # 2) Формування моделі (структура + початковий стан)
    run_step(
        "2/5 Формування гібридної моделі (GNN + Transformer) для METR-LA",
        [
            PYTHON_BIN,
            "-m",
            "training.build_metr_la_hybrid_model",
        ],
    )

    # 3) Навчання моделі
    run_step(
        "3/5 Навчання гібридної моделі на METR-LA",
        [
            PYTHON_BIN,
            "-m",
            "training.train_metr_la_hybrid",
        ],
    )

    # 4) Аналіз якості (evaluation)
    run_step(
        "4/5 Аналіз якості гібридної моделі на TEST-спліті METR-LA",
        [
            PYTHON_BIN,
            "-m",
            "training.evaluate_metr_la_hybrid",
        ],
    )

    # 5) Побудова графіків (Plotly)
    run_step(
        "5/5 Побудова графіків результатів (Plotly)",
        [
            PYTHON_BIN,
            "analysis/plot_metr_la_results_plotly.py",
        ],
    )

    print(
        "  \n Повний експеримент METR-LA HYBRID завершено.\n"
        "   - Підготовлені дані: data/processed/metr_la.pt\n"
        "   - Модель: models/traffic_gnn_metrla_hybrid.pt (та *_init.pt)\n"
        "   - Метрики дивись у виводі evaluate-скрипта\n"
        "   - Графіки: analysis/figures/*.html (відкривай у браузері)\n"
    )


if __name__ == "__main__":
    main()