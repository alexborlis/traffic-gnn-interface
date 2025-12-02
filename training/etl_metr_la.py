"""
ETL для датасету METR-LA у форматі CSV (Zenodo).

Кроки:
1. Зчитати CSV (формат: часова колонка + 207 колонок сенсорів).
2. Знайти колонку часу та перетворити в pd.Timestamp.
3. Відсортувати за часом, прибрати дублі.
4. Обробити пропуски та екстремуми швидкості.
5. Розбити на train/val/test (по часу).
6. Порахувати mean/std по train для кожного сенсора, нормалізувати.
7. Сформувати X [T_eff, N, 1] та y [T_eff, N] з горизонтом прогнозу H.
8. Побудувати простий edge_index (кільце) як тимчасову топологію.
9. Зберегти все в один .pt-файл.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor


def detect_time_column(df: pd.DataFrame) -> str:
    """
    Автоматично шукає колонку, що містить час.

    Евристики:
    - якщо є колонка з назвою, що містить 'time' або 'date' -> використовуємо її;
    - інакше припускаємо, що перша колонка — timestamp.
    """
    lower_cols = [c.lower() for c in df.columns]
    for col, lower in zip(df.columns, lower_cols):
        if "time" in lower or "date" in lower:
            return col
    # fallback: перша колонка
    return df.columns[0]


def clean_speeds(
    values: pd.DataFrame,
    min_speed: float = 0.0,
    max_speed: float = 120.0,
) -> pd.DataFrame:
    """
    Очищує матрицю швидкостей:
    - значення < min_speed або > max_speed вважаємо аномаліями -> ставимо NaN.
    """
    cleaned = values.copy()
    mask_invalid = (cleaned < min_speed) | (cleaned > max_speed)
    cleaned = cleaned.mask(mask_invalid, np.nan)
    return cleaned


def fill_missing(values: pd.DataFrame) -> pd.DataFrame:
    """
    Заповнює пропуски по кожному сенсору:
    - спочатку forward-fill;
    - потім, якщо на початку залишились NaN, заповнюємо їх середнім по колонці.
    """
    filled = values.copy()
    # forward-fill по часовій осі
    filled = filled.ffill(axis=0)
    # якщо на початку колонки залишились NaN -> заповнюємо середнім
    col_means = filled.mean(axis=0, skipna=True)
    filled = filled.fillna(col_means)
    return filled


def train_val_test_split(
    data: Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Розбиває по часовій осі (перший вимір T) на train / val / test.
    """
    T = data.shape[0]
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test


def compute_normalization_stats(train: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Обчислює mean/std по кожному сенсору (нормалізація per-node).

    Формат train: [T_train, N, 1]
    Повертає:
        mean: [N]
        std:  [N]
    """
    # Стискаємо часову вісь T
    # train: [T, N, 1] -> [T, N]
    train_2d = train.squeeze(-1)
    mean = train_2d.mean(dim=0)  # [N]
    std = train_2d.std(dim=0)    # [N]
    # захист від нульового std
    std = torch.where(std < 1e-6, torch.tensor(1.0, device=std.device), std)
    return mean, std


def normalize_data(data: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Нормалізує дані x_norm = (x - mean) / std.

    data: [T, N, 1]
    mean, std: [N]
    """
    # розширюємо mean/std до [1, N, 1]
    mean = mean.view(1, -1, 1)
    std = std.view(1, -1, 1)
    return (data - mean) / std


def build_edge_index(num_nodes: int) -> Tensor:
    """
    Тимчасово будуємо простий "кільцевий" граф:

        0 -> 1 -> 2 -> ... -> N-1 -> 0

    Формат: [2, E], як прийнято в PyTorch Geometric.
    """
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)

    src = []
    dst = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        src.append(i)
        dst.append(j)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def build_x_y(
    speeds: Tensor,
    horizon: int,
) -> Tuple[Tensor, Tensor]:
    """
    Формує X та y для прогнозу на horizon кроків вперед.

    speeds: [T, N] (швидкості після очищення, але ДО нормалізації)
    horizon: H

    Повертає:
        X: [T_eff, N, 1]
        y: [T_eff, N]
    де T_eff = T - H
    """
    T, N = speeds.shape
    if horizon >= T:
        raise ValueError(f"horizon={horizon} занадто великий для T={T}")

    T_eff = T - horizon
    # X бере перші T_eff кроків
    X = speeds[:T_eff]              # [T_eff, N]
    # y бере значення, зсунуті на H вперед
    y = speeds[horizon:]            # [T_eff, N]

    # додаємо ось ознак
    X = X.unsqueeze(-1)             # [T_eff, N, 1]
    return X, y


def run_etl(
    input_csv: Path,
    output_pt: Path,
    horizon: int = 12,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> None:
    print(f"[ETL METR-LA] Читаємо CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    if df.empty:
        raise ValueError(f"Файл {input_csv} порожній")

    # 1) Визначаємо часову колонку та приводимо до datetime
    time_col = detect_time_column(df)
    print(f"[ETL METR-LA] Виявлено часову колонку: {time_col}")

    df[time_col] = pd.to_datetime(df[time_col])

    # 2) Сортуємо за часом, прибираємо дублі
    df = df.sort_values(by=time_col).drop_duplicates(subset=[time_col])

    # 3) Відокремлюємо швидкості сенсорів
    sensor_cols = [c for c in df.columns if c != time_col]
    print(f"[ETL METR-LA] Кількість сенсорів: {len(sensor_cols)}")

    speeds_df = df[sensor_cols].astype(float)

    # 4) Очищення екстремальних значень
    speeds_df = clean_speeds(speeds_df, min_speed=0.0, max_speed=120.0)

    # 5) Заповнення пропусків
    speeds_df = fill_missing(speeds_df)

    # 6) Перетворення в тензор [T, N]
    speeds = torch.from_numpy(speeds_df.to_numpy()).float()  # [T, N]

    # 7) Формуємо X та y з горизонтом прогнозу
    X_all, y_all = build_x_y(speeds, horizon=horizon)  # X: [T_eff, N, 1], y: [T_eff, N]
    T_eff, N, _ = X_all.shape
    print(f"[ETL METR-LA] Після врахування горизонту H={horizon}: T_eff={T_eff}, N={N}")

    # 8) train/val/test спліт по часовій осі
    X_train, X_val, X_test = train_val_test_split(X_all, train_ratio, val_ratio)
    # для обчислення статистик важливий тільки train
    mean, std = compute_normalization_stats(X_train)

    # 9) Нормалізуємо всі спліти одними й тими ж mean/std
    X_all_norm = normalize_data(X_all, mean, std)

    # 10) Побудова простого edge_index (наступним етапом можна замінити на OSM-граф)
    edge_index = build_edge_index(N)

    # 11) Формуємо payload для збереження
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sensor_ids": sensor_cols,
        "timestamps": df[time_col].iloc[:T_eff].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "X": X_all_norm,      # [T_eff, N, 1] нормалізовані
        "y": y_all,           # [T_eff, N] сирі значення швидкості (до нормалізації)
        "edge_index": edge_index,
        "horizon": horizon,
        "mean": mean,
        "std": std,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }

    print(f"[ETL METR-LA] Зберігаємо датасет до {output_pt}")
    torch.save(payload, output_pt)
    print("[ETL METR-LA] Готово.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETL-конвеєр для METR-LA CSV (Zenodo формат).")
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Шлях до METR-LA.csv (скачаного з Zenodo).",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        required=True,
        help="Шлях до .pt файлу з обробленим датасетом.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Горизонт прогнозу (кількість кроків вперед).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Частка даних для train (по часу).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Частка даних для val (по часу).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_etl(
        input_csv=Path(args.input_csv),
        output_pt=Path(args.output_pt),
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )