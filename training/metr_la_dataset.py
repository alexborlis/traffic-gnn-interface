# training/metr_la_dataset.py

"""
Dataset-клас для роботи з попередньо обробленим датасетом METR-LA.

Джерело даних — файл data/processed/metr_la.pt, який формується
скриптом training/etl_metr_la.py.

Формат збережених даних (payload):
    {
        "sensor_ids": List[str],
        "timestamps": List[str],
        "X": Tensor [T_eff, N, 1],   # нормалізовані ознаки (швидкість)
        "y": Tensor [T_eff, N],      # таргет (швидкість через H кроків)
        "edge_index": Tensor [2, E],
        "horizon": int,
        "mean": Tensor [N],
        "std": Tensor [N],
        "train_ratio": float,
        "val_ratio": float,
    }

Цей Dataset:
- робить часовий train/val/test split по осі T;
- на кожному кроці t повертає (x_t, edge_index, y_t).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


SplitType = Literal["train", "val", "test"]


@dataclass
class MetrLaDatasetConfig:
    """Конфігурація для METR-LA Dataset."""

    data_path: Path
    split: SplitType


class MetrLaDataset(Dataset):
    """
    Dataset для METR-LA з підтримкою split'ів train/val/test.

    Кожен елемент — це один часовий крок t:
        x_t: Tensor [N, 1]     - ознаки (нормалізована швидкість)
        edge_index: Tensor [2, E]  - структура графа (спільна для всіх t)
        y_t: Tensor [N]        - таргет (швидкість через horizon кроків)
    """

    def __init__(self, config: MetrLaDatasetConfig) -> None:
        super().__init__()
        self.config = config

        if not self.config.data_path.is_file():
            raise FileNotFoundError(
                f"Файл з даними для METR-LA не знайдено: {self.config.data_path}"
            )

        payload = torch.load(self.config.data_path, map_location="cpu")

        # Зберігаємо тензори
        self.X: Tensor = payload["X"]          # [T_eff, N, 1], нормалізовані
        self.y: Tensor = payload["y"]          # [T_eff, N], сирі значення
        self.edge_index: Tensor = payload["edge_index"]  # [2, E]

        self.sensor_ids = payload["sensor_ids"]
        self.timestamps = payload["timestamps"]
        self.horizon: int = int(payload["horizon"])
        self.train_ratio: float = float(payload["train_ratio"])
        self.val_ratio: float = float(payload["val_ratio"])

        # Перевірки розмірностей
        if self.X.dim() != 3:
            raise ValueError(f"Очікується X розмірності [T, N, 1], отримано {self.X.shape}")
        if self.y.dim() != 2:
            raise ValueError(f"Очікується y розмірності [T, N], отримано {self.y.shape}")

        T_eff, N, F = self.X.shape
        if F != 1:
            raise ValueError(f"Очікується одна ознака (F=1), отримано F={F}")
        if self.y.shape != (T_eff, N):
            raise ValueError(
                f"Невідповідність розмірностей X і y: X={self.X.shape}, y={self.y.shape}"
            )

        # Обчислюємо межі сплітів по осі часу
        self.T_eff = T_eff
        self.N = N

        self.train_end = int(self.T_eff * self.train_ratio)
        self.val_end = int(self.T_eff * (self.train_ratio + self.val_ratio))

        if self.train_end <= 0 or self.val_end <= self.train_end or self.val_end >= self.T_eff:
            raise ValueError(
                f"Н некоректний split: T_eff={self.T_eff}, "
                f"train_ratio={self.train_ratio}, val_ratio={self.val_ratio}, "
                f"-> train_end={self.train_end}, val_end={self.val_end}"
            )

        # Визначаємо інтервал індексів для обраного split
        if self.config.split == "train":
            self.start_idx = 0
            self.end_idx = self.train_end
        elif self.config.split == "val":
            self.start_idx = self.train_end
            self.end_idx = self.val_end
        else:  # "test"
            self.start_idx = self.val_end
            self.end_idx = self.T_eff

        if self.start_idx >= self.end_idx:
            raise ValueError(
                f"Порожній split '{self.config.split}': [{self.start_idx}, {self.end_idx})"
            )

    def __len__(self) -> int:
        # Кількість часових кроків у відповідному split
        return self.end_idx - self.start_idx

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Повертає один часовий крок t для обраного split.

        :param index: локальний індекс усередині split'а (0..len-1)
        :return:
            x_t: Tensor [N, 1]        - ознаки
            edge_index: Tensor [2, E] - структура графа
            y_t: Tensor [N]           - таргет
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} поза межами [0, {len(self) - 1}]")

        global_idx = self.start_idx + index

        x_t: Tensor = self.X[global_idx]      # [N, 1]
        y_t: Tensor = self.y[global_idx]      # [N]

        # edge_index один і той самий для всіх t
        edge_index: Tensor = self.edge_index  # [2, E]

        return x_t, edge_index, y_t