# training/toy_dataset.py

"""
Toy-датаcет для тренування графової моделі на простій задачі.

Мета:
- мати мінімальний, але робочий набір даних для відлагодження train-loop;
- уникнути складного ETL до того, як ми перевіримо, що тренування взагалі працює.

Модель бачить:
- один і той самий граф (3 вузли, прості ребра);
- для кожного "зразка" трохи різні node_features;
- ціль (target) - проста функція від node_features, наприклад y = 2 * x.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ToyTrafficDataset(Dataset):
    """
    Невеликий синтетичний датасет для тренування GNN.

    - Граф: 3 вузли, ребра (0-1), (1-2), (2-0) - простий цикл.
    - Ознаки: випадкові числа з N(0,1) або з обмеженого діапазону.
    - Ціль: y = 2 * x для кожного вузла (простий регресійний таргет).
    """

    def __init__(self, num_samples: int = 512, seed: int = 42) -> None:
        super().__init__()
        self.num_samples = num_samples

        # Фіксуємо seed, щоб результати були відтворюваними
        g = torch.Generator()
        g.manual_seed(seed)

        # Створюємо всі node_features наперед: [num_samples, num_nodes, num_features]
        # У нас буде 3 вузли та 1 ознака на вузол
        self.node_features: Tensor = torch.randn(
            num_samples,
            3,   # num_nodes
            1,   # num_features
            generator=g,
            dtype=torch.float32,
        )

        # Ціль: y = 2 * x для кожного вузла (та сама розмірність, тільки без останньої осі)
        self.targets: Tensor = 2.0 * self.node_features.squeeze(-1)  # [num_samples, 3]

        # Edge_index - один і той самий для всіх прикладів:
        # 0 -> 1, 1 -> 2, 2 -> 0 (цикл)
        self.edge_index: Tensor = torch.tensor(
            [
                [0, 1, 2],  # from
                [1, 2, 0],  # to
            ],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param idx: індекс зразка
        :return:
            - node_features: [num_nodes, num_features]
            - edge_index: [2, num_edges]
            - targets: [num_nodes]
        """
        x = self.node_features[idx]          # [3, 1]
        y = self.targets[idx]                # [3]
        return x, self.edge_index, y