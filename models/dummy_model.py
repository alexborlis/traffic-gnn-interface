import torch
from torch import nn
from torch import Tensor


class DummyTrafficModel(nn.Module):
    """
    Проста "заглушка" моделі для локального запуску API без навчання.

    Ідея:
    - ми очікуємо на вхід матрицю ознак вузлів графа: [num_nodes, num_features];
    - edge_index (список ребер) наразі ігноруємо;
    - на виході повертаємо для КОЖНОГО вузла одне число (наприклад, "прогноз швидкості").
    """

    def __init__(self, input_dim: int = 1):
        super().__init__()
        # Проста навчувана константа, яку модель додає до середнього значення
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, node_features: Tensor, edge_index: Tensor | None = None) -> Tensor:
        """
        :param node_features: тензор розмірності [num_nodes, num_features]
        :param edge_index: тензор з ребрами графа [2, num_edges] або None (ігнорується)
        :return: тензор розмірності [num_nodes], де кожне значення - "прогноз" для вузла
        """
        # Якщо вхід порожній – повертаємо порожній тензор
        if node_features.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=node_features.device)

        # Середнє значення по всіх вузлах та всіх їхніх ознаках
        mean_tensor = node_features.mean()  # тензор-скаляр

        # Додаємо bias і перетворюємо результат у звичайний float
        # detach() + item() гарантує, що тут немає градієнтів і це чистий Python-скаляр
        base_value = (mean_tensor + self.bias.squeeze(0)).item()

        num_nodes = node_features.size(0)

        # Створюємо вихід: для кожного вузла одне й те саме значення base_value
        output = torch.full(
            size=(num_nodes,),
            fill_value=base_value,
            dtype=torch.float32,
            device=node_features.device,
        )
        return output