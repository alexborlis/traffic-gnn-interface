import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class TrafficGraphNeuralNetwork(nn.Module):
    """
    Модель графової нейронної мережі на основі GCNConv для прогнозування транспортного потоку.
    Вхід: ознаки вузлів та індекси ребер (edge_index)
    Вихід: регресійне передбачення по кожному вузлу.
    """

    def __init__(self, input_features: int = 16, hidden_units: int = 32, output_features: int = 1):
        super().__init__()
        self.first_layer = GCNConv(input_features, hidden_units)
        self.activation_function = nn.ReLU()
        self.second_layer = GCNConv(hidden_units, output_features)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Подає вхідні дані через два GCN-шари
        """
        hidden = self.first_layer(node_features, edge_index)
        hidden = self.activation_function(hidden)
        output = self.second_layer(hidden, edge_index)
        return output.squeeze(-1)
