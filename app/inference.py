import time
import torch
import numpy as np
from app.schemas import PredictionRequest

class InferenceEngine:
    """
    Обгортка для інференсу GNN-моделі.
    Містить методи: warmup, predict
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = next(model.parameters()).device
        self.model_version = "unknown"  # Опційно можна підтягнути версію з S3 або ENV

    def warmup(self) -> None:
        """
        Попередній запуск з фіктивними даними для прискорення Cold Start
        """
        dummy_input = PredictionRequest(
            edge_index=[[0, 1], [1, 2]],
            node_features=[[0.0] * 16] * 3
        )
        self.predict(dummy_input)

    def predict(self, request: PredictionRequest) -> list[float]:
        """
        Основний метод для інференсу. Приймає PredictionRequest, повертає список передбачень.
        """
        edge_index = torch.tensor(request.edge_index, dtype=torch.long, device=self.device)
        node_features = torch.tensor(request.node_features, dtype=torch.float, device=self.device)

        if edge_index.shape[0] != 2:
            edge_index = edge_index.T.contiguous()

        self.model.eval()
        with torch.no_grad():
            output = self.model(node_features, edge_index)

        return output.detach().cpu().tolist()