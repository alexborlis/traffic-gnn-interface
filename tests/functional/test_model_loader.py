import pytest
import os
from unittest import mock
from app.model_loader import load_model_instance

@mock.patch.dict(os.environ, {
    "MODEL_ARTIFACT_URI": "./tests/unit/test_model.pt",
    "MODEL_CLASS_PATH": "models.traffic_gnn.TrafficGraphNeuralNetwork",
    "INFERENCE_DEVICE": "cpu"
})
def test_load_model_instance_local(tmp_path):
    # Збереження фейкової моделі на диск
    from models.traffic_gnn import TrafficGraphNeuralNetwork
    import torch

    model = TrafficGraphNeuralNetwork()
    test_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), test_path)

    os.environ["MODEL_ARTIFACT_URI"] = str(test_path)

    loaded_model = load_model_instance()
    assert isinstance(loaded_model, TrafficGraphNeuralNetwork)