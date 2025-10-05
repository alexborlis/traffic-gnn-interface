import torch
from app.inference import InferenceEngine
from models.traffic_gnn import TrafficGraphNeuralNetwork
from app.schemas import PredictionRequest

def test_inference_predict_with_dummy_data():
    model = TrafficGraphNeuralNetwork()
    engine = InferenceEngine(model)
    engine.model.eval()

    dummy_request = PredictionRequest(
        edge_index=[[0, 1], [1, 2]],
        node_features=[[0.5] * 16, [0.2] * 16, [0.1] * 16]
    )

    predictions = engine.predict(dummy_request)
    assert isinstance(predictions, list)
    assert len(predictions) == 3
