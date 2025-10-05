from pydantic import BaseModel, Field
from typing import List

class HealthStatusResponse(BaseModel):
    status: str = "ok"

class ReadinessStatusResponse(BaseModel):
    status: str = "ready"

class PredictionRequest(BaseModel):
    edge_index: List[List[int]] = Field(..., example=[[0, 1], [1, 2]])
    node_features: List[List[float]] = Field(..., example=[[0.1, 0.2], [0.3, 0.4]])

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    latency_ms: float
