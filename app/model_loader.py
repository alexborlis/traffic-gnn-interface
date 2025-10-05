import os
import torch
import importlib
from pathlib import Path
from app.utils_s3 import download_model_from_s3

def load_model_instance():
    """Завантаження та ініціалізація моделі (з S3 або локально)"""
    artifact_uri = os.environ["MODEL_ARTIFACT_URI"]
    class_path = os.environ["MODEL_CLASS_PATH"]
    device = os.getenv("INFERENCE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    if artifact_uri.startswith("s3://"):
        local_path = "/opt/model/model.pt"
        download_model_from_s3(artifact_uri, local_path)
    else:
        local_path = artifact_uri

    model = model_class()
    model.load_state_dict(torch.load(local_path, map_location=device))
    model.to(device)
    model.eval()
    return model
