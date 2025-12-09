import json
from pathlib import Path

import requests
import torch

BASE_URL = "http://localhost:8000"


def main() -> None:
    data_path = Path("data/processed/metr_la.pt")
    data = torch.load(data_path)

    # Подивимось форми для надійності
    X = data["X"]               # очікується: [T, N, F] або [num_samples, N, F]
    edge_index = data["edge_index"]  # [2, E]

    print("X shape:", X.shape)
    print("edge_index shape:", edge_index.shape)

    # Беремо перший часовий зріз / перший семпл
    # Якщо X має форму [T, N, F] -> X[0] дає [N, F]
    # Якщо [num_samples, N, F] -> логіка та ж
    x_t = X[0]                  # [N, F]
    node_features = x_t.tolist()
    edge_index_list = edge_index.tolist()

    payload = {
        "node_features": node_features,
        "edge_index": edge_index_list,
    }

    print("Sending payload snippet:")
    print("node_features[0:3]:", node_features[:3])

    resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)

    print("Status:", resp.status_code)
    try:
        print("Body (parsed):", json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except Exception:
        print("Raw body:", resp.text)


if __name__ == "__main__":
    main()