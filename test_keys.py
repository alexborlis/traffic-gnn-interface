import torch
from pathlib import Path

data_path = Path("data/processed/metr_la.pt")
data = torch.load(data_path)

print(type(data))
try:
    print("keys:", data.keys())
except AttributeError:
    print("no .keys(), object is:", data)