# GNN-сервіс для інференсу трафіку з FastAPI та PyTorch

## Побудова образів

### GPU:
```bash
docker build -f Dockerfile.gpu \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg PYTORCH_VERSION=2.4.0 \
  --build-arg TORCHVISION_VERSION=0.19.0 \
  --build-arg CUDA_SUFFIX=cu121 \
  --build-arg DOWNLOAD_MODEL_ON_BUILD=false \
  -t traffic-gnn:gpu .
```

### CPU:
```bash
docker build -f Dockerfile.cpu \
  --build-arg PYTHON_VERSION=3.11 \
  --build-arg PYTORCH_VERSION=2.4.0 \
  --build-arg TORCHVISION_VERSION=0.19.0 \
  --build-arg CUDA_SUFFIX=cpu \
  --build-arg DOWNLOAD_MODEL_ON_BUILD=false \
  -t traffic-gnn:cpu .
```

## Запуск GPU
```bash
docker run --rm -it --gpus all -p 8080:8080 \
  -e MODEL_ARTIFACT_URI=s3://my-bucket/models/gnn/latest/model.pt \
  -e S3_ENDPOINT_URL=https://minio.example.com \
  -e S3_ACCESS_KEY=XXX \
  -e S3_SECRET_KEY=YYY \
  -e S3_REGION=eu-central-1 \
  -e INFERENCE_DEVICE=cuda:0 \
  --name traffic-gnn traffic-gnn:gpu
```

## Тестові запити:
```bash
curl -s localhost:8080/healthz
curl -s localhost:8080/readyz
curl -s -X POST localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"edge_index": [[0,1],[1,2]], "node_features": [[0.5,0.6],[0.3,0.1],[0.0,1.0]]}'
```

---

## Важливо
- Додаткові бібліотеки `torch-scatter`, `torch-sparse`, тощо — мають бути сумісні з PyTorch/CUDA. Підібрати `.whl` з https://data.pyg.org/whl/
- Не зберігайте секрети в Dockerfile. Усі ключі передаються через `ENV`.
