.PHONY: build-gpu build-cpu run-gpu run-cpu test format lint

APP_NAME=traffic-gnn
PYTHON_VERSION=3.11
PYTORCH_VERSION=2.4.0
TORCHVISION_VERSION=0.19.0
CUDA_SUFFIX=cu121
APP_PORT=8080

build-gpu:
	docker build -f Dockerfile.gpu \
	  --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	  --build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
	  --build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
	  --build-arg CUDA_SUFFIX=$(CUDA_SUFFIX) \
	  --build-arg DOWNLOAD_MODEL_ON_BUILD=false \
	  -t $(APP_NAME):gpu .

build-cpu:
	docker build -f Dockerfile.cpu \
	  --build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
	  --build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
	  --build-arg TORCHVISION_VERSION=$(TORCHVISION_VERSION) \
	  --build-arg CUDA_SUFFIX=cpu \
	  --build-arg DOWNLOAD_MODEL_ON_BUILD=false \
	  -t $(APP_NAME):cpu .

run-gpu:
	docker run --rm -it --gpus all -p $(APP_PORT):$(APP_PORT) \
	  -e MODEL_ARTIFACT_URI=./artifacts/model.pt \
	  -e INFERENCE_DEVICE=cuda:0 \
	  --name $(APP_NAME)-gpu $(APP_NAME):gpu

run-cpu:
	docker run --rm -it -p $(APP_PORT):$(APP_PORT) \
	  -e MODEL_ARTIFACT_URI=./artifacts/model.pt \
	  -e INFERENCE_DEVICE=cpu \
	  --name $(APP_NAME)-cpu $(APP_NAME):cpu

test:
	pytest tests/

format:
	black .

lint:
	flake8 .