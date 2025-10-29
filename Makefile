PYTHON := python3.11
IMAGE_NAME := traffic-gnn
APP_PORT := 8080

# Create virtual environment and install CPU dependencies
install:
	$(PYTHON) -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Create virtual environment and install GPU dependencies
install-gpu:
	$(PYTHON) -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements-gpu.txt

# Format code
format:
	black app models tests

# Run linters
lint:
	flake8 app models tests

# Run tests
test:
	pytest tests

# Run local FastAPI server
run:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(APP_PORT)

# Docker build for CPU
build-cpu:
	docker build -f Dockerfile.cpu \
		--build-arg PYTHON_VERSION=3.11 \
		--build-arg PYTORCH_VERSION=2.4.0 \
		--build-arg TORCHVISION_VERSION=0.19.0 \
		--build-arg CUDA_SUFFIX=cpu \
		--build-arg DOWNLOAD_MODEL_ON_BUILD=false \
		-t $(IMAGE_NAME):cpu .

# Docker build for GPU
build-gpu:
	docker build -f Dockerfile.gpu \
		--build-arg PYTHON_VERSION=3.11 \
		--build-arg PYTORCH_VERSION=2.4.0 \
		--build-arg TORCHVISION_VERSION=0.19.0 \
		--build-arg CUDA_SUFFIX=cu121 \
		--build-arg DOWNLOAD_MODEL_ON_BUILD=false \
		-t $(IMAGE_NAME):gpu .

# Run container locally (CPU)
run-cpu:
	docker run --rm -it -p $(APP_PORT):8080 --name $(IMAGE_NAME)-cpu $(IMAGE_NAME):cpu

# Run container locally (GPU)
run-gpu:
	docker run --rm -it --gpus all -p $(APP_PORT):8080 --name $(IMAGE_NAME)-gpu $(IMAGE_NAME):gpu

# Clean project
clean:
	rm -rf .venv __pycache__ .pytest_cache .mypy_cache *.log artifacts/*.pt