# Traffic GNN Interface  
### Інтелектуальна система прогнозування транспортного потоку на основі графових нейронних мереж

Цей проєкт реалізує повноцінну систему для прогнозування транспортного потоку, створену на основі графових нейронних мереж (GNN та Hybrid GNN+Transformer). Архітектура включає ETL-процес, побудову графа дорожньої мережі, формування ознак, навчання моделей, оцінку результатів та продакшн-сервіс інференсу FastAPI.

Система побудована модульно, масштабовано та з фокусом на практичне застосування в інтелектуальних транспортних системах (ITS).

---

# 📦 Структура проєкту

```
traffic-gnn-interface/
│
├── app/                     # FastAPI сервіс інференсу
│   ├── main.py
│   ├── inference.py
│   └── model_loader.py
│
├── models/                  # GNN та Hybrid GNN+Transformer моделі
│   └── traffic_gnn.py
│
├── training/                # ETL, Dataset, навчання, метрики
│   ├── etl_metr_la.py
│   ├── metr_la_dataset.py
│   ├── train_metr_la.py
│   ├── train_metr_la_hybrid.py
│   ├── build_metr_la_hybrid_model.py
│   └── metrics.py
│
├── analysis/                # Візуалізація результатів
│   ├── plot_metr_la_results_plotly.py
│   └── figures/
│
├── pipeline/                # Повний pipeline
│   └── run_metr_la_hybrid_pipeline.py
│
├── Dockerfile.cpu
├── Dockerfile.gpu
├── requirements.txt
└── README.md
```

---

# 🚀 Алгоритм роботи системи

## **1. Етап ETL — training/etl_metr_la.py**

Процес включає:

- завантаження сирих часових рядів METR-LA;
- очистку, фільтрацію, ресемплінг;
- нормалізацію;
- формування історичних вікон (temporal windowing);
- побудову графа (`edge_index`):
  ```python
  edge_index = build_edge_index(N)
  ```
- серіалізацію у формат тензорів або parquet.

Запуск:

```
python training/etl_metr_la.py
```

---

## **2. Формування Dataset — metr_la_dataset.py**

Dataset відповідає за:

- завантаження згенерованих ETL-файлів;
- формування train/val/test вибірок;
- повернення тензорів:

```
x: [N, input_dim]
edge_index: [2, E]
target: [N, 1]
```

---

## **3. Моделі прогнозування — models/traffic_gnn.py**

### **TrafficGraphNeuralNetwork**
Базова GNN-модель:

- GraphConvLayer → ReLU → Dropout;
- adjacency normalization;
- прогноз вузлових значень.

### **HybridTrafficGraphNeuralNetwork**
Гібридна модель:

1. GNN — витягує просторові ознаки;
2. Transformer Encoder — моделює часову структуру;
3. Лінійний шар → прогноз.

Виклик:

```python
y = model(x, edge_index)
```

---

## **4. Навчання моделі**

### Файли:

- `training/train_metr_la.py`  
- `training/train_metr_la_hybrid.py`

Алгоритм:

- batch learning;
- оптимізатор AdamW;
- рання зупинка;
- MAE/RMSE/MAPE;
- збереження моделі (`state_dict`).

Запуск:

```
python training/train_metr_la_hybrid.py
```

---

## **5. Аналіз результатів**

### analysis/plot_metr_la_results_plotly.py

Генерує:

- графіки MAE/RMSE/MAPE;
- heatmaps просторової помилки;
- порівняння моделей GNN vs Hybrid.

Усі результати — у каталозі:

```
analysis/figures/
```

---

## **6. Повний pipeline**

### pipeline/run_metr_la_hybrid_pipeline.py

Виконує:

1. ETL  
2. Dataset  
3. Навчання  
4. Оцінку  
5. Збереження моделі  

Запуск:

```
python pipeline/run_metr_la_hybrid_pipeline.py
```

---

# ⚡ Сервіс інференсу FastAPI

### app/main.py

Endpoint:

```
POST /predict
```

Приклад JSON:

```json
{
  "edge_index": [[0,1],[1,2]],
  "node_features": [[0.5],[0.6],[0.8]]
}
```

### app/model_loader.py

Завантаження моделі:

```python
model = model_class()
model.load_state_dict(torch.load(path))
model.eval()
```

### Запуск:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

# 🐳 Docker

### CPU:

```
docker build -f Dockerfile.cpu -t traffic-gnn:cpu .
docker run -p 8000:8000 traffic-gnn:cpu
```

### GPU:

```
docker build -f Dockerfile.gpu -t traffic-gnn:gpu .
docker run --gpus all -p 8000:8000 traffic-gnn:gpu
```

---

# 📈 Технології

- Python 3.12  
- PyTorch 2.x  
- FastAPI  
- Plotly  
- Docker (CPU/GPU)  
- AWS S3 (артефакти моделей)

---

# 🎯 Можливості системи

- Повний ML-конвеєр  
- Гібридна модель GNN + Transformer  
- Високопродуктивний інференс  
- Візуалізація результатів  
- Масштабованість на інші міста  
- Підтримка GPU/CPU  

---

# 🧩 Подальший розвиток

- інтеграція з реальними OSM-графами;  
- використання GAT, ST-GCN, TGNN;  
- онлайн-інференс і потокова обробка;  
- хмарний тренувальний конвеєр.

---

# 📜 Ліцензія

MIT License
