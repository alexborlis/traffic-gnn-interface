# Repository Audit — traffic-gnn-interface (v0.1)

Дата: 2025-10-29  
Гілка: M2-31_audit_structure

## 1) Baseline (Before)
Видимі ключові артефакти (з кореня):
- ✅ `app/`
- ✅ `tests/`
- ✅ `models/`
- ✅ `Dockerfile.cpu`, `Dockerfile.gpu`
- ✅ `requirements-gpu.txt`
- ⚠️ `requirements.txt` — **порожній**
- ✅ `Makefile`
- ✅ `entrypoint.sh`
- ✅ `.dockerignore`, `.gitignore`, `.python-version`, `.venv/`
- ⚠️ `.idea/` — IDE-конфіги (перевірити, що ігноряться)
- ✅ `structure.txt`
- ❌ `cli.py` (не знайдено в корені)
- ❌ `docker-compose.yml` (не знайдено)

## 2) Naming & Duplicates (попередній огляд)
- Модулі Python: перевірити snake_case у `app/**.py` (окремий крок нижче).
- Дублювання функцій/класів: потрібен grep-аналіз (див. крок нижче).

## 3) Dependencies & Tests (поточний стан)
- `requirements.txt` — порожній → інсталяція **CPU**-середовища не відтворюється.
- `requirements-gpu.txt` — заповнений → GPU-варіант присутній.
- Тести: є `tests/` (статус виконання буде зафіксовано після прогона).

## 4) Config & Secrets
- Потрібно додати `.env.example` (мінімальний перелік змінних).
- Перевірити `.gitignore`, що ігнорує `.env`, `models/*` артефакти, кеші, ваги (`*.pt`, `*.pth`, `*.onnx`, `*.ckpt`), `.idea/`, `.venv/`, `__pycache__/`.

## 5) Entrypoints
- CLI: відсутній явний `cli.py` → додати в беклог каркас CLI.
- API/Serving: є `entrypoint.sh` і `app/` (можливо FastAPI/uvicorn) — треба підтвердити командою запуску та імпортами.

## 6) Backlog (1h-сабтаски)
- [ ] S2-1: Заповнити `requirements.txt` (CPU-мінімум) на базі реальних імпортів з `app/`.
- [ ] S2-2: Додати `.env.example`; оновити `.gitignore` (перевірити `.idea/`, моделі, ваги, кеші).
- [ ] S2-3: Перевірити і виправити snake_case імен модулів у `app/` (якщо є відхилення).
- [ ] S2-4: Додати `cli.py` з командами `etl|train|serve` (мінімальні заглушки).
- [ ] S2-5: Smoke-тест `pytest -q` і зафіксувати падіння з короткою діагностикою.
- [ ] S2-6: Перевірити варіанти запуску API: `uvicorn app.main:app` або `python -m app` (залежно від структури) + інструкцію в README.
- [ ] S2-7: Додати `docker-compose.yml` (опційно) з сервісом API і томами для `models/`.
- [ ] S2-8: Оновити README (Setup/Run/Docker) з живими командами.

## 7) DoD check (для цієї сабтаски)
- [ ] `structure.txt` у репозиторії (стан "до").
- [ ] README містить Project Structure + базові інструкції запуску.
- [ ] `.env.example` створено; `.gitignore` охоплює секрети/артефакти.
- [ ] Звіт `AUDIT_REPORT.md (v0.1)` заповнений фактичними даними.
