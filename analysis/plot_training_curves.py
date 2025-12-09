import json
from pathlib import Path

import matplotlib.pyplot as plt


# ---------- Налаштування шляхів ----------

ROOT = Path(__file__).resolve().parents[1] if __file__.endswith(".py") else Path(".")
MODELS_DIR = ROOT / "models"

BASELINE_LOG = MODELS_DIR / "training_log_metrla_baseline.json"
HYBRID_LOG = MODELS_DIR / "training_log_metrla_hybrid.json"

OUT_DIR = ROOT / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_history(path: Path) -> dict:
    """Завантажує JSON з історією навчання."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss(history_base: dict, history_hybrid: dict) -> None:
    """Будує графік train/val loss для базової та гібридної моделей."""
    epochs_base = history_base["epoch"]
    epochs_hybrid = history_hybrid["epoch"]

    plt.figure(figsize=(8, 5))

    # Базова модель
    plt.plot(epochs_base, history_base["train_loss"], label="Baseline – train loss")
    plt.plot(epochs_base, history_base["val_loss"], label="Baseline – val loss", linestyle="--")

    # Гібридна модель
    plt.plot(epochs_hybrid, history_hybrid["train_loss"], label="Hybrid – train loss")
    plt.plot(epochs_hybrid, history_hybrid["val_loss"], label="Hybrid – val loss", linestyle="--")

    plt.xlabel("Епоха")
    plt.ylabel("MSE (loss)")
    plt.title("Динаміка функції втрат під час навчання (METR-LA)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "metrla_train_val_loss_baseline_vs_hybrid.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Збережено графік loss: {out_path}")


def plot_mae(history_base: dict, history_hybrid: dict) -> None:
    """Будує графік MAE для train/val для обох моделей."""
    epochs_base = history_base["epoch"]
    epochs_hybrid = history_hybrid["epoch"]

    plt.figure(figsize=(8, 5))

    # Базова модель
    plt.plot(epochs_base, history_base["train_mae"], label="Baseline – train MAE")
    plt.plot(epochs_base, history_base["val_mae"], label="Baseline – val MAE", linestyle="--")

    # Гібридна модель
    plt.plot(epochs_hybrid, history_hybrid["train_mae"], label="Hybrid – train MAE")
    plt.plot(epochs_hybrid, history_hybrid["val_mae"], label="Hybrid – val MAE", linestyle="--")

    plt.xlabel("Епоха")
    plt.ylabel("MAE")
    plt.title("Динаміка MAE під час навчання (METR-LA)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "metrla_train_val_mae_baseline_vs_hybrid.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Збережено графік MAE: {out_path}")


def main() -> None:
    # 1. Завантажуємо історії навчання
    history_base = load_history(BASELINE_LOG)
    history_hybrid = load_history(HYBRID_LOG)

    # 2. Будуємо графік loss
    plot_loss(history_base, history_hybrid)

    # 3. Будуємо графік MAE
    plot_mae(history_base, history_hybrid)


if __name__ == "__main__":
    main()