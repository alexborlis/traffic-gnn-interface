import re
import subprocess
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_HTML = OUTPUT_DIR / "metr_la_hybrid_metrics.html"


def run_evaluate_script() -> str:
    """
    Запускає `python -m training.evaluate_metr_la_hybrid`
    і повертає повний текстовий вивід.
    """
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "training.evaluate_metr_la_hybrid",
    ]

    print(f"=== Вивід training.evaluate_metr_la_hybrid ===")
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    # Одразу виводимо stdout / stderr, щоб було видно в консолі
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"evaluate_metr_la_hybrid завершився з кодом {completed.returncode}"
        )

    return completed.stdout


def parse_metrics(output: str) -> dict:
    """
    Дістає MSE, MAE, RMSE, MAPE з текстового виводу evaluate-скрипта.
    """
    pattern = (
        r"MSE \(loss\):\s*([0-9.+-eE]+)\s*[\r\n]+"
        r"MAE:\s*([0-9.+-eE]+)\s*[\r\n]+"
        r"RMSE:\s*([0-9.+-eE]+)\s*[\r\n]+"
        r"MAPE:\s*([0-9.+-eE]+)"
    )

    match = re.search(pattern, output)
    if not match:
        raise ValueError("Не вдалося розпарсити метрики з виводу evaluate-скрипта.")

    mse = float(match.group(1))
    mae = float(match.group(2))
    rmse = float(match.group(3))
    mape = float(match.group(4))

    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}


def build_plot(metrics: dict) -> None:
    """
    Будує більш зрозумілий графік:
    - окремий subplot для MSE / MAE / RMSE
    - окремий subplot для MAPE (у мільйонах %)
    """
    mse = metrics["mse"]
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    mape = metrics["mape"]

    # Для MAPE робимо масштабування, щоб не було страшних чисел
    # Наприклад, 296_000_000 % -> 296.00 (тобто ×10^6 %)
    mape_mln = mape / 1e6

    print(f"[PLOT METR-LA HYBRID] Розпізнані метрики: {metrics}")
    print(f"[PLOT METR-LA HYBRID] MAPE у млн %: {mape_mln:.3f}")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "METR-LA HYBRID — MSE / MAE / RMSE на тестовому наборі",
            "METR-LA HYBRID — MAPE на тестовому наборі (в млн %)",
        ),
        horizontal_spacing=0.18,
    )

    # ---- Лівий subplot: MSE / MAE / RMSE ----
    left_metrics_names = ["MSE", "MAE", "RMSE"]
    left_metrics_values = [mse, mae, rmse]

    fig.add_trace(
        go.Bar(
            x=left_metrics_names,
            y=left_metrics_values,
            text=[f"{v:.3f}" for v in left_metrics_values],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Значення",
        row=1,
        col=1,
    )

    # ---- Правий subplot: тільки MAPE (масштабований) ----
    fig.add_trace(
        go.Bar(
            x=["MAPE"],
            y=[mape_mln],
            text=[f"{mape_mln:.3f} млн %"],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Значення (×10⁶ %)",
        row=1,
        col=2,
    )

    fig.update_layout(
        title="METR-LA HYBRID — підсумкові метрики на тестовому наборі",
        bargap=0.4,
    )

    fig.write_html(str(OUTPUT_HTML), include_plotlyjs="cdn")
    print(f"[PLOT METR-LA HYBRID] Графік збережено до {OUTPUT_HTML}")


def main() -> None:
    output = run_evaluate_script()
    metrics = parse_metrics(output)
    build_plot(metrics)


if __name__ == "__main__":
    main()