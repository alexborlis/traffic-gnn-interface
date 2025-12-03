"""
analysis/plot_metr_la_results_plotly.py

Скрипт будує основні графіки для експериментів на METR-LA
з використанням Plotly замість matplotlib.

Графіки зберігаються у вигляді HTML-файлів у теці analysis/figures.
Їх можна відкрити у браузері та:
  - або зробити скріншот для вставки в диплом,
  - або зберегти як PDF/зображення через можливості браузера.

Використовуються реальні числові результати з логів навчання:
- Базова GNN-модель (TrafficGraphNeuralNetwork)
- Гібридна GNN+Transformer (HybridTrafficGraphNeuralNetwork)
"""

from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px


def ensure_output_dir() -> Path:
    """
    Гарантує існування теки для збереження графіків.
    Повертає Path до теки.
    """
    output_dir = Path("analysis") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_test_metrics_comparison():
    """
    Побудова стовпчикової діаграми з порівнянням фінальних метрик
    на TEST-спліті для базової та гібридної моделі.

    Використані значення – це ті, які ти отримав із:
      - python -m training.evaluate_metr_la
      - python -m training.evaluate_metr_la_hybrid
    (скопійовані сюди як константи для репродуктивності).
    """

    # Назви моделей (базова й гібридна)
    models = ["GNN (base)", "GNN+Transformer (hybrid)"]

    # Метрики з логів evaluate:
    # === METR-LA: підсумкові метрики на TEST-спліті ===
    # MSE (loss): 329.3090
    # MAE:        12.4110
    # RMSE:       16.0130
    # MAPE:       355937951.18 %
    #
    # === METR-LA HYBRID: підсумкові метрики на TEST-спліті ===
    # MSE (loss): 282.2325
    # MAE:        11.2217
    # RMSE:       14.1633
    # MAPE:       279924152.97 %

    mae_values = [12.4110, 11.2217]
    rmse_values = [16.0130, 14.1633]

    fig = go.Figure()

    # Стовпчики MAE
    fig.add_trace(
        go.Bar(
            x=models,
            y=mae_values,
            name="MAE",
            text=[f"{v:.2f}" for v in mae_values],
            textposition="auto",
        )
    )

    # Стовпчики RMSE
    fig.add_trace(
        go.Bar(
            x=models,
            y=rmse_values,
            name="RMSE",
            text=[f"{v:.2f}" for v in rmse_values],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=(
            "METR-LA: порівняння якості моделей на TEST-спліті<br>"
            "(базова GNN vs гібридна GNN+Transformer)"
        ),
        xaxis_title="Модель",
        yaxis_title="Значення метрики",
        barmode="group",
        legend_title="Метрика",
        template="plotly_white",
    )

    return fig


def build_val_loss_by_epoch():
    """
    Побудова лінійного графіка з динамікою val-loss по епохах
    для базової та гібридної моделі.

    Дані взято з логів:
      - training.train_metr_la
      - training.train_metr_la_hybrid
    """

    # Номери епох (обидві моделі навчалися 5 епох)
    epochs = [1, 2, 3, 4, 5]

    # Val loss з логів базової моделі:
    # Epoch 001: Val loss=274.7725
    # Epoch 002: Val loss=273.1533
    # Epoch 003: Val loss=275.7384
    # Epoch 004: Val loss=273.5546
    # Epoch 005: Val loss=273.2219
    val_loss_base = [274.7725, 273.1533, 275.7384, 273.5546, 273.2219]

    # Val loss з логів гібридної моделі:
    # Epoch 001: Val loss=237.1299
    # Epoch 002: Val loss=243.3542
    # Epoch 003: Val loss=236.7402
    # Epoch 004: Val loss=237.7945
    # Epoch 005: Val loss=237.6656
    val_loss_hybrid = [237.1299, 243.3542, 236.7402, 237.7945, 237.6656]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_loss_base,
            mode="lines+markers",
            name="Val loss – GNN (base)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_loss_hybrid,
            mode="lines+markers",
            name="Val loss – GNN+Transformer (hybrid)",
        )
    )

    fig.update_layout(
        title="METR-LA: динаміка валідаційної помилки (MSE) по епохах",
        xaxis_title="Епоха",
        yaxis_title="Val loss (MSE)",
        template="plotly_white",
    )

    fig.update_xaxes(dtick=1)

    return fig


def build_relative_improvement_bar():
    """
    Додатковий графік: відносне покращення гібридної моделі
    відносно базової у відсотках по MAE та RMSE.
    """

    # Базові значення
    mae_base = 12.4110
    rmse_base = 16.0130

    # Гібридні значення
    mae_hybrid = 11.2217
    rmse_hybrid = 14.1633

    # Обчислюємо відносне покращення (відсоткове зниження)
    mae_improvement = (mae_base - mae_hybrid) / mae_base * 100.0
    rmse_improvement = (rmse_base - rmse_hybrid) / rmse_base * 100.0

    metrics = ["MAE", "RMSE"]
    improvements = [mae_improvement, rmse_improvement]

    fig = px.bar(
        x=metrics,
        y=improvements,
        text=[f"{v:.2f}%" for v in improvements],
        labels={"x": "Метрика", "y": "Покращення, %"},
        title="METR-LA: відносне покращення гібридної моделі<br>"
              "відносно базової (зниження MAE та RMSE)",
    )

    fig.update_traces(textposition="auto")
    fig.update_layout(template="plotly_white")

    return fig


def main():
    """
    Точка входу: генерує всі графіки й зберігає їх у HTML.
    """
    output_dir = ensure_output_dir()

    # 1) Порівняння MAE/RMSE на TEST-спліті
    fig_test = build_test_metrics_comparison()
    fig_test_path = output_dir / "metr_la_test_mae_rmse.html"
    fig_test.write_html(fig_test_path)
    print(f"[OK] Збережено графік порівняння MAE/RMSE: {fig_test_path}")

    # 2) Динаміка val-loss по епохах
    fig_val = build_val_loss_by_epoch()
    fig_val_path = output_dir / "metr_la_val_loss_by_epoch.html"
    fig_val.write_html(fig_val_path)
    print(f"[OK] Збережено графік val-loss по епохах: {fig_val_path}")

    # 3) Відносне покращення в %
    fig_imp = build_relative_improvement_bar()
    fig_imp_path = output_dir / "metr_la_relative_improvement.html"
    fig_imp.write_html(fig_imp_path)
    print(f"[OK] Збережено графік відносного покращення: {fig_imp_path}")

    print("\nУсі графіки збережено як HTML. "
          "Відкрий їх у браузері й зроби скріншоти/експорт у зображення для диплома.")


if __name__ == "__main__":
    main()