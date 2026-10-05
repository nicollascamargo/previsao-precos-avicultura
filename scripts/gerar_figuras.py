"""Generate the figures embedded in the README.

    python scripts/gerar_figuras.py

Reads the artifacts produced by `make backtest` and writes PNGs into
``reports/figuras/``. Kept as a script rather than a notebook so that CI can
regenerate the charts without a kernel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from precos_avicultura.config import DATE_COL, PROCESSED_DIR, TARGET  # noqa: E402
from precos_avicultura.data.loaders import load_panel  # noqa: E402

FIG_DIR = Path("reports/figuras")
COR_REAL = "#1b2a41"
COR_PREV = "#c1440e"
COR_AUX = "#8d99ae"


def _estilo(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def figura_serie(panel: pd.DataFrame) -> None:
    """The target and its main cost driver, on twin axes."""
    df = panel.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df[DATE_COL], df[TARGET], color=COR_REAL, linewidth=1.8, label="Broiler (BRL/kg)")
    ax.set_ylabel("Broiler (BRL/kg)", color=COR_REAL)
    _estilo(ax)

    ax2 = ax.twinx()
    ax2.plot(
        df[DATE_COL], df["preco_milho"], color=COR_AUX, linewidth=1.4, label="Corn (BRL/bag)"
    )
    ax2.set_ylabel("Corn (BRL/60kg bag)", color=COR_AUX)
    ax2.spines[["top"]].set_visible(False)

    ax.set_title(
        "Broiler price and corn cost — pass-through happens with a lag",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    linhas = ax.get_lines() + ax2.get_lines()
    ax.legend(linhas, [ln.get_label() for ln in linhas], loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "serie_frango_milho.png", dpi=150)
    plt.close(fig)


def figura_backtest(predicoes: pd.DataFrame, modelo: str = "gbm_delta") -> None:
    """Predicted vs realised across the walk-forward window, per horizon."""
    horizontes = sorted(predicoes["horizonte"].unique())
    fig, axes = plt.subplots(len(horizontes), 1, figsize=(11, 3.1 * len(horizontes)), sharex=True)
    axes = [axes] if len(horizontes) == 1 else list(axes)

    for ax, h in zip(axes, horizontes, strict=False):
        sub = predicoes[
            (predicoes["horizonte"] == h) & (predicoes["modelo"] == modelo)
        ].sort_values("data_alvo")
        naive = predicoes[
            (predicoes["horizonte"] == h) & (predicoes["modelo"] == "naive")
        ].sort_values("data_alvo")
        datas = pd.to_datetime(sub["data_alvo"])

        ax.plot(datas, sub["y_real"], color=COR_REAL, linewidth=2, label="Actual")
        ax.plot(
            datas, sub["y_previsto"], color=COR_PREV, linewidth=1.8, linestyle="--", label=modelo
        )
        ax.plot(
            pd.to_datetime(naive["data_alvo"]),
            naive["y_previsto"],
            color=COR_AUX,
            linewidth=1.2,
            linestyle=":",
            label="naive",
        )
        erro = (sub["y_real"] - sub["y_previsto"]).abs().mean()
        ax.set_title(f"h = {h} month(s) — MAE {erro:.3f} BRL/kg", loc="left", fontsize=11)
        ax.set_ylabel("BRL/kg")
        _estilo(ax)
        if h == horizontes[0]:
            ax.legend(frameon=False, ncol=3, loc="upper left")

    fig.suptitle(
        "Walk-forward backtest: predicted vs actual",
        x=0.01,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "backtest_previsto_vs_real.png", dpi=150)
    plt.close(fig)


def figura_metricas(metricas: pd.DataFrame) -> None:
    """MAE by model and horizon — the one chart a reviewer actually reads."""
    pivot = metricas.pivot(index="modelo", columns="horizonte", values="mae")
    pivot = pivot.sort_values(pivot.columns[0])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    pivot.plot(kind="barh", ax=ax, width=0.78, colormap="copper")
    ax.set_xlabel("MAE (BRL/kg) — lower is better")
    ax.set_ylabel("")
    ax.set_title(
        "Error by model and horizon — baselines included",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax.legend(title="horizon (months)", frameon=False)
    _estilo(ax)
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mae_por_modelo.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    predicoes = pd.read_csv(PROCESSED_DIR / "backtest_previsoes.csv")
    metricas = pd.read_csv(PROCESSED_DIR / "backtest_metricas.csv")

    figura_serie(panel)
    figura_backtest(predicoes)
    figura_metricas(metricas)
    print(f"Figures saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
