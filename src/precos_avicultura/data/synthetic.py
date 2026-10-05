"""Synthetic but economically-shaped data, so anyone can clone and run.

CEPEA/ESALQ redistributes its price series under terms that do not allow me to
ship a copy inside this repository. Rather than leave the project un-runnable,
`make demo` generates a panel with the same schema and the same causal
structure the real one has:

    feed cost (corn, soy)  --lagged 2-4 months-->  broiler price
    FX  --> commodity prices (both are exported)
    inflation --> nominal drift in everything

The lag from feed cost to broiler price is the whole reason this problem is
forecastable at all: a producer who fixes corn today is locked into a cost that
shows up at the slaughterhouse a quarter later. The generator encodes that lag
explicitly, which means a model that finds it will beat a naive forecast, and a
model that does not find it will not. That is exactly the signal a backtest is
supposed to measure.

Numbers are on a plausible BRL scale but are NOT real observations and must
never be presented as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DATE_COL, TARGET


def make_synthetic_panel(
    n_months: int = 156,
    start: str = "2013-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a monthly panel with the project schema.

    Parameters
    ----------
    n_months:
        Length of the series. 156 ~= 13 years, enough for a 24-month
        walk-forward test window plus a decent training history.
    start:
        First month of the series.
    seed:
        Reproducibility.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_months, freq="MS")
    t = np.arange(n_months)

    # --- FX: slow random walk with drift, floored to stay positive ----------
    cambio = 2.0 + np.cumsum(rng.normal(0.02, 0.09, n_months))
    cambio = np.clip(cambio, 1.6, None)

    # --- Inflation: mean-reverting monthly rate in percent ------------------
    ipca = np.zeros(n_months)
    ipca[0] = 0.45
    for i in range(1, n_months):
        ipca[i] = 0.35 + 0.55 * (ipca[i - 1] - 0.35) + rng.normal(0, 0.18)
    nivel_precos = np.cumprod(1 + ipca / 100)

    # --- Corn: seasonal (harvest pressure Feb-Jul), FX-linked, noisy -------
    sazonal_milho = -4.0 * np.sin(2 * np.pi * (t % 12) / 12)
    milho = (
        55
        + 9.0 * (cambio - cambio[0])
        + sazonal_milho
        + np.cumsum(rng.normal(0, 0.8, n_months))
    ) * nivel_precos
    milho = np.clip(milho, 25, None)

    # --- Soy: correlated with corn through the same export channel ---------
    soja = (
        110
        + 1.15 * (milho / nivel_precos - 55)
        + 14.0 * (cambio - cambio[0])
        + np.cumsum(rng.normal(0, 1.1, n_months))
    ) * nivel_precos
    soja = np.clip(soja, 60, None)

    # --- Broiler: feed cost passed through with a 2-4 month lag ------------
    custo_racao = 0.62 * milho + 0.38 * (soja / 2.0)
    passthrough = np.zeros(n_months)
    for lag, peso in ((2, 0.30), (3, 0.42), (4, 0.20)):
        passthrough[lag:] += peso * custo_racao[:-lag]
    passthrough[:4] = passthrough[4]

    sazonal_frango = 0.28 * np.sin(2 * np.pi * ((t % 12) - 2) / 12)  # Q4 demand
    frango = (
        4.10
        + 0.030 * (passthrough - passthrough.mean())
        + sazonal_frango
        + np.cumsum(rng.normal(0, 0.035, n_months))
    )
    frango = np.clip(frango, 2.5, None)

    panel = pd.DataFrame(
        {
            DATE_COL: idx,
            TARGET: np.round(frango, 3),
            "preco_milho": np.round(milho, 2),
            "preco_soja": np.round(soja, 2),
            "cambio_usdbrl": np.round(cambio, 4),
            "ipca": np.round(ipca, 3),
        }
    )
    return panel
