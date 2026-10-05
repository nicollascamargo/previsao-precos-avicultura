"""Baselines.

A gradient boosting model that cannot beat "next month will look like this
month" is not a result, it is a liability. These three baselines are the bar,
and they are evaluated through exactly the same walk-forward loop as the
learned model — same origins, same labels, same metrics.

All forecasters share a minimal interface:

    fit(y: pd.Series) -> self
    predict(horizon: int) -> float
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveLast:
    """Random-walk forecast: the last observed value, carried forward.

    For most commodity price series in levels this is a genuinely hard
    benchmark, which is the whole point of using it.
    """

    name = "naive"

    def fit(self, y: pd.Series) -> NaiveLast:
        self._last = float(y.iloc[-1])
        return self

    def predict(self, horizon: int) -> float:
        return self._last


class SeasonalNaive:
    """Value observed the same month last year."""

    name = "naive_sazonal"

    def __init__(self, seasonal_period: int = 12) -> None:
        self.seasonal_period = seasonal_period

    def fit(self, y: pd.Series) -> SeasonalNaive:
        self._y = y.to_numpy(dtype=float)
        return self

    def predict(self, horizon: int) -> float:
        m = self.seasonal_period
        n = len(self._y)
        if n < m:
            return float(self._y[-1])
        # Textbook seasonal naive: the observation at the same position of the
        # most recent complete season. For h > m it wraps and reuses that same
        # season rather than inventing one.
        pos = n - m + ((horizon - 1) % m)
        return float(self._y[pos])


class DriftForecast:
    """Random walk with drift estimated over the last ``window`` months."""

    name = "drift"

    def __init__(self, window: int = 12) -> None:
        self.window = window

    def fit(self, y: pd.Series) -> DriftForecast:
        arr = y.to_numpy(dtype=float)
        self._last = float(arr[-1])
        w = min(self.window, len(arr) - 1)
        self._slope = float((arr[-1] - arr[-1 - w]) / w) if w > 0 else 0.0
        return self

    def predict(self, horizon: int) -> float:
        return self._last + self._slope * horizon


class SarimaxForecast:
    """SARIMAX(1,1,1)(1,0,1,12) — the classical statistical reference.

    Optional: ``statsmodels`` is only imported when this class is used, and the
    walk-forward loop skips it unless explicitly requested, because refitting a
    state-space model at every origin is slow and CI does not need it.
    """

    name = "sarimax"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, y: pd.Series) -> SarimaxForecast:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._res = SARIMAX(
            y.astype(float),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, horizon: int) -> float:
        return float(np.asarray(self._res.forecast(steps=horizon))[-1])


BASELINES: dict[str, type] = {
    NaiveLast.name: NaiveLast,
    SeasonalNaive.name: SeasonalNaive,
    DriftForecast.name: DriftForecast,
}
