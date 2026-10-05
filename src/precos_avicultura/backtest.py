"""Expanding-window walk-forward evaluation.

A random train/test split on a time series is not a weak evaluation, it is a
wrong one: it lets the model train on 2024 to predict 2019. This module does
the honest thing instead.

For every forecast origin ``t`` in the test window:

* the model is refit from scratch on rows whose label was **already observable
  at t** — that is, origins ``s`` with ``s + h <= t``;
* it then predicts the single row at ``t``;
* the baselines see the exact same history and the same origin.

Refitting at every origin is more expensive than fitting once, and it is the
only way the reported error resembles what the model would have produced in
production.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    DATE_COL,
    DEFAULT_TEST_SIZE,
    MIN_TRAIN_SIZE,
    SEASONAL_PERIOD,
    TARGET,
)
from .evaluation.metrics import results_table, summarize
from .features import build_supervised
from .models.baselines import BASELINES, SarimaxForecast
from .models.forecaster import PriceForecaster


def _target_series(panel: pd.DataFrame) -> pd.Series:
    df = panel.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values(DATE_COL).set_index(DATE_COL)[TARGET].astype(float)


def walk_forward(
    panel: pd.DataFrame,
    horizon: int,
    model_specs: list[tuple[str, str]] | None = None,
    test_size: int = DEFAULT_TEST_SIZE,
    min_train_size: int = MIN_TRAIN_SIZE,
    include_baselines: bool = True,
    with_sarimax: bool = False,
) -> pd.DataFrame:
    """Run the walk-forward loop for one horizon.

    Parameters
    ----------
    model_specs:
        List of ``(estimator_name, target_mode)`` pairs. Defaults to the GBM and
        the ridge, both on the delta target, plus the GBM on levels so the
        comparison in the README is reproducible.

    Returns
    -------
    Long-format DataFrame with one row per (origin, model): columns
    ``origem``, ``data_alvo``, ``horizonte``, ``modelo``, ``y_real``,
    ``y_previsto``, ``y_origem``.
    """
    if model_specs is None:
        model_specs = [("gbm", "delta"), ("ridge", "delta"), ("gbm", "level")]

    X, y = build_supervised(panel, horizon=horizon)
    levels = _target_series(panel)

    if len(X) <= min_train_size:
        raise ValueError(
            f"only {len(X)} usable origins for horizon {horizon}; "
            f"need more than min_train_size={min_train_size}"
        )

    test_origins = X.index[-test_size:]
    rows: list[dict] = []

    for origin in test_origins:
        label_cutoff = origin - pd.DateOffset(months=horizon)
        train_mask = X.index <= label_cutoff
        if train_mask.sum() < min_train_size:
            continue

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_origin = X.loc[[origin]]
        target_date = origin + pd.DateOffset(months=horizon)
        y_true = float(y.loc[origin])
        y_origin = float(levels.loc[origin])

        base = {
            "origem": origin,
            "data_alvo": target_date,
            "horizonte": horizon,
            "y_real": y_true,
            "y_origem": y_origin,
            "n_treino": int(train_mask.sum()),
        }

        for estimator_name, target_mode in model_specs:
            model = PriceForecaster(
                horizon=horizon,
                estimator_name=estimator_name,
                target_mode=target_mode,
            )
            model.fit(X_train, y_train)
            rows.append(
                {**base, "modelo": model.name, "y_previsto": float(model.predict(X_origin)[0])}
            )

        if include_baselines:
            history = levels.loc[:origin]
            for name, cls in BASELINES.items():
                forecaster = cls().fit(history)
                rows.append(
                    {**base, "modelo": name, "y_previsto": float(forecaster.predict(horizon))}
                )
            if with_sarimax:
                try:
                    forecaster = SarimaxForecast().fit(history)
                    rows.append(
                        {
                            **base,
                            "modelo": "sarimax",
                            "y_previsto": float(forecaster.predict(horizon)),
                        }
                    )
                except Exception as exc:  # pragma: no cover - numerical failure
                    rows.append({**base, "modelo": "sarimax", "y_previsto": float("nan")})
                    print(f"  sarimax failed at {origin.date()}: {exc}")

    if not rows:
        raise ValueError("no forecast origin satisfied min_train_size; shorten the test window")

    return pd.DataFrame(rows)


def score_predictions(
    predictions: pd.DataFrame, panel: pd.DataFrame, seasonal_period: int = SEASONAL_PERIOD
) -> pd.DataFrame:
    """Aggregate per-origin predictions into a model comparison table."""
    levels = _target_series(panel)
    first_origin = predictions["origem"].min()
    # MASE scale comes from data strictly before the test window, so the
    # denominator itself cannot be contaminated by the evaluation period.
    train_levels = levels.loc[levels.index < first_origin].to_numpy()

    records = []
    for (horizon, model), group in predictions.groupby(["horizonte", "modelo"]):
        group = group.dropna(subset=["y_previsto"])
        if group.empty:
            continue
        metrics = summarize(
            group["y_real"].to_numpy(),
            group["y_previsto"].to_numpy(),
            y_train=train_levels,
            y_origin=group["y_origem"].to_numpy(),
            seasonal_period=seasonal_period,
        )
        records.append({"horizonte": int(horizon), "modelo": model, **metrics})

    return results_table(records)


def run_full_backtest(
    panel: pd.DataFrame,
    horizons: tuple[int, ...],
    test_size: int = DEFAULT_TEST_SIZE,
    with_sarimax: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest every horizon; return ``(predictions, scores)``."""
    frames = [
        walk_forward(panel, horizon=h, test_size=test_size, with_sarimax=with_sarimax)
        for h in horizons
    ]
    predictions = pd.concat(frames, ignore_index=True)
    return predictions, score_predictions(predictions, panel)
