"""Forecast accuracy metrics.

RMSE alone tells you nothing about whether a model is worth deploying — a
number in BRL/kg has no natural reference point. MASE fixes that: it divides
the error by the error of a seasonal-naive forecast computed on the *training*
window, so a value below 1 means the model genuinely adds information and a
value above 1 means you should ship the naive rule instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(err**2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent. Undefined points (both zero) are skipped."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 12,
) -> float:
    """Mean Absolute Scaled Error against an in-sample seasonal-naive forecast.

    Falls back to a one-step naive scale when the training window is shorter
    than one full season.
    """
    y_train = np.asarray(y_train, dtype=float)
    step = seasonal_period if len(y_train) > seasonal_period else 1
    scale = np.mean(np.abs(y_train[step:] - y_train[:-step]))
    if not np.isfinite(scale) or scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / scale


def directional_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, y_origin: np.ndarray
) -> float:
    """Share of origins where the sign of the predicted move was right.

    For a buyer deciding whether to close a contract now or wait, getting the
    direction right often matters more than shaving cents off the RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_origin = np.asarray(y_origin, dtype=float)
    real = np.sign(y_true - y_origin)
    pred = np.sign(y_pred - y_origin)
    # A forecast equal to the origin makes no directional call at all. Scoring
    # it as "wrong" would report 0% for the naive baseline, which is misleading
    # — it is undefined, not bad. Those origins are excluded.
    mask = (real != 0) & (pred != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(real[mask] == pred[mask]))


def summarize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    y_origin: np.ndarray | None = None,
    seasonal_period: int = 12,
) -> dict[str, float]:
    """All metrics in one dict, ready to go into a results table."""
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, y_train, seasonal_period),
        "n": int(len(y_true)),
    }
    if y_origin is not None:
        out["acerto_direcao"] = directional_accuracy(y_true, y_pred, y_origin)
    return out


def results_table(records: list[dict]) -> pd.DataFrame:
    """Tidy comparison table sorted by horizon then MAE."""
    df = pd.DataFrame(records)
    sort_cols = [c for c in ("horizonte", "mae") if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df
