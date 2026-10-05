"""Feature engineering for a direct multi-horizon forecast.

The single most important invariant in this module:

    row at forecast origin ``t`` may only contain information observable at
    ``t``; the label for that row is the target at ``t + h``.

Every transformation below is written so that it cannot see past ``t``. Lag 0
means "the value published for month t", which is legitimately known at the
moment you forecast. Rolling windows end at ``t`` inclusive. Nothing uses
``.shift(-k)`` except the label itself.

``tests/test_features.py`` enforces this empirically by mutating the future and
asserting the feature matrix does not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    DATE_COL,
    EXOG,
    LAGS,
    ROLLING_WINDOWS,
    TARGET,
)


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Smooth month-of-year encoding.

    Fourier terms instead of 12 dummies: the seasonal effect in poultry is
    broad (Q4 demand, mid-year harvest pressure on feed), not month-specific,
    and two columns cost far fewer degrees of freedom than eleven.
    """
    month = index.month.to_numpy()
    return pd.DataFrame(
        {
            "mes_sin": np.sin(2 * np.pi * month / 12),
            "mes_cos": np.cos(2 * np.pi * month / 12),
            "mes_sin2": np.sin(4 * np.pi * month / 12),
            "mes_cos2": np.cos(4 * np.pi * month / 12),
        },
        index=index,
    )


def build_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the origin-indexed feature matrix (no labels).

    Parameters
    ----------
    panel:
        Validated monthly panel (see ``data.loaders.validate_panel``).

    Returns
    -------
    DataFrame indexed by forecast origin date.
    """
    df = panel.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).set_index(DATE_COL)

    series = [TARGET, *EXOG]
    feats: dict[str, pd.Series] = {}

    for col in series:
        s = df[col]
        for lag in LAGS:
            feats[f"{col}_lag{lag}"] = s.shift(lag)
        for window in ROLLING_WINDOWS:
            # min_periods=window keeps early rows NaN instead of averaging a
            # partial window, which would make the first observations mean
            # something different from the rest.
            feats[f"{col}_media{window}"] = s.rolling(window, min_periods=window).mean()
            feats[f"{col}_dp{window}"] = s.rolling(window, min_periods=window).std()
        # Momentum: month-over-month and year-over-year change.
        feats[f"{col}_var1"] = s.pct_change(1)
        feats[f"{col}_var12"] = s.pct_change(12)

    # --- domain features -----------------------------------------------------
    # Margin proxy: how many kilos of broiler one bag of corn costs. This is
    # the number a producer actually watches; when it compresses, supply is
    # cut a couple of months later and price rises.
    feats["relacao_frango_milho"] = df[TARGET] / df["preco_milho"]
    feats["relacao_frango_soja"] = df[TARGET] / df["preco_soja"]
    # Feed basket at the standard broiler inclusion rate (~62% corn, 38% meal).
    feats["custo_racao"] = 0.62 * df["preco_milho"] + 0.38 * (df["preco_soja"] / 2.0)
    feats["custo_racao_var3"] = feats["custo_racao"].pct_change(3)
    # Real (deflated) target: strips the part of the move that is just inflation.
    deflator = (1 + df["ipca"] / 100).cumprod()
    feats["preco_frango_real"] = df[TARGET] / deflator
    feats["ipca_acum3"] = df["ipca"].rolling(3, min_periods=3).sum()

    X = pd.DataFrame(feats, index=df.index)
    X = pd.concat([X, _calendar_features(df.index)], axis=1)
    return X


def build_supervised(
    panel: pd.DataFrame, horizon: int, dropna: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` for a direct ``horizon``-step-ahead model.

    ``X.index`` is the forecast origin. ``y.iloc[i]`` is the target observed
    ``horizon`` months after ``X.index[i]``. Rows whose label is not yet
    observable (the tail) are dropped when ``dropna`` is True.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    X = build_feature_frame(panel)

    df = panel.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    y = (
        df.sort_values(DATE_COL)
        .set_index(DATE_COL)[TARGET]
        .shift(-horizon)
        .rename(f"{TARGET}_h{horizon}")
    )

    if dropna:
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X.loc[mask], y.loc[mask]

    return X, y


def latest_origin_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Feature row for the most recent origin, for live prediction.

    Unlike ``build_supervised`` this keeps the final row even though its label
    does not exist yet — that is precisely the row you want to forecast from.
    """
    X = build_feature_frame(panel)
    X = X.loc[X.notna().all(axis=1)]
    if X.empty:
        raise ValueError(
            "not enough history to build a complete feature row; at least "
            f"{max(max(LAGS), max(ROLLING_WINDOWS), 12) + 1} months are required"
        )
    return X.iloc[[-1]]
