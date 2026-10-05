"""The learned forecaster.

Two decisions here carry most of the weight, and both are about the fact that a
commodity price in levels is non-stationary:

1. **The model predicts the change, not the level.** A tree ensemble can only
   output values it saw in training. Fed 2013-2021 prices and asked about 2024,
   a level-target model is structurally incapable of predicting a price above
   the highest it ever saw — it will flat-line at the ceiling. Modelling
   ``y[t+h] - y[t]`` and adding the result back to the last observed price
   removes that ceiling entirely. This single change is usually worth more than
   any amount of hyper-parameter tuning.

2. **Gradient boosting over linear regression** — but the linear model stays in
   the repo and in the backtest, because on short series it frequently wins,
   and reporting only the model that happens to be fashionable is how portfolio
   projects lose credibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_STATE, TARGET

ANCHOR_COLUMN = f"{TARGET}_lag0"


def make_gbm() -> HistGradientBoostingRegressor:
    """Small, heavily-regularised GBM: ~100 training rows is not a lot of data."""
    return HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=8,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def make_ridge() -> Pipeline:
    """Standardised ridge with an internal alpha search."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]
    )


ESTIMATORS: dict[str, callable] = {
    "gbm": make_gbm,
    "ridge": make_ridge,
}


@dataclass
class PriceForecaster:
    """Direct h-step forecaster with an optional difference target.

    Parameters
    ----------
    horizon:
        Months ahead this instance predicts. One instance per horizon.
    estimator_name:
        Key into :data:`ESTIMATORS`.
    target_mode:
        ``"delta"`` models ``y[t+h] - y[t]`` (default, see module docstring);
        ``"level"`` models ``y[t+h]`` directly and exists so the backtest can
        demonstrate *why* delta is the default.
    """

    horizon: int
    estimator_name: str = "gbm"
    target_mode: str = "delta"
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.estimator_name not in ESTIMATORS:
            raise ValueError(
                f"unknown estimator {self.estimator_name!r}; "
                f"choose from {sorted(ESTIMATORS)}"
            )
        if self.target_mode not in {"delta", "level"}:
            raise ValueError("target_mode must be 'delta' or 'level'")
        self.estimator = ESTIMATORS[self.estimator_name]()

    @property
    def name(self) -> str:
        return f"{self.estimator_name}_{self.target_mode}"

    @staticmethod
    def _anchor(X: pd.DataFrame) -> np.ndarray:
        if ANCHOR_COLUMN not in X.columns:
            raise KeyError(
                f"{ANCHOR_COLUMN} is required to anchor a delta forecast; "
                "it is produced by features.build_feature_frame"
            )
        return X[ANCHOR_COLUMN].to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PriceForecaster:
        self.feature_names = list(X.columns)
        target = y.to_numpy(dtype=float)
        if self.target_mode == "delta":
            target = target - self._anchor(X)
        self.estimator.fit(X.to_numpy(dtype=float), target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names and list(X.columns) != self.feature_names:
            X = X.loc[:, self.feature_names]
        pred = np.asarray(self.estimator.predict(X.to_numpy(dtype=float)), dtype=float)
        if self.target_mode == "delta":
            pred = pred + self._anchor(X)
        return pred

    def permutation_importance(
        self, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10
    ) -> pd.DataFrame:
        """Which features actually carry signal, measured by shuffling them.

        Used for the model card. Permutation importance is computed on held-out
        origins, never on the training rows, or it just reports memorisation.
        """
        from sklearn.inspection import permutation_importance as sk_perm

        target = y.to_numpy(dtype=float)
        if self.target_mode == "delta":
            target = target - self._anchor(X)
        result = sk_perm(
            self.estimator,
            X.to_numpy(dtype=float),
            target,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )
        return (
            pd.DataFrame(
                {
                    "feature": list(X.columns),
                    "importancia": result.importances_mean,
                    "desvio": result.importances_std,
                }
            )
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True)
        )
