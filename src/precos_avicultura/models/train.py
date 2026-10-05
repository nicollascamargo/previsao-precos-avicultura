"""Fit the production models and write their artifacts.

An artifact here is not just a pickled estimator. It carries the metadata you
need six months later to answer "what is this thing and should I trust it":
the training window, the feature contract, the walk-forward metrics it was
approved on, and the top drivers. If the API cannot tell you how good a model
is, nobody should be calling it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ..backtest import score_predictions, walk_forward
from ..config import (
    DATE_COL,
    DEFAULT_TEST_SIZE,
    HORIZONS,
    MODELS_DIR,
    TARGET,
)
from ..features import build_supervised
from .forecaster import PriceForecaster

ARTIFACT_VERSION = 1


def _panel_fingerprint(panel: pd.DataFrame) -> str:
    """Short content hash, so a prediction can be traced back to its data."""
    payload = panel.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def artifact_path(horizon: int, models_dir: Path = MODELS_DIR) -> Path:
    return Path(models_dir) / f"modelo_h{horizon}.joblib"


def train_horizon(
    panel: pd.DataFrame,
    horizon: int,
    estimator_name: str = "gbm",
    target_mode: str = "delta",
    test_size: int = DEFAULT_TEST_SIZE,
    models_dir: Path = MODELS_DIR,
    evaluate: bool = True,
) -> dict:
    """Backtest, then refit on all available data and persist the artifact."""
    metrics: dict = {}
    importances: list[dict] = []

    if evaluate:
        predictions = walk_forward(
            panel,
            horizon=horizon,
            model_specs=[(estimator_name, target_mode)],
            test_size=test_size,
        )
        scores = score_predictions(predictions, panel)
        chosen = f"{estimator_name}_{target_mode}"
        row = scores.loc[scores["modelo"] == chosen]
        if not row.empty:
            metrics = row.drop(columns=["horizonte", "modelo"]).iloc[0].to_dict()
        baselines = scores.loc[scores["modelo"] != chosen, ["modelo", "mae"]]
        metrics["mae_baselines"] = {
            str(r.modelo): float(r.mae) for r in baselines.itertuples()
        }

    X, y = build_supervised(panel, horizon=horizon)
    model = PriceForecaster(
        horizon=horizon, estimator_name=estimator_name, target_mode=target_mode
    )
    model.fit(X, y)

    if evaluate and len(X) > 24:
        # Importance measured on the most recent quarter of origins.
        holdout = max(12, len(X) // 4)
        importances = (
            model.permutation_importance(X.iloc[-holdout:], y.iloc[-holdout:])
            .head(12)
            .to_dict("records")
        )

    panel_dates = pd.to_datetime(panel[DATE_COL])
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "horizonte": horizon,
        "modelo": model.name,
        "target": TARGET,
        "model": model,
        "feature_names": model.feature_names,
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "janela_treino": {
            "inicio": panel_dates.min().strftime("%Y-%m"),
            "fim": panel_dates.max().strftime("%Y-%m"),
            "n_observacoes": int(len(panel)),
            "n_origens": int(len(X)),
        },
        "hash_dados": _panel_fingerprint(panel),
        "metricas_backtest": metrics,
        "importancias": importances,
    }

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path(horizon, models_dir))
    return artifact


def train_all(
    panel: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
    estimator_name: str = "gbm",
    target_mode: str = "delta",
    test_size: int = DEFAULT_TEST_SIZE,
    models_dir: Path = MODELS_DIR,
    evaluate: bool = True,
) -> dict:
    """Train one model per horizon and write ``models/model_card.json``."""
    cards = {}
    for horizon in horizons:
        artifact = train_horizon(
            panel,
            horizon=horizon,
            estimator_name=estimator_name,
            target_mode=target_mode,
            test_size=test_size,
            models_dir=models_dir,
            evaluate=evaluate,
        )
        cards[str(horizon)] = {
            k: v for k, v in artifact.items() if k != "model"
        }

    card_path = Path(models_dir) / "model_card.json"
    card_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    return cards


def load_artifact(horizon: int, models_dir: Path = MODELS_DIR) -> dict:
    path = artifact_path(horizon, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make train` (or `make demo`) before serving."
        )
    return joblib.load(path)
