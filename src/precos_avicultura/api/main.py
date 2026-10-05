"""FastAPI service.

Design notes worth defending in an interview:

* Artifacts are loaded once at startup and cached, not re-read per request.
* The caller sends the history; the service does not own a database. That keeps
  the deployable unit stateless and horizontally scalable, and it makes the
  service trivially testable.
* Every prediction ships with the backtest error of the model that produced it.
  A point forecast without its historical error is a number pretending to be
  information.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from .. import __version__
from ..config import DATE_COL, HORIZONS, MODELS_DIR, TARGET
from ..data.loaders import validate_panel
from ..features import latest_origin_features
from ..models.train import artifact_path, load_artifact
from .schemas import (
    HealthResponse,
    ModelCardResponse,
    PrevisaoItem,
    PrevisaoRequest,
    PrevisaoResponse,
)

AVISO = (
    "Previsao estatistica para apoio a decisao de compra e venda. "
    "Nao constitui recomendacao de investimento nem garantia de preco futuro."
)

app = FastAPI(
    title="Previsao de precos - avicultura",
    version=__version__,
    description=(
        "Forecasts the monthly broiler price 1-3 months ahead from feed cost, "
        "FX and inflation drivers. Every response carries the walk-forward "
        "error of the model that produced it."
    ),
)


def models_dir() -> Path:
    """Where artifacts live, resolved at call time.

    Reading the environment on every call (instead of binding a default at
    import time) is what lets the same image be pointed at a mounted volume in
    production and at a temp directory in the test suite.
    """
    return Path(os.getenv("PRECOS_MODELS_DIR", str(MODELS_DIR)))


@cache
def _load_cached(horizon: int, directory: str) -> dict:
    return load_artifact(horizon, Path(directory))


def _artifact(horizon: int) -> dict:
    return _load_cached(horizon, str(models_dir()))


def available_horizons() -> list[int]:
    directory = models_dir()
    return [h for h in HORIZONS if artifact_path(h, directory).exists()]


@app.get("/health", response_model=HealthResponse, tags=["infra"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        versao=__version__,
        horizontes_disponiveis=available_horizons(),
    )


@app.get("/modelo/{horizonte}", response_model=ModelCardResponse, tags=["modelo"])
def model_card(horizonte: int) -> ModelCardResponse:
    """Provenance and measured accuracy for one horizon."""
    try:
        art = _artifact(horizonte)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ModelCardResponse(
        horizonte=art["horizonte"],
        modelo=art["modelo"],
        treinado_em=art["treinado_em"],
        janela_treino=art["janela_treino"],
        hash_dados=art["hash_dados"],
        metricas_backtest=art["metricas_backtest"],
        principais_variaveis=art["importancias"][:5],
    )


@app.post("/prever", response_model=PrevisaoResponse, tags=["modelo"])
def prever(req: PrevisaoRequest) -> PrevisaoResponse:
    """Forecast from a caller-supplied history."""
    disponiveis = available_horizons()
    if not disponiveis:
        raise HTTPException(
            status_code=503,
            detail="no trained artifact found; run `make train` before serving",
        )

    horizontes = req.horizontes or disponiveis
    desconhecidos = sorted(set(horizontes) - set(disponiveis))
    if desconhecidos:
        raise HTTPException(
            status_code=400,
            detail=f"horizons {desconhecidos} were not trained; available: {disponiveis}",
        )

    panel = pd.DataFrame([obs.model_dump() for obs in req.historico])
    try:
        panel = validate_panel(panel)
        X = latest_origin_features(panel)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"invalid history: {exc}") from exc

    origem = pd.to_datetime(panel[DATE_COL]).max()
    preco_origem = float(panel.loc[panel[DATE_COL] == origem, TARGET].iloc[0])

    previsoes: list[PrevisaoItem] = []
    for horizonte in sorted(horizontes):
        art = _artifact(horizonte)
        faltando = [c for c in art["feature_names"] if c not in X.columns]
        if faltando:
            raise HTTPException(
                status_code=500,
                detail=f"feature contract mismatch for h={horizonte}: missing {faltando[:5]}",
            )
        valor = float(art["model"].predict(X)[0])
        metricas = art.get("metricas_backtest", {})
        previsoes.append(
            PrevisaoItem(
                horizonte=horizonte,
                data_alvo=(origem + pd.DateOffset(months=horizonte)).strftime("%Y-%m-%d"),
                preco_previsto=round(valor, 3),
                variacao_prevista_pct=round((valor / preco_origem - 1) * 100, 2),
                modelo=art["modelo"],
                mae_backtest=metricas.get("mae"),
                mase_backtest=metricas.get("mase"),
            )
        )

    return PrevisaoResponse(
        origem=origem.strftime("%Y-%m-%d"),
        preco_origem=round(preco_origem, 3),
        previsoes=previsoes,
        aviso=AVISO,
    )
