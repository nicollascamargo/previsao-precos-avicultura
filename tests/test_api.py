"""API tests against a real artifact trained into a temp directory."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from precos_avicultura.api import main as api_main
from precos_avicultura.data.synthetic import make_synthetic_panel
from precos_avicultura.models.train import train_all


@pytest.fixture(scope="module")
def modelos_dir(tmp_path_factory) -> Path:
    """Train two quick horizons; evaluation is skipped to keep the suite fast."""
    destino = tmp_path_factory.mktemp("modelos")
    panel = make_synthetic_panel(n_months=120, seed=5)
    train_all(
        panel,
        horizons=(1, 2),
        estimator_name="ridge",
        models_dir=destino,
        evaluate=False,
    )
    return destino


@pytest.fixture(scope="module")
def client(modelos_dir: Path):
    """Serve the temp artifacts by setting the same env var production uses."""
    anterior = os.environ.get("PRECOS_MODELS_DIR")
    os.environ["PRECOS_MODELS_DIR"] = str(modelos_dir)
    api_main._load_cached.cache_clear()
    try:
        with TestClient(api_main.app) as c:
            yield c
    finally:
        api_main._load_cached.cache_clear()
        if anterior is None:
            os.environ.pop("PRECOS_MODELS_DIR", None)
        else:
            os.environ["PRECOS_MODELS_DIR"] = anterior


@pytest.fixture(scope="module")
def historico() -> list[dict]:
    panel = make_synthetic_panel(n_months=40, seed=99)
    panel["data"] = pd.to_datetime(panel["data"]).dt.strftime("%Y-%m-%d")
    return panel.to_dict("records")


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_card_traz_proveniencia(client) -> None:
    resp = client.get("/modelo/1")
    assert resp.status_code == 200
    card = resp.json()
    assert card["horizonte"] == 1
    assert card["janela_treino"]["n_observacoes"] == 120
    assert card["hash_dados"]


def test_prever_retorna_um_item_por_horizonte(client, historico) -> None:
    resp = client.post("/prever", json={"historico": historico, "horizontes": [1, 2]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["previsoes"]) == 2
    assert body["previsoes"][0]["horizonte"] == 1
    assert body["preco_origem"] > 0
    assert "recomendacao de investimento" in body["aviso"]


def test_previsao_e_plausivel(client, historico) -> None:
    """A 1-month forecast should stay within a sane band of the last price."""
    resp = client.post("/prever", json={"historico": historico, "horizontes": [1]})
    body = resp.json()
    previsto = body["previsoes"][0]["preco_previsto"]
    origem = body["preco_origem"]
    assert 0.5 * origem < previsto < 1.8 * origem


def test_historico_curto_e_rejeitado(client, historico) -> None:
    resp = client.post("/prever", json={"historico": historico[:10]})
    assert resp.status_code == 422


def test_horizonte_nao_treinado(client, historico) -> None:
    resp = client.post("/prever", json={"historico": historico, "horizontes": [3]})
    assert resp.status_code == 400
    assert "were not trained" in resp.json()["detail"]


def test_historico_com_buraco_e_rejeitado(client, historico) -> None:
    com_buraco = historico[:20] + historico[22:]
    resp = client.post("/prever", json={"historico": com_buraco})
    assert resp.status_code == 422
