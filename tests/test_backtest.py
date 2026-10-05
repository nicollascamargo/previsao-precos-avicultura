"""Walk-forward, metric and data-contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precos_avicultura.backtest import score_predictions, walk_forward
from precos_avicultura.config import DATE_COL, TARGET
from precos_avicultura.data.loaders import PanelValidationError, validate_panel
from precos_avicultura.data.synthetic import make_synthetic_panel
from precos_avicultura.evaluation.metrics import mae, mase, rmse, smape
from precos_avicultura.features import build_supervised
from precos_avicultura.models.baselines import DriftForecast, NaiveLast, SeasonalNaive


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_synthetic_panel(n_months=132, seed=11)


@pytest.fixture(scope="module")
def predicoes(panel: pd.DataFrame) -> pd.DataFrame:
    return walk_forward(
        panel, horizon=1, model_specs=[("ridge", "delta")], test_size=12
    )


def test_treino_nunca_usa_rotulo_futuro(panel: pd.DataFrame, predicoes: pd.DataFrame) -> None:
    """At origin t the training set may only contain labels observed by t.

    This is the single assertion that separates a credible backtest from an
    optimistic one, so it is recomputed here from scratch rather than trusted:
    the number of training rows reported by the loop must equal the number of
    origins whose target date falls on or before the forecast origin.
    """
    horizon = int(predicoes["horizonte"].iloc[0])
    X, _ = build_supervised(panel, horizon=horizon)

    for origem in pd.to_datetime(sorted(predicoes["origem"].unique())):
        corte = origem - pd.DateOffset(months=horizon)
        esperado = int((X.index <= corte).sum())
        reportado = int(predicoes.loc[predicoes["origem"] == origem, "n_treino"].iloc[0])
        assert reportado == esperado, f"origin {origem.date()}: {reportado} != {esperado}"
        # And the strict version: every training label predates the origin.
        assert (X.index[X.index <= corte] + pd.DateOffset(months=horizon) <= origem).all()


def test_data_alvo_coerente(predicoes: pd.DataFrame) -> None:
    h = int(predicoes["horizonte"].iloc[0])
    esperado = pd.to_datetime(predicoes["origem"]) + pd.DateOffset(months=h)
    assert (pd.to_datetime(predicoes["data_alvo"]) == esperado).all()


def test_backtest_sem_nan(predicoes: pd.DataFrame) -> None:
    assert not predicoes[["y_real", "y_previsto", "y_origem"]].isna().any().any()


def test_score_retorna_todos_os_modelos(panel: pd.DataFrame, predicoes: pd.DataFrame) -> None:
    scores = score_predictions(predicoes, panel)
    assert {"naive", "naive_sazonal", "drift", "ridge_delta"} <= set(scores["modelo"])
    assert (scores["n"] > 0).all()


def test_janela_de_teste_e_respeitada(panel: pd.DataFrame) -> None:
    pred = walk_forward(panel, horizon=2, model_specs=[("ridge", "delta")], test_size=10)
    assert pred["origem"].nunique() <= 10


def test_min_train_size_bloqueia_serie_curta() -> None:
    curto = make_synthetic_panel(n_months=40, seed=3)
    with pytest.raises(ValueError):
        walk_forward(curto, horizon=1, test_size=5, min_train_size=200)


# ------------------------------------------------------------- baselines ----
def test_naive_repete_ultimo_valor() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    assert NaiveLast().fit(y).predict(3) == 3.0


def test_seasonal_naive_pega_mesmo_mes() -> None:
    y = pd.Series(np.arange(24, dtype=float))
    # Origin is index 23; forecasting h=1 should return the value 12 months
    # before the target date, i.e. index 12.
    assert SeasonalNaive(12).fit(y).predict(1) == 12.0


def test_drift_extrapola_tendencia() -> None:
    y = pd.Series(np.arange(24, dtype=float))
    assert DriftForecast(window=12).fit(y).predict(2) == pytest.approx(25.0)


# --------------------------------------------------------------- metrics ----
def test_metricas_zeram_em_previsao_perfeita() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0
    assert rmse(y, y) == 0
    assert smape(y, y) == 0


def test_mase_menor_que_um_quando_melhor_que_naive() -> None:
    # Trending series: the seasonal-naive error scale is 12 units per year.
    treino = np.arange(36, dtype=float)
    y_true = np.array([2.0, 2.0])
    quase_certo = np.array([2.05, 1.95])
    assert mase(y_true, quase_certo, treino, seasonal_period=12) < 1


def test_mase_indefinida_quando_escala_e_zero() -> None:
    """A perfectly seasonal series gives a zero denominator; return NaN, not inf."""
    treino = np.array([1.0, 3.0] * 18)  # period 2: lag-12 differences are all 0
    resultado = mase(np.array([2.0]), np.array([2.5]), treino, seasonal_period=12)
    assert np.isnan(resultado)


# ------------------------------------------------------------ validation ----
def test_validate_rejeita_mes_faltante(panel: pd.DataFrame) -> None:
    com_buraco = panel.drop(index=50).reset_index(drop=True)
    with pytest.raises(PanelValidationError, match="contiguous"):
        validate_panel(com_buraco)


def test_validate_rejeita_coluna_ausente(panel: pd.DataFrame) -> None:
    with pytest.raises(PanelValidationError, match="missing required columns"):
        validate_panel(panel.drop(columns=[TARGET]))


def test_validate_ordena_por_data(panel: pd.DataFrame) -> None:
    embaralhado = panel.sample(frac=1, random_state=0).reset_index(drop=True)
    ok = validate_panel(embaralhado)
    assert ok[DATE_COL].is_monotonic_increasing
