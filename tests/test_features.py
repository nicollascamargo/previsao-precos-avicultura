"""The tests that actually matter in a forecasting project.

Anyone can assert that a DataFrame has the right number of columns. The test
that earns its keep is the leakage test: it mutates the future and asserts that
the past did not move. If a single `.shift(-1)` ever slips into the feature
code, this fails immediately — and it is the only thing standing between an
honest backtest and a spectacular, meaningless R-squared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from precos_avicultura.config import DATE_COL, TARGET
from precos_avicultura.data.synthetic import make_synthetic_panel
from precos_avicultura.features import (
    build_feature_frame,
    build_supervised,
    latest_origin_features,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_synthetic_panel(n_months=120, seed=7)


def test_features_nao_enxergam_o_futuro(panel: pd.DataFrame) -> None:
    """Perturbing month t+1..T must not change any feature at origin t."""
    cut = 80
    original = build_feature_frame(panel)

    corrompido = panel.copy()
    numeric_cols = [c for c in corrompido.columns if c != DATE_COL]
    rng = np.random.default_rng(0)
    corrompido.loc[cut + 1 :, numeric_cols] *= rng.uniform(5, 20, size=1)[0]

    perturbado = build_feature_frame(corrompido)

    inicio = original.index[cut]
    pd.testing.assert_frame_equal(
        original.loc[:inicio],
        perturbado.loc[:inicio],
        check_exact=False,
        rtol=1e-12,
        obj="features up to the forecast origin",
    )


def test_label_e_o_alvo_h_meses_a_frente(panel: pd.DataFrame) -> None:
    horizon = 3
    X, y = build_supervised(panel, horizon=horizon)
    serie = panel.set_index(pd.to_datetime(panel[DATE_COL]))[TARGET]

    for origem in X.index[:10]:
        data_alvo = origem + pd.DateOffset(months=horizon)
        assert y.loc[origem] == pytest.approx(serie.loc[data_alvo])


def test_lag0_e_o_valor_da_origem(panel: pd.DataFrame) -> None:
    """The anchor used by the delta model must be the price at the origin."""
    X = build_feature_frame(panel)
    serie = panel.set_index(pd.to_datetime(panel[DATE_COL]))[TARGET]
    completos = X.dropna().index
    assert np.allclose(
        X.loc[completos, f"{TARGET}_lag0"].to_numpy(),
        serie.loc[completos].to_numpy(),
    )


def test_supervised_sem_nan(panel: pd.DataFrame) -> None:
    X, y = build_supervised(panel, horizon=1)
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert len(X) == len(y)
    assert X.index.equals(y.index)


def test_horizonte_invalido(panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_supervised(panel, horizon=0)


def test_latest_origin_mantem_ultima_linha(panel: pd.DataFrame) -> None:
    """Live prediction uses the final month, whose label does not exist yet."""
    X = latest_origin_features(panel)
    assert len(X) == 1
    assert X.index[0] == pd.to_datetime(panel[DATE_COL]).max()


def test_historico_curto_falha_alto() -> None:
    curto = make_synthetic_panel(n_months=8, seed=1)
    with pytest.raises(ValueError, match="not enough history"):
        latest_origin_features(curto)
