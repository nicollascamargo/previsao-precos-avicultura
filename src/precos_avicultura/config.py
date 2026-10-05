"""Central configuration: paths, column contracts and default hyper-parameters.

Keeping these in one place means the notebook, the CLI, the tests and the API
all agree on what a "dataset" looks like. Any change to the schema breaks in a
single, obvious spot instead of silently drifting across modules.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

PANEL_PATH = PROCESSED_DIR / "panel_mensal.csv"

# ---------------------------------------------------------------- schema ----
DATE_COL = "data"
TARGET = "preco_frango"

#: Exogenous drivers. Corn and soy are the two largest cost components of a
#: broiler's feed conversion; FX matters because both are exported commodities
#: priced off the international market; IPCA anchors everything in real terms.
EXOG: tuple[str, ...] = (
    "preco_milho",
    "preco_soja",
    "cambio_usdbrl",
    "ipca",
)

COLUMNS: tuple[str, ...] = (DATE_COL, TARGET, *EXOG)

# ------------------------------------------------------------- modelling ----
#: Forecast horizons in months. Direct strategy: one model per horizon.
HORIZONS: tuple[int, ...] = (1, 2, 3)

#: Lags applied to every series. Lag 0 is the value observed at the forecast
#: origin itself, which is legitimately available at prediction time.
LAGS: tuple[int, ...] = (0, 1, 2, 3, 6, 12)

#: Rolling-window means, all ending at the forecast origin (inclusive).
ROLLING_WINDOWS: tuple[int, ...] = (3, 6, 12)

#: Seasonal period of a monthly series.
SEASONAL_PERIOD = 12

#: Number of most recent observations held out for walk-forward evaluation.
DEFAULT_TEST_SIZE = 24

#: Minimum number of training rows before the first forecast origin.
MIN_TRAIN_SIZE = 36

RANDOM_STATE = 42
