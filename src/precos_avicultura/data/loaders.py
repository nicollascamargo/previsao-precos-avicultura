"""Loading and validating the monthly panel.

Two paths into the project:

1. ``load_panel()`` reads ``data/processed/panel_mensal.csv`` — whatever you
   put there, real or synthetic.
2. ``build_real_panel()`` documents how to assemble the real series. It is
   deliberately a thin, explicit function: every source is named, with its
   frequency and its aggregation rule, so the provenance of each column is
   auditable instead of buried in a notebook cell.

Both end in ``validate_panel()``, which is the only contract the rest of the
codebase depends on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import COLUMNS, DATE_COL, EXOG, PANEL_PATH, TARGET

#: Where each column comes from when you run this on real data.
#: Daily series are aggregated to monthly means; monthly series are used as-is.
SOURCES: dict[str, dict[str, str]] = {
    TARGET: {
        "source": "CEPEA/ESALQ - Indicador do frango congelado (SP)",
        "url": "https://www.cepea.esalq.usp.br/br/indicador/frango.aspx",
        "frequency": "daily",
        "aggregation": "monthly mean",
        "unit": "BRL/kg",
    },
    "preco_milho": {
        "source": "CEPEA/ESALQ - Indicador do milho (Campinas)",
        "url": "https://www.cepea.esalq.usp.br/br/indicador/milho.aspx",
        "frequency": "daily",
        "aggregation": "monthly mean",
        "unit": "BRL/60kg bag",
    },
    "preco_soja": {
        "source": "CEPEA/ESALQ - Indicador da soja (Paranagua)",
        "url": "https://www.cepea.esalq.usp.br/br/indicador/soja.aspx",
        "frequency": "daily",
        "aggregation": "monthly mean",
        "unit": "BRL/60kg bag",
    },
    "cambio_usdbrl": {
        "source": "Banco Central do Brasil - SGS series 3698 (PTAX sale)",
        "url": "https://www3.bcb.gov.br/sgspub/",
        "frequency": "daily",
        "aggregation": "monthly mean",
        "unit": "BRL per USD",
    },
    "ipca": {
        "source": "IBGE/SIDRA - IPCA monthly variation, table 1737",
        "url": "https://sidra.ibge.gov.br/tabela/1737",
        "frequency": "monthly",
        "aggregation": "as published",
        "unit": "% month over month",
    },
}


class PanelValidationError(ValueError):
    """Raised when a panel does not satisfy the project's data contract."""


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema, ordering and monthly continuity; return a clean copy.

    A forecasting pipeline that silently accepts a gap in a monthly index will
    produce lag features that quietly mean something different from what their
    names say. Failing loudly here is cheaper than debugging that later.
    """
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise PanelValidationError(f"missing required columns: {missing}")

    out = df.loc[:, list(COLUMNS)].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    if out[DATE_COL].duplicated().any():
        dupes = out.loc[out[DATE_COL].duplicated(), DATE_COL].tolist()
        raise PanelValidationError(f"duplicated months: {dupes[:5]}")

    expected = pd.date_range(
        out[DATE_COL].iloc[0], out[DATE_COL].iloc[-1], freq="MS"
    )
    if len(expected) != len(out) or not (out[DATE_COL].to_numpy() == expected.to_numpy()).all():
        raise PanelValidationError(
            "index is not a contiguous monthly series starting on the first "
            "day of each month; reindex or interpolate before continuing"
        )

    numeric = [TARGET, *EXOG]
    if out[numeric].isna().any().any():
        na_cols = out[numeric].columns[out[numeric].isna().any()].tolist()
        raise PanelValidationError(f"NaN values in: {na_cols}")

    return out


def load_panel(path: str | Path = PANEL_PATH) -> pd.DataFrame:
    """Read and validate the processed monthly panel."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make demo` to generate a synthetic panel, "
            "or build the real one with build_real_panel()."
        )
    return validate_panel(pd.read_csv(path))


def save_panel(df: pd.DataFrame, path: str | Path = PANEL_PATH) -> Path:
    """Validate and persist a panel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_panel(df).to_csv(path, index=False)
    return path


def build_real_panel(raw_dir: str | Path) -> pd.DataFrame:
    """Assemble the real panel from files downloaded into ``raw_dir``.

    Expected files (download manually, respecting each provider's terms):

        cepea_frango.csv   columns: data, valor   (daily)
        cepea_milho.csv    columns: data, valor   (daily)
        cepea_soja.csv     columns: data, valor   (daily)
        bcb_ptax.csv       columns: data, valor   (daily)
        ibge_ipca.csv      columns: data, valor   (monthly)

    Daily series are averaged within the month. The result is validated and
    can be handed straight to ``save_panel``.
    """
    raw_dir = Path(raw_dir)
    spec = {
        TARGET: ("cepea_frango.csv", True),
        "preco_milho": ("cepea_milho.csv", True),
        "preco_soja": ("cepea_soja.csv", True),
        "cambio_usdbrl": ("bcb_ptax.csv", True),
        "ipca": ("ibge_ipca.csv", False),
    }

    frames: list[pd.Series] = []
    for column, (filename, is_daily) in spec.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"expected {path} (see docstring for the layout)")
        serie = pd.read_csv(path)
        serie["data"] = pd.to_datetime(serie["data"])
        serie = serie.set_index("data")["valor"].astype(float)
        monthly = serie.resample("MS").mean() if is_daily else serie.resample("MS").last()
        frames.append(monthly.rename(column))

    panel = pd.concat(frames, axis=1).dropna().reset_index()
    panel = panel.rename(columns={"data": DATE_COL})
    return validate_panel(panel)
