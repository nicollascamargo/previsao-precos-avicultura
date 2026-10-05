"""Request and response contracts for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ObservacaoMensal(BaseModel):
    """One month of observed data."""

    data: str = Field(..., description="Month, ISO format (YYYY-MM-01)", examples=["2024-01-01"])
    preco_frango: float = Field(..., gt=0, description="Broiler price, BRL/kg")
    preco_milho: float = Field(..., gt=0, description="Corn price, BRL per 60kg bag")
    preco_soja: float = Field(..., gt=0, description="Soybean price, BRL per 60kg bag")
    cambio_usdbrl: float = Field(..., gt=0, description="USD/BRL monthly average")
    ipca: float = Field(..., description="IPCA month-over-month variation, %")


class PrevisaoRequest(BaseModel):
    """At least 25 contiguous months are required to fill every lag window."""

    historico: list[ObservacaoMensal] = Field(..., min_length=25)
    horizontes: list[int] | None = Field(
        default=None, description="Horizons in months; defaults to every trained horizon"
    )

    @field_validator("horizontes")
    @classmethod
    def _positive(cls, v: list[int] | None) -> list[int] | None:
        if v is not None and any(h < 1 for h in v):
            raise ValueError("horizons must be >= 1")
        return v


class PrevisaoItem(BaseModel):
    horizonte: int
    data_alvo: str
    preco_previsto: float
    variacao_prevista_pct: float
    modelo: str
    mae_backtest: float | None = None
    mase_backtest: float | None = None


class PrevisaoResponse(BaseModel):
    origem: str
    preco_origem: float
    previsoes: list[PrevisaoItem]
    aviso: str


class ModelCardResponse(BaseModel):
    horizonte: int
    modelo: str
    treinado_em: str
    janela_treino: dict
    hash_dados: str
    metricas_backtest: dict
    principais_variaveis: list[dict]


class HealthResponse(BaseModel):
    status: str
    versao: str
    horizontes_disponiveis: list[int]
